import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
from DataUtils import prepare_datasets, standardize, save_data_info
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import os
import config

from pathlib import Path

torch.serialization.add_safe_globals([TensorDataset])

def prepare_dataset_MLP(data_map):
    '''
    Create the input and target datasets. Return as np arrays.
    '''
    #Thresh all of the 2D arrays into 1D arrays
    qc_autoconv_cloud = np.array(data_map['qc_autoconv_cloud']).transpose()
    nc_autoconv_cloud = np.array(data_map['nc_autoconv_cloud']).transpose()
    tke_sgs = np.array(data_map['tke_sgs']).flatten().transpose()
    input_data = np.stack((qc_autoconv_cloud, nc_autoconv_cloud, tke_sgs), axis = 1)
    
    auto_cldmsink_b_cloud = np.array(data_map['auto_cldmsink_b_cloud']).flatten().transpose()

    target_data = auto_cldmsink_b_cloud # (time * height_channels)
    return input_data, target_data

def concat_data(data_maps, model_name, model_folder_path, data_name = ''):
    '''
    Concatenate data from data maps into tensors
    :param data_maps: list of data maps
    '''
    from DataUtils import standardize
    input_list = []
    target_list = []

    for data_map in data_maps:
        inputs, targets = prepare_dataset_MLP(data_map) 
        input_list.append(inputs) 
        target_list.append(targets) 

    # Concatenate data
    input_data = np.concatenate(input_list, axis=0)
    target_data = np.concatenate(target_list, axis=0)

    #Filter out physically impossible values
    filter = (target_data > 0) & (input_data[:, 0] > 0) & (input_data[:, 1] > 0) & (input_data[:, 2] > 0)
    input_data = np.log(input_data[filter])
    target_data = np.log(target_data[filter])

    rng = np.random.default_rng(42069)
    indices = rng.permutation(len(input_data))
    
    input_data = input_data[indices]
    target_data = target_data[indices]

    test_percentage = 0.1 # 10% of data used for testing
    test_size = int(len(input_data) * test_percentage)
    train_size = len(input_data) - test_size

    train_input_data = input_data[:train_size]
    train_target_data = target_data[:train_size]

    #Save data for undoing transforms (use train data statistics to prevent leakage)
    save_data_info(train_input_data, train_target_data, model_folder_path, model_name, data_name)

    train_input_data = torch.FloatTensor(standardize(train_input_data))
    train_target_data = torch.FloatTensor(standardize(train_target_data)).unsqueeze(1)

    dims = (0)
    test_input_data = standardize(input_data[train_size:])
    test_target_data = standardize(target_data[train_size:])

    train_dataset = TensorDataset(torch.FloatTensor(train_input_data), torch.FloatTensor(train_target_data))
    test_dataset = TensorDataset(torch.FloatTensor(test_input_data), torch.FloatTensor(test_target_data))
    
    torch.save(test_dataset, Path(model_folder_path) / f'{model_name}_{data_name}_test_dataset.pth')

    return train_dataset

def create_MLP_dataset(data_folder_path, model_name, model_folder_path, subset):
    data_maps = prepare_datasets(data_folder_path)
    train_dataset = concat_data(data_maps, model_name, model_folder_path, subset)
    return train_dataset

def create_ensemble_dataset(data_folder_path, model_name, ensemble_model_name, model_folder_path, regions):
    if model_name != "Ensemble":
        raise ValueError("model_name not Ensemble.")
    train_datasets = []
    test_datasets = []
    for region in regions:
        train_dataset = create_MLP_dataset(data_folder_path, ensemble_model_name, model_folder_path, region)
        train_datasets.append(train_dataset)
        test_datasets.append(torch.load(Path(model_folder_path) \
                            /ensemble_model_name / f'{ensemble_model_name}_{region}_test_dataset.pth', weights_only = True))

    ensemble_train_dataset = ConcatDataset(train_datasets)
    ensemble_test_dataset = ConcatDataset(test_datasets)

    torch.save(ensemble_test_dataset, Path(model_folder_path) / f'{model_name}_test_dataset.pth')

    return ensemble_train_dataset

def main():
    ensemble_model_name = "DeepMLP"
    model_name = "Ensemble" 
    data_folder_path = config.DATA_FOLDER_PATH
    model_folder_path = config.MODEL_FOLDER_PATH
    regions = config.REGIONS
    subset = regions[0]
    create_MLP_dataset(data_folder_path, ensemble_model_name, model_folder_path, subset)
    model_name = "Ensemble"
    create_ensemble_dataset(data_folder_path, model_name, ensemble_model_name, model_folder_path, regions)
    
if __name__ == "__main__":
    main()