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
torch.manual_seed(config.SEED)

def prepare_dataset_MLP(data_map):
    '''
    Create the input and target datasets. Return as np arrays.
    '''
    #Thresh all of the 2D arrays into 1D arrays
    qc_autoconv_cloud = np.array(data_map['qc_autoconv_cloud']).transpose()
    nc_autoconv_cloud = np.array(data_map['nc_autoconv_cloud']).transpose()
    tke_sgs = np.array(data_map['tke_sgs']).transpose()
    input_data = np.stack((qc_autoconv_cloud, nc_autoconv_cloud, tke_sgs), axis = 1)
    
    auto_cldmsink_b_cloud = np.array(data_map['auto_cldmsink_b_cloud']).flatten().transpose()

    target_data = auto_cldmsink_b_cloud # (time * height_channels)
    region_tags = np.full((len(target_data),), data_map["region"])
    return input_data, target_data, region_tags

def concat_data(data_maps, model_name, model_folder_path, data_name = '', standardize = True):
    '''
    Concatenate data from data maps into tensors
    :param data_maps: list of data maps
    '''
    from DataUtils import standardize
    input_list, target_list, region_list = [], [], []

    for data_map in data_maps:
        inputs, targets, region_tags = prepare_dataset_MLP(data_map)
        input_list.append(inputs)
        target_list.append(targets)
        region_list.append(region_tags)

    # Concatenate data
    input_data = np.concatenate(input_list, axis=0)
    target_data = np.concatenate(target_list, axis=0)
    region_tags = np.concatenate(region_list, axis=0)

    # Filter out physically impossible values
    filter = (target_data > 0) & (input_data[:, 0] > 0) & (input_data[:, 1] > 0) & (input_data[:, 2] > 0)
    input_data = np.log(input_data[filter])
    target_data = np.log(target_data[filter])
    region_tags = region_tags[filter]  # Filter regions similarly

    rng = np.random.default_rng(config.SEED)
    indices = rng.permutation(len(input_data))

    input_data = input_data[indices]
    target_data = target_data[indices]
    region_tags = region_tags[indices]

    test_percentage = 0.1  # 10% of data used for testing
    val_percentage = 0.2
    test_size = int(len(input_data) * test_percentage)
    val_size = int(len(input_data) * val_percentage)
    train_size = len(input_data) - test_size - val_size

    train_input_data = input_data[:train_size]
    train_target_data = target_data[:train_size]
    train_region_tags = region_tags[:train_size]

    val_input_data = input_data[train_size:train_size + val_size]
    val_target_data = target_data[train_size:train_size + val_size]
    val_region_tags = region_tags[train_size:train_size + val_size]

    test_input_data = input_data[train_size + val_size:]
    test_target_data = target_data[train_size + val_size:]
    test_region_tags = region_tags[train_size + val_size:]  # Retain region tags for test set

    # Save data for undoing transforms
    save_data_info(train_input_data, train_target_data, model_folder_path, model_name, data_name)

    if standardize:
        train_input_mean = np.mean(train_input_data, axis=0)
        train_target_mean = np.mean(train_target_data)

        train_input_std = np.std(train_input_data, axis=0)
        train_target_std = np.std(train_target_data)

        train_input_data = standardize(train_input_data, dims=(0))
        train_target_data = standardize(train_target_data)

        val_input_data = standardize(val_input_data, train_input_mean, train_input_std, dims=(0))
        val_target_data = standardize(val_target_data, train_target_mean, train_target_std)

        test_input_data = standardize(test_input_data, train_input_mean, train_input_std, dims=(0))
        test_target_data = standardize(test_target_data, train_target_mean, train_target_std)

    print(train_input_data.shape, train_target_data.shape, val_input_data.shape, val_target_data.shape, test_input_data.shape, test_target_data.shape)

    # Return region tags for test data
    return (train_input_data, train_target_data, train_region_tags,
            val_input_data, val_target_data, val_region_tags,
            test_input_data, test_target_data, test_region_tags)

def create_MLP_dataset(data_folder_path, model_name, model_folder_path, subset, standardize = True):
    data_maps = prepare_datasets(data_folder_path)
    train_input_data, train_target_data, train_region_tags, \
    val_input_data, val_target_data, val_region_tags, \
    test_input_data, test_target_data, test_region_tags = concat_data(
        data_maps, model_name, model_folder_path, data_name=subset, standardize=standardize)

    train_dataset = TensorDataset(torch.FloatTensor(train_input_data), 
                                  torch.FloatTensor(train_target_data).unsqueeze(1),
                                  torch.tensor(train_region_tags))
    val_dataset = TensorDataset(torch.FloatTensor(val_input_data), 
                                torch.FloatTensor(val_target_data).unsqueeze(1), 
                                torch.tensor(val_region_tags))
    test_dataset = TensorDataset(torch.FloatTensor(test_input_data), 
                                  torch.FloatTensor(test_target_data).unsqueeze(1), 
                                  torch.tensor(test_region_tags))
    print(test_region_tags.shape)
    torch.save(test_dataset, Path(model_folder_path) / f'{model_name}_{subset}_test_dataset.pth')

    return train_dataset, val_dataset, test_dataset

def create_MLP_dataset_split(data_folder_path, model_name, model_folder_path, regions, standardize = True):
    train_datasets, val_datasets, test_datasets = [], [], []
    for region in regions:
        print(f"{data_folder_path}/{region}")
        datasets = create_MLP_dataset(f"{data_folder_path}/{region}", model_name, model_folder_path, region, standardize = standardize)
        train_datasets.append(datasets[0])
        val_datasets.append(datasets[1])
        test_datasets.append(datasets[2])
    
    full_train_dataset = ConcatDataset(train_datasets)
    full_val_dataset = ConcatDataset(val_datasets)
    full_test_dataset = ConcatDataset(test_datasets)
    
    print(model_folder_path)
    torch.save(full_test_dataset, Path(model_folder_path) / f'{model_name}_all_split_test_dataset.pth')
    
    return full_train_dataset, full_val_dataset
    
def create_ensemble_dataset(data_folder_path, model_name, ensemble_model_name, model_folder_path, regions):
    if model_name != "Ensemble":
        raise ValueError("model_name not Ensemble.")
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for region in regions:
        train_dataset, val_dataset = create_MLP_dataset(f'{data_folder_path}/{region}', ensemble_model_name, model_folder_path, region, standardize = False)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        print(data_folder_path, region)
        test_datasets.append(torch.load(Path(model_folder_path) \
                            / f'{ensemble_model_name}_{region}_test_dataset.pth', weights_only = True))
        
    # Concatenate datasets
    ensemble_train_dataset = ConcatDataset(train_datasets)
    ensemble_val_dataset = ConcatDataset(val_datasets)
    ensemble_test_dataset = ConcatDataset(test_datasets)

    # Access inputs and targets from each dataset
    train_inputs = torch.cat([d.tensors[0] for d in train_datasets], dim=0)
    train_targets = torch.cat([d.tensors[1] for d in train_datasets], dim=0)

    save_data_info(np.array(train_inputs), np.array(train_targets), model_folder_path, model_name)

    val_inputs = torch.cat([d.tensors[0] for d in val_datasets], dim=0)
    val_targets = torch.cat([d.tensors[1] for d in val_datasets], dim=0)

    test_inputs = torch.cat([d.tensors[0] for d in test_datasets], dim=0)
    test_targets = torch.cat([d.tensors[1] for d in test_datasets], dim=0)

    # Calculate mean and std from the training data
    train_input_mean = torch.mean(train_inputs, dim=0)
    train_input_std = torch.std(train_inputs, dim=0)

    train_target_mean = torch.mean(train_targets, dim=0)
    train_target_std = torch.std(train_targets, dim=0)

    train_inputs = standardize(train_inputs, train_input_mean, train_input_std)
    train_targets = standardize(train_targets, train_target_mean, train_target_std)

    val_inputs = standardize(val_inputs, train_input_mean, train_input_std)
    val_targets = standardize(val_targets, train_target_mean, train_target_std)

    test_inputs = standardize(test_inputs, train_input_mean, train_input_std)
    test_targets = standardize(test_targets, train_target_mean, train_target_std)

    ensemble_train_dataset = TensorDataset(train_inputs, train_targets)
    ensemble_val_dataset = TensorDataset(val_inputs, val_targets)
    ensemble_test_dataset = TensorDataset(test_inputs, test_targets)

    # Save the test dataset
    torch.save(ensemble_test_dataset, Path(model_folder_path) / f'{model_name}_test_dataset.pth')

    return ensemble_train_dataset, ensemble_val_dataset

def main():
    model_name = "DeepMLP" 

    regions = config.REGIONS

    model_folder_path = f"{config.MODEL_FOLDER_PATH}/{model_name}"
    
    create_MLP_dataset(f"{config.DATA_FOLDER_PATH}/all", model_name, model_folder_path, 'all', standardize = True)
        
    
if __name__ == "__main__":
    main()