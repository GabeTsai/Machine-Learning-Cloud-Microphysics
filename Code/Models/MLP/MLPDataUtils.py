import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
sys.path.append('../../')
from DataUtils import prepare_datasets, standardize, save_data_info

from Visualizations import histogram, histogram_single

from pathlib import Path

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

    rng = np.random.default_rng(777777)
    indices = rng.permutation(len(input_data))
    
    input_data = input_data[indices]
    target_data = target_data[indices]

    test_percentage = 0.1 # 10% of data used for testing
    test_size = int(len(input_data) * test_percentage)
    train_size = len(input_data) - test_size

    train_input_data = input_data[:train_size]
    train_target_data = target_data[:train_size]

    #Save data for undoing transforms (use train data statistics to prevent leakage)
    save_data_info(train_input_data, train_target_data, model_folder_path, model_name)

    dims = (0)
    test_input_data = standardize(input_data[train_size:])
    test_target_data = standardize(target_data[train_size:])
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_input_data), torch.FloatTensor(test_target_data))
    
    torch.save(test_dataset, Path(model_folder_path)/ f'{model_name}_test_dataset.pth')

    return torch.FloatTensor(standardize(train_input_data)), torch.FloatTensor(standardize(train_target_data)).unsqueeze(1)

def create_MLP_dataset(data_folder_path, model_name, model_folder_path):

    data_maps = prepare_datasets(data_folder_path)
    inputs, targets = concat_data(data_maps, model_name, model_folder_path)
    return inputs, targets

def main():
    model_name = 'MLP3'
    inputs, targets = create_MLP_dataset('../../../Data/NetCDFFiles', model_name, f'../../../SavedModels/{model_name}')
    print(inputs.shape, targets.shape)
    print(np.mean(np.array(inputs), axis = 0))
    # histogram_single(inputs[:, 0], '', model_name, 'qc', '../../../Visualizations')
    # histogram_single(inputs[:, 1], '', model_name, 'nc', '../../../Visualizations')
    # histogram_single(inputs[:, 2], '', model_name, 'tke', '../../../Visualizations')
    # histogram_single(targets, '', model_name, 'target', '../../../Visualizations')

if __name__ == "__main__":
    main()