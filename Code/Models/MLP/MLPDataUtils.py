import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
sys.path.append('../../')
from CreateDataLists import * # Import functions and variables from CreateDataLists.py
from Visualizations import histogram
from pathlib import Path

def prepare_dataset_MLP(data_map, include_qr_nr = True):
    '''
    Create the input and target datasets. Return as np arrays.
    '''
    #Thresh all of the 2D arrays into 1D arrays
    qc_autoconv_cloud = np.array(data_map['qc_autoconv_cloud']).flatten().transpose()
    nc_autoconv_cloud = np.array(data_map['nc_autoconv_cloud']).flatten().transpose()
    if include_qr_nr:
        qr_autoconv_cloud = np.array(data_map['qr_autoconv_cloud']).flatten().transpose()
        nr_autoconv_cloud = np.array(data_map['nr_autoconv_cloud']).flatten().transpose()
        auto_cldmsink_b_cloud = np.array(data_map['auto_cldmsink_b_cloud']).flatten().transpose()
        input_data = np.stack((qc_autoconv_cloud, nc_autoconv_cloud, 
                                qr_autoconv_cloud, nr_autoconv_cloud), axis = 1)
    else:
        input_data = np.stack((qc_autoconv_cloud, nc_autoconv_cloud), axis = 1)
    
    auto_cldmsink_b_cloud = np.array(data_map['auto_cldmsink_b_cloud']).flatten().transpose()

    target_data = auto_cldmsink_b_cloud # (time * height_channels)

    print(input_data.shape)
    return input_data, target_data

def concat_data(data_maps, model_folder_path):
    '''
    Concatenate data from data maps into tensors
    :param data_maps: list of data maps
    '''
    input_list = []
    target_list = []

    for data_map in data_maps:
        inputs, targets = prepare_dataset_MLP(data_map, False) 
        input_list.append(inputs) 
        target_list.append(targets) 

    # Concatenate the data
    input_data = np.concatenate(input_list, axis=0)
    target_data = np.concatenate(target_list, axis=0)

    #Remove outliers
    print(target_data.shape)
    target_data_mask = target_data != np.max(target_data)
    target_data = target_data[target_data_mask]
    input_data = input_data[target_data_mask]

    save_data_info(input_data, target_data, model_folder_path, 'MLP')
    
    input_data = min_max_normalize(input_data)
    target_data = min_max_normalize(target_data)

    print(input_data.shape)
    print(target_data.shape)

    return torch.FloatTensor(input_data), torch.FloatTensor(target_data)

def create_MLP_dataset(data_folder_path, model_folder_path):
    data_maps = prepare_datasets(data_folder_path)
    inputs, targets = concat_data(data_maps, model_folder_path)
    return inputs, targets

def main():
    model_name = 'MLP'
    create_MLP_dataset('../../../Data/NetCDFFiles', f'../../../SavedModels/{model_name}')

if __name__ == "__main__":
    main()