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

def prepare_dataset_LSTM(data_map, seq_length):
    '''
    Return input and target datasets for LSTM model. 
    '''
    qc_autoconv_cloud = np.array(data_map['qc_autoconv_cloud'])
    nc_autoconv_cloud = np.array(data_map['nc_autoconv_cloud'])
    auto_cldmsink_b_cloud = np.array(data_map['auto_cldmsink_b_cloud'])
    inputs = []
    targets = []

    for i in range(len(qc_autoconv_cloud) - seq_length): #apply a sliding window of size seq_length to extract input sequences
        sequence = np.stack((qc_autoconv_cloud[:, i:i+seq_length], nc_autoconv_cloud[:, i:i+seq_length]), axis = 1) #(height_channels, num_features, seq_length)
        inputs.append(sequence) #(height_channels, seq_length, num_features)
        targets.append(auto_cldmsink_b_cloud[:, i+seq_length]) #(height_channels)
    
    input_data = np.concatenate(inputs, axis = 0)
    target_data = np.concatenate(targets, axis = 0)

    return input_data, target_data

def concat_data(data_maps, model_folder_path, seq_length, model_name):
    input_list = []
    target_list = []

    for data_map in data_maps:
        inputs, targets = prepare_dataset_LSTM(data_map, seq_length)
        input_list.append(inputs) 
        target_list.append(targets)

    input_data = np.concatenate(input_list, axis=0)
    target_data = np.concatenate(target_list, axis=0)

    input_data = np.transpose(input_data, (0, 2, 1))

    save_data_info(input_data, target_data, model_folder_path, model_name)

    input_data = min_max_normalize(input_data, model_name)
    target_data = min_max_normalize(target_data)

    filter_mask = remove_outliers(target_data)
    target_data = target_data[filter_mask]
    input_data = input_data[filter_mask]
    
    return torch.FloatTensor(input_data), torch.FloatTensor(target_data)

def create_LSTM_dataset(data_folder_path, model_folder_path, seq_length, model_name):
    data_maps = prepare_datasets(data_folder_path)
    inputs, targets = concat_data(data_maps, model_folder_path, seq_length, model_name)
    return inputs, targets

def main():
    data_folder_path = '../../../Data/NetCDFFiles'
    model_folder_path = '../../../SavedModels/LSTM'
    seq_length = 8
    inputs, targets = create_LSTM_dataset(data_folder_path, model_folder_path, seq_length, 'LSTM')
    print(inputs.shape)
    print(targets.shape)

if __name__ == "__main__":
    main()