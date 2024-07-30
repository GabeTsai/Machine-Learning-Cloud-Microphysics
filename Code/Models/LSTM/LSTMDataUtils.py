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
from sklearn.preprocessing import RobustScaler, QuantileTransformer

def prepare_dataset_LSTM(data_map, min_seq_length, max_seq_length, max_zeros=100):
    '''
    Return input of seq_length min to max seq_length and target datasets for LSTM model. 
    '''
    qc_autoconv_cloud = np.array(data_map['qc_autoconv_cloud'])
    nc_autoconv_cloud = np.array(data_map['nc_autoconv_cloud'])
    auto_cldmsink_b_cloud = np.array(data_map['auto_cldmsink_b_cloud'])

    inputs = []
    targets = []
    for seq_length in range(min_seq_length, max_seq_length + 1):  # for all seq lengths,
        for i in range(len(qc_autoconv_cloud[0]) - seq_length):  # apply a sliding window of size seq_length to extract input sequences
            padding = max_seq_length - seq_length
            qc_seq = qc_autoconv_cloud[:, i:i + seq_length]
            nc_seq = nc_autoconv_cloud[:, i:i + seq_length]
            target = auto_cldmsink_b_cloud[:, i + seq_length]

            if padding:  # pad to max_seq_length
                qc_seq = np.concatenate((qc_seq, np.zeros((qc_seq.shape[0], padding))), axis=1)
                nc_seq = np.concatenate((nc_seq, np.zeros((nc_seq.shape[0], padding))), axis=1)
            
            sequence = np.stack((qc_seq, nc_seq), axis=1)  # (height_channels, num_features, seq_length)

            inputs.append(sequence)  # (height_channels, seq_length, num_features)
            targets.append(target)  # (height_channels)        

    input_data = np.concatenate(inputs, axis=0)
    target_data = np.concatenate(targets, axis=0)

    return input_data, target_data

def concat_data(data_maps, model_folder_path, min_seq_length, max_seq_length, model_name):
    input_list = []
    target_list = []

    for data_map in data_maps:
        inputs, targets = prepare_dataset_LSTM(data_map, min_seq_length, max_seq_length)
        input_list.append(inputs) 
        target_list.append(targets)

    input_data = np.concatenate(input_list, axis=0)
    target_data = np.concatenate(target_list, axis=0)

    input_data = np.transpose(input_data, (0, 2, 1))

    filter_mask = remove_outliers(target_data)
    target_data = target_data[filter_mask]
    input_data = input_data[filter_mask]

    save_data_info(input_data, target_data, model_folder_path, model_name)
    # robust_scaler = RobustScaler()
    # target_data = robust_scaler.fit_transform(np.expand_dims(target_data, axis = 1))
    input_data = min_max_normalize(input_data, model_name)
    target_data = min_max_normalize(target_data)
    
    return torch.FloatTensor(input_data), torch.FloatTensor(target_data)

def create_LSTM_dataset(data_folder_path, model_folder_path, model_name):
    min_seq_length = 8
    max_seq_length = 16
    data_maps = prepare_datasets(data_folder_path)
    inputs, targets = concat_data(data_maps, model_folder_path, min_seq_length, max_seq_length, model_name)
    return inputs, targets

def main():
    data_folder_path = '../../../Data/NetCDFFiles'
    model_folder_path = '../../../SavedModels/LSTM'
    
    inputs, targets = create_LSTM_dataset(data_folder_path, model_folder_path, 'LSTM')
    print(inputs.shape)
    print(targets.shape)
    histogram(targets, targets, 'LSTM', '../../../Visualizations')


if __name__ == "__main__":
    main()