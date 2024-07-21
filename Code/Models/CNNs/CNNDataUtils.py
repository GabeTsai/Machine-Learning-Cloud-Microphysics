import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
sys.path.append('../../')
from CreateDataLists import * # Import functions and variables from CreateDataLists.py
from pathlib import Path

def save_CNN_data(filePath, index_list, data_map):
    '''
    Dump data to JSON files
    '''
    data_map['time'] = cloud_ds['time'].values.tolist()
    data_map['height'] = cloud_ds['z'].values.tolist()
    height_map = {}
    for i in index_list:
        height_map[i] = cloud_ds[var_name[0]].z.values[i]
    with open(filePath + '/dataCNN.json', 'w') as f:
        json.dump(data_map, f)
    with open(filePath + '/heightCNN.json', 'w') as f:
        json.dump(height_map, f)
    
def create_CNN_dataset(data_folder):
    '''
    Create the input and target datasets. Return as torch tensors.
    '''
    with open(data_folder + '/dataCNN.json', 'r') as f:
        data_map = json.load(f)
    qc_autoconv_cloud = np.array(np.transpose(data_map['qc_autoconv_cloud']))
    nc_autoconv_cloud = np.array(np.transpose(data_map['nc_autoconv_cloud']))
    qr_autoconv_cloud = np.array(np.transpose(data_map['qr_autoconv_cloud']))
    nr_autoconv_cloud = np.array(np.transpose(data_map['nr_autoconv_cloud']))
    auto_cldmsink_b_cloud = np.array(np.transpose(data_map['auto_cldmsink_b_cloud']))

    input_data = np.stack((qc_autoconv_cloud, nc_autoconv_cloud, 
                            qr_autoconv_cloud, nr_autoconv_cloud), axis = 1) # (time, 4, height_channels)
    target_data = auto_cldmsink_b_cloud # (time, height_channels)
    
    return torch.FloatTensor(input_data), torch.FloatTensor(target_data)

def main():
    index_list = find_nonzero_threshold(cloud_ds[var_name[0]], THRESHOLD_VALUES)
    data_map = create_data_map(index_list)
    save_CNN_data(data_folder_file_path, index_list, data_map)
    input_data, target_data = create_CNN_dataset(data_folder_file_path)
    print(input_data.shape)
    print(target_data.shape)

if __name__ == "__main__":
    main()