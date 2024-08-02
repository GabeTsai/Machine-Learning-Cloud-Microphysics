import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
sys.path.append('../../')
from DataUtils import * # Import functions and variables from CreateDataLists.py
from pathlib import Path

def save_CNN_data(filePath, cloud_ds, index_list, data_map):
    '''
    Dump data to JSON files
    '''
    data_map['time'] = cloud_ds['time'].values.tolist()
    data_map['height'] = cloud_ds['z'].values.tolist()
    height_map = {}
    for i in index_list:
        height_map[i] = cloud_ds[var_names[0]].z.values[i]
    with open(filePath / 'dataCNN.json', 'w') as f:
        json.dump(data_map, f)
    with open(filePath / 'heightCNN.json', 'w') as f:
        json.dump(height_map, f)
    
def create_CNN_dataset(data_folder):
    '''
    Create the input and target datasets. Return as torch tensors.
    '''
    with open(data_folder / 'dataCNN.json', 'r') as f:
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
    data_folder_path = '../../../Data'
    data_file_path = Path(data_folder_path) / 'NetCDFFiles' / 'ena25jan2023.nc'
    data_map, index_list, cloud_ds = create_data_map(data_file_path)
    save_CNN_data(Path(data_folder_path) / 'CNNs', cloud_ds, index_list, data_map)
    input_data, target_data = create_CNN_dataset(Path(data_folder_path) / 'CNNs')
    print(input_data.shape)
    print(target_data.shape)

if __name__ == "__main__":
    main()