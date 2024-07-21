import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
sys.path.append('../../')
from CreateDataLists import * # Import functions and variables from CreateDataLists.py
from pathlib import Path

def create_MLP_dataset(data_map):
    '''
    Create the input and target datasets. Return as torch tensors.
    '''
    #Thresh all of the 2D arrays into 1D arrays
    qc_autoconv_cloud = np.array(data_map['qc_autoconv_cloud']).flatten().transpose()
    nc_autoconv_cloud = np.array(data_map['nc_autoconv_cloud']).flatten().transpose()
    qr_autoconv_cloud = np.array(data_map['qr_autoconv_cloud']).flatten().transpose()
    nr_autoconv_cloud = np.array(data_map['nr_autoconv_cloud']).flatten().transpose()
    auto_cldmsink_b_cloud = np.array(data_map['auto_cldmsink_b_cloud']).flatten().transpose()

    input_data = np.stack((qc_autoconv_cloud, nc_autoconv_cloud, 
                            qr_autoconv_cloud, nr_autoconv_cloud), axis = 1) # (time, 4, height_channels)
    target_data = auto_cldmsink_b_cloud # (time, height_channels)
    
    return torch.FloatTensor(input_data), torch.FloatTensor(target_data)

def main():
    index_list = find_nonzero_threshold(cloud_ds[var_name[0]], THRESHOLD_VALUES)
    data_map = create_data_map(index_list)
    inputs, targets = create_MLP_dataset(data_map)
    print(inputs.shape)
    print(targets.shape)

if __name__ == "__main__":
    main()