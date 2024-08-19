import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
sys.path.append('../../')
from DataUtils import * # Import functions and variables from DataUtils.py
from Visualizations import histogram

torch.manual_seed(3407) #is all you need

def main():
    data_file_path = '../../../Data/00d-03h-00m-00s-000ms.h5'
    log_map = {
        'qc_autoconv': True,
        'nc_autoconv': False,
        'tke_sgs': True,
        'auto_cldmsink_b': True}
    data_map = prepare_hdf_dataset(data_file_path)
    input_data, target_data = create_deep_dataset_subset(data_map, log_map, 0.2)
    print(input_data.shape, target_data.shape)
    histogram(target_data, target_data, 'GEN', 'GENTargetValues', '../../../Visualizations')
    histogram(input_data[:, 0], input_data[:, 0], 'GEN', 'GENqc', '../../../Visualizations')
    histogram(input_data[:, 1], input_data[:, 1], 'GEN', 'GENnc', '../../../Visualizations')
    histogram(input_data[:, 2], input_data[:, 2], 'GEN', 'GENtke', '../../../Visualizations')
    

if __name__ == '__main__':
    main()