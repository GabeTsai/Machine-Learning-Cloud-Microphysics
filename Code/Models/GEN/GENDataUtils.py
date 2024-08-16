import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
sys.path.append('../../')
from DataUtils import * # Import functions and variables from DataUtils.py
from Visualizations import histogram

def createGENDataset(datamap, log_map):
    '''
    Extract arrays for GEN model.

    Args:
        datamap (dict): Dictionary mapping variable names to data arrays.
        log_list (dict): Dictionary mapping variable names to boolean values indicating whether to log-transform the data.
    
    Returns:
        tuple: Tuple containing transformed input and target data arrays.
    '''
    
    for key in datamap:
        datamap[key] = datamap[key].flatten().transpose()
    
    data_list = [
        datamap['qc_autoconv'], 
        datamap['nc_autoconv'], 
        datamap['tke_sgs'], 
        datamap['auto_cldmsink_b']
    ]
    
    # Create a filter mask for non-outliers across all relevant variables
    filter_mask = (
        remove_outliers(data_list[-1]) &  # auto_cldmsink_b
        remove_outliers(data_list[2]) &   # tke_sgs
        remove_outliers(data_list[1]) &   # nc_autoconv
        remove_outliers(data_list[0])     # qc_autoconv
    )
    
    # Apply the filter mask and log transform where necessary
    for i, (data, key) in enumerate(zip(data_list, log_map)):
        data_list[i] = data[filter_mask]
        if log_map[key]:
            data_list[i] = np.log1p(data_list[i])
    
    input_data = np.stack(data_list, axis=1)
    input_data = standardize(input_data, (0))
    target_data = standardize(data_list[-1])

    return input_data, target_data

def main():
    data_file_path = '../../../Data/00d-03h-00m-00s-000ms.h5'
    log_map = {
        'qc_autoconv': True,
        'nc_autoconv': False,
        'tke_sgs': True,
        'auto_cldmsink_b': True}
    data_map = prepare_hdf_dataset(data_file_path)
    input_data, target_data = createGENDataset(data_map, log_map)
    print(input_data.shape, target_data.shape)
    histogram(target_data, target_data, 'GEN', 'GENTargetValues', '../../../Visualizations')
    histogram(input_data[:, 0], input_data[:, 0], 'GEN', 'GENqc', '../../../Visualizations')
    histogram(input_data[:, 1], input_data[:, 1], 'GEN', 'GENnc', '../../../Visualizations')
    histogram(input_data[:, 2], input_data[:, 2], 'GEN', 'GENtke', '../../../Visualizations')
    

if __name__ == '__main__':
    main()