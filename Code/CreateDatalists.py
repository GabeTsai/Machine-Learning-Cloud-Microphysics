import xarray as xr
import numpy as np
from pathlib import Path
import os
import json

#Open file, initialize data arrays
var_names = ['qc_autoconv_cloud', 'nc_autoconv_cloud', 'qr_autoconv_cloud', 'nr_autoconv_cloud', 'auto_cldmsink_b_cloud']
log_list = [False, False, True, True, True] # True if log transformation is needed

THRESHOLD_VALUES = 0.62 * 721
THRESHOLD = 1e-6

def min_max_normalize(data):
    '''
    Normalize data to range [0, 1]
    '''
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def min_max_denormalize(data, min_val, max_val):
    '''
    Denormalize data from range [0, 1] to [min_val, max_val]
    '''
    return data * (max_val - min_val) + min_val

def find_nonzero_threshold(dataset, num_values):
    '''
    Return list of indices of height levels with more than num_values non-zero values
    '''
    index_list = []
    for i in range(dataset.shape[1]):
        array = dataset.isel(z = i).data
        count = np.count_nonzero(array >= THRESHOLD)
        if count > num_values:
            index_list.append(i)
    return index_list

def extract_data(dataset, index_list):
    '''
    Return data array with only the height levels specified in index_list
    '''
    data_list = []
    for index in index_list:
        data_list.append(dataset[:, index])
    data_array = np.array(data_list)
    return data_array

def prepare_dataset(dataset, log, index_list):
    '''
    Log transform, normalize, and return data array
    '''
    dataset_copy = dataset.values
    if log:
        dataset_copy = np.log(dataset_copy, out = np.zeros_like(dataset_copy, dtype=np.float64), where = (dataset_copy > 0))
        dataset_copy = np.nan_to_num(dataset_copy, nan = 0)

    data = extract_data(dataset_copy, index_list)
    # data = min_max_normalize(data)
    return data.tolist()

def create_data_map(data_file_path):
    '''
    Create map to store all data arrays
    '''
    cloud_ds = xr.open_dataset(data_file_path, group = 'DiagnosticsClouds/profiles')
    index_list = find_nonzero_threshold(cloud_ds[var_names[0]], THRESHOLD_VALUES)
    data_map = {}
    for i in range(len(var_names)):
        data_map[var_names[i]] = prepare_dataset(cloud_ds[var_names[i]], log_list[i], index_list)
    return data_map, index_list, cloud_ds

def prepare_datasets(data_folder_path):
    '''
    Return a list of logged, un-normalized data maps for each NetCDF file in the data folder
    '''
    data_maps = []
    data_list = os.listdir(data_folder_path)
    print(len(data_list))
    for data_file in data_list:
        data_file_path = Path(data_folder_path) / str(data_file)
        data_map, index_list, cloud_ds = create_data_map(data_file_path)
        data_maps.append(data_map)
    
    return data_maps

def save_data_info(inputs, targets, model_folder_path, model_name):
    input_data_map = {}
    input_data_map['qc'] = {'min': np.min(inputs[:, 0]), 'max': np.max(inputs[:, 0])}
    input_data_map['nc'] = {'min': np.min(inputs[:, 1]), 'max': np.max(inputs[:, 1])}
    input_data_map['qr'] = {'min': np.min(inputs[:, 2]), 'max': np.max(inputs[:, 2])}
    input_data_map['nr'] = {'min': np.min(inputs[:, 3]), 'max': np.max(inputs[:, 3])}

    target_data_map = {}
    target_data_map['mean'] = np.mean(targets)
    target_data_map['min'] = np.min(targets)
    target_data_map['max'] = np.max(targets)

    with open(Path(model_folder_path) / f'{model_name}_target_data_map.json', 'w') as f:
        json.dump(target_data_map, f)

    with open(Path(model_folder_path) / f'{model_name}_input_data_map.json', 'w') as f:
        json.dump(input_data_map, f)