import xarray as xr
import numpy as np
from pathlib import Path
import os
import json

#Open file, initialize data arrays
var_names = ['qc_autoconv_cloud', 'nc_autoconv_cloud', 'qr_autoconv_cloud', 'nr_autoconv_cloud', 'auto_cldmsink_b_cloud']

THRESHOLD_VALUES = 0.75 * 721
THRESHOLD = 1e-6

def remove_outliers(arr): 
    '''
    Return mask of outliers to be removed from array using 1.5 IQR rule
    '''
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (arr > lower_bound) & (arr < upper_bound)

def min_max_normalize(data, dims = None):
    '''
    Normalize data to range [0, 1], feature-specific or global. Ignore zeros.
    '''
    non_zero_mask = data != 0    
    masked_data = np.ma.masked_array(data, mask=~non_zero_mask)

    if dims != None:
        axes = dims
        min_vals = np.min(masked_data, axis = axes, keepdims = True) #normalize features over all batches/seqs
        max_vals = np.max(masked_data, axis= axes, keepdims=True)
    else:
        min_vals = np.min(masked_data)
        max_vals = np.max(masked_data)

    normalized_data = np.where(non_zero_mask, (data - min_vals) / (max_vals - min_vals), 0)
    
    return normalized_data

def min_max_denormalize(data, min_val, max_val):
    '''
    Denormalize single column of data from range [0, 1] to [min_val, max_val]
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

def log_ignore_zero(arr):
    arr_copy = np.log(arr, out = np.zeros_like(arr, dtype=np.float64), where = (arr > 0))
    arr_copy = np.nan_to_num(arr_copy, nan = 0)
    return arr_copy

def prepare_dataset(dataset, index_list):
    '''
    Log transform, normalize, and return data array
    '''
    dataset_copy = dataset.values
    dataset_copy = log_ignore_zero(dataset_copy)
    data = extract_data(dataset_copy, index_list)
    return data.tolist()

def create_data_map(data_file_path):
    '''
    Create map to store all data arrays
    '''
    cloud_ds = xr.open_dataset(data_file_path, group = 'DiagnosticsClouds/profiles')
    index_list = find_nonzero_threshold(cloud_ds[var_names[0]], THRESHOLD_VALUES)
    data_map = {}
    for i in range(len(var_names)):
        data_map[var_names[i]] = prepare_dataset(cloud_ds[var_names[i]], index_list)
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
    
    target_data_map = {}
    non_zero_mask = targets != 0
    masked_targets = targets[non_zero_mask]
    target_data_map['mean'] = np.mean(targets)
    target_data_map['min'] = np.min(masked_targets)
    target_data_map['max'] = np.max(masked_targets)
    
    qc = inputs[:, 0]
    nc = inputs[:, 1]
    qc_nonzero_mask = qc != 0
    masked_qc = qc[qc_nonzero_mask]
    nc_nonzero_mask = nc != 0
    masked_nc = nc[nc_nonzero_mask]
    input_data_map['qc'] = {'min': np.min(masked_qc), 'max': np.max(masked_qc)}
    input_data_map['nc'] = {'min': np.min(masked_nc), 'max': np.max(masked_nc)}

    with open(Path(model_folder_path) / f'{model_name}_target_data_map.json', 'w') as f:
        json.dump(target_data_map, f)

    with open(Path(model_folder_path) / f'{model_name}_input_data_map.json', 'w') as f:
        json.dump(input_data_map, f)