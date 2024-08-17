import xarray as xr
import numpy as np
from pathlib import Path
import os
import json

#Open file, initialize data arrays
cloud_var_names = ['qc_autoconv_cloud', 'nc_autoconv_cloud', 'qr_autoconv_cloud', 'nr_autoconv_cloud','auto_cldmsink_b_cloud']
turb_var_names = ['tke_sgs']
THRESHOLD_VALUES = 0.62 * 721
THRESHOLD = 1e-6

def remove_outliers(arr): 
    '''
    Return mask of outliers to be removed from array using 1.5 IQR rule.

    :param arr: Input NumPy array to check for outliers.
    :return: Mask of the same shape as `arr` where outliers are marked as False.
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

    :param data: Data to be normalized.
    :param dims: Dimensions to normalize over. If None, global min and max used to normalize.
    :return: Normalized data.
    '''
    non_zero_mask = data != 0    
    masked_data = np.ma.masked_array(data, mask=~non_zero_mask)

    if dims != None:
        axes = dims
        min_vals = np.min(masked_data, axis = axes, keepdims = True) 
        max_vals = np.max(masked_data, axis= axes, keepdims=True)
    else:
        min_vals = np.min(masked_data)
        max_vals = np.max(masked_data)

    normalized_data = np.where(non_zero_mask, (data - min_vals) / (max_vals - min_vals), 0)
    
    return normalized_data

def min_max_denormalize(data, min_val, max_val):
    '''
    Denormalize single column of data from range [0, 1] to [min_val, max_val].

    :param data: Normalized data to be denormalized.
    :param min_val: Minimum value of the original data range.
    :param max_val: Maximum value of the original data range.
    :return: Denormalized data.
    '''
    return data * (max_val - min_val) + min_val

def find_nonzero_threshold(dataset, num_values):
    '''
    Return list of indices of height levels with more than num_values non-zero values.

    :param dataset: Dataset to search through.
    :param num_values: Threshold number of non-zero values.
    :return: List of indices of height levels meeting the criteria.
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
    Return data array with only the height levels specified in index_list.

    :param dataset: Input dataset from which to extract data.
    :param index_list: List of indices specifying height levels to extract.
    :return: Extracted data array.
    '''
    data_list = []
    for index in index_list:
        data_list.append(dataset[:, index])
    data_array = np.array(data_list)
    return data_array

def log_ignore_zero(arr):
    '''
    Apply log transform to array, ignoring zeros.

    :param arr: Input array to transform.
    :return: Log-transformed array.
    '''
    arr_copy = np.log(arr, out = np.zeros_like(arr, dtype=np.float64), where = (arr > 0))
    arr_copy = np.nan_to_num(arr_copy, nan = 0)
    return arr_copy

def prepare_dataset(dataset, index_list):
    '''
    Log transform, normalize, and return data array.

    :param dataset: Dataset to prepare.
    :param index_list: List of indices specifying height levels to include.
    :return: Prepared data array.
    '''
    dataset_copy = dataset.values
    dataset_copy = extract_data(dataset_copy, index_list)
    dataset_copy = log_ignore_zero(dataset_copy)
    return dataset_copy.tolist()

def create_data_map(data_file_path):
    '''
    Create map to store all data arrays (input and target arrays).

    :param data_file_path: Path to the data file.
    :return: Dictionary containing prepared data arrays.
    '''
    cloud_ds = xr.open_dataset(data_file_path, group = 'DiagnosticsClouds/profiles')
    index_list = find_nonzero_threshold(cloud_ds[cloud_var_names[0]], THRESHOLD_VALUES)
    data_map = {}
    for i in range(len(cloud_var_names)):
        data_map[cloud_var_names[i]] = prepare_dataset(cloud_ds[cloud_var_names[i]], index_list)
    turb_ds = xr.open_dataset(data_file_path, group = 'DiagnosticState/profiles')
    for i in range(len(turb_var_names)):
        data_map[turb_var_names[i]] = prepare_dataset(turb_ds[turb_var_names[i]], index_list)
    return data_map

def prepare_datasets(data_folder_path):
    '''
    Return a list of logged, un-normalized data maps for each NetCDF file in the data folder.

    :param data_folder_path: Path to the folder containing data files.
    :return: List of data maps.
    '''
    data_maps = []
    data_list = os.listdir(data_folder_path)
    print(len(data_list))
    for data_file in data_list:
        data_file_path = Path(data_folder_path) / str(data_file)
        data_map = create_data_map(data_file_path)
        data_maps.append(data_map)
    
    return data_maps

def save_data_info(inputs, targets, model_folder_path, model_name):
    '''
    Save data information (mean, min, max) to JSON files. 
    Needed for rescaling model outputs to real-life interpretable values. 

    :param inputs: Input data array.
    :param targets: Target data array.
    :param model_folder_path: Path to the folder to save JSON files.
    :param model_name: Name of the model to use in filenames.
    '''

    input_data_map = {}
    
    target_data_map = {}
    non_zero_mask = targets != 0
    masked_targets = targets[non_zero_mask]
    target_data_map['mean'] = np.mean(targets)
    target_data_map['min'] = np.min(masked_targets)
    target_data_map['max'] = np.max(masked_targets)
    
    qc = inputs[:, 0]
    nc = inputs[:, 1]
    tke_sgs = inputs[:, 2]
    masked_qc = qc[qc !=0]
    masked_nc = nc[nc != 0]
    masked_tke_sgs = tke_sgs[tke_sgs != 0]
    input_data_map['qc'] = {'min': np.min(masked_qc), 'max': np.max(masked_qc)}
    input_data_map['nc'] = {'min': np.min(masked_nc), 'max': np.max(masked_nc)}
    input_data_map['tke_sgs'] = {'min': np.min(masked_tke_sgs), 'max': np.max(masked_tke_sgs)}

    with open(Path(model_folder_path) / f'{model_name}_target_data_map.json', 'w') as f:
        json.dump(target_data_map, f)

    with open(Path(model_folder_path) / f'{model_name}_input_data_map.json', 'w') as f:
        json.dump(input_data_map, f)