import xarray as xr
import numpy as np
import torch
import h5py
from pathlib import Path
import os
import json

#Open file, initialize data arrays
nc_cloud_var_names = ['qc_autoconv_cloud', 'nc_autoconv_cloud','auto_cldmsink_b_cloud']
hdf_cloud_var_names = ['qc_autoconv', 'nc_autoconv', 'auto_cldmsink_b']
turb_var_names = ['tke_sgs']
NC_THRESHOLD_VALUES = 0.62 * 721
HDF_THRESHOLD_PERCENTAGE = 0.95
THRESHOLD = 1e-6

def remove_outliers(arr):
    """
    Return mask of outliers to be removed from array using 1.5 IQR rule.

    Args:
        arr (np.array): Input NumPy array to check for outliers.

    Returns:
        np.array: Mask of the same shape as `arr` where outliers are marked as False.
    """
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return (arr > lower_bound) & (arr < upper_bound)

def min_max_normalize(data, dims=None):
    """
    Normalize data to range [0, 1], feature-specific or global. Ignore zeros.

    Args:
        data (np.array): Data to be normalized.
        dims (tuple, optional): Dimensions to normalize over. If None, global min and max used to normalize.

    Returns:
        np.array: Normalized data.
    """
    non_zero_mask = data != 0    
    masked_data = np.ma.masked_array(data, mask=~non_zero_mask)

    if dims is not None:
        axes = dims
        min_vals = np.min(masked_data, axis=axes, keepdims=True)
        max_vals = np.max(masked_data, axis=axes, keepdims=True)
    else:
        min_vals = np.min(masked_data)
        max_vals = np.max(masked_data)

    normalized_data = np.where(non_zero_mask, (data - min_vals) / (max_vals - min_vals), 0)
    
    return normalized_data

def min_max_denormalize(data, min_val, max_val):
    """
    Denormalize single column of data from range [0, 1] to [min_val, max_val].

    Args:
        data (np.array): Normalized data to be denormalized.
        min_val (float): Minimum value of the original data range.
        max_val (float): Maximum value of the original data range.

    Returns:
        np.array: Denormalized data.
    """
    return data * (max_val - min_val) + min_val

def standardize(data, dims = None):
    """
    Standardize data to have mean 0 and standard deviation 1.

    Args:
        data (np.array): Data to standardize.
        dims (tuple, optional): Dimensions to standardize over. If None, global mean and std used to standardize.

    Returns:
        np.array: Standardized data with mean 0 and standard deviation 1.
    """

    if dims is not None:
        axes = dims
        mean = np.mean(data, axis=axes, keepdims=True)
        std = np.std(data, axis=axes, keepdims=True)
    else:
        mean = np.mean(data)
        std = np.std(data)
    return (data - mean) / std

def find_nonzero_threshold(dataset, num_values, hdf=False):
    """
    Return list of indices of height levels with more than num_values non-zero values.

    Args:
        dataset (ndarray): Dataset to search through.
        num_values (int): Threshold number of non-zero values.
        hdf (bool, optional): Boolean flag indicating whether dataset is from HDF5 file. Defaults to False.

    Returns:
        list: List of indices of height levels meeting the criteria.
    """
    index_list = []
    if hdf:
        print(dataset.shape)
        for i in range(dataset.shape[2]):
            if np.count_nonzero(dataset[:, :, i] >= THRESHOLD) > num_values:
                index_list.append(i)
    else:
        for i in range(dataset.shape[1]):
            array = dataset.isel(z=i).data
            count = np.count_nonzero(array >= THRESHOLD)
            if count > num_values:
                index_list.append(i)
    return index_list

def extract_data(dataset, index_list, dim=1):
    """
    Return data array with only the height levels specified in index_list.

    Args:
        dataset (np.array): Input dataset from which to extract data.
        index_list (list): List of indices specifying height levels to extract.
        dim (int, optional): Dimension along which to extract data. Defaults to 1.

    Returns:
        np.array: Extracted data array.
    """
    data_list = []
    for index in index_list:
        data_list.append(np.take(dataset, index, axis=dim))
    data_array = np.array(data_list)
    return data_array

def log_ignore_zero(arr):
    """
    Apply log transform to array, ignoring zeros.

    Args:
        arr (np.array): Input array to transform.

    Returns:
        np.array: Log-transformed array.
    """
    arr_copy = np.log(arr, out=np.zeros_like(arr, dtype=np.float64), where=(arr > 0))
    arr_copy = np.nan_to_num(arr_copy, nan=0)
    return arr_copy

def prepare_dataset(dataset, index_list):
    """
    Log transform, normalize, and return data array.

    Args:
        dataset (xarray.Dataset): Dataset to prepare.
        index_list (list): List of indices specifying height levels to include.

    Returns:
        list: Prepared data array.
    """
    dataset_copy = dataset.values
    dataset_copy = extract_data(dataset_copy, index_list)
    dataset_copy = log_ignore_zero(dataset_copy)
    return dataset_copy.tolist()

def create_data_map(data_file_path, hdf=False):
    """
    Create map to store all data arrays (input and target arrays).

    Args:
        data_file_path (str): Path to the data file.
        hdf (bool, optional): Flag indicating if the file is in HDF format. Defaults to False.

    Returns:
        dict: Dictionary containing prepared data arrays.
    """
    cloud_ds = xr.open_dataset(data_file_path, group='DiagnosticsClouds/profiles')
    index_list = find_nonzero_threshold(cloud_ds[nc_cloud_var_names[0]], NC_THRESHOLD_VALUES)
    data_map = {}
    for i in range(len(nc_cloud_var_names)):
        data_map[nc_cloud_var_names[i]] = prepare_dataset(cloud_ds[nc_cloud_var_names[i]], index_list)
    turb_ds = xr.open_dataset(data_file_path, group='DiagnosticState/profiles')
    for i in range(len(turb_var_names)):
        data_map[turb_var_names[i]] = prepare_dataset(turb_ds[turb_var_names[i]], index_list)
    return data_map

def prepare_datasets(data_folder_path):
    """
    Return a list of logged, un-normalized data maps for each NetCDF file in the data folder.

    Args:
        data_folder_path (str): Path to the folder containing data files.

    Returns:
        list: List of data maps.
    """
    data_maps = []
    data_list = os.listdir(data_folder_path)
    for data_file in data_list:
        data_file_path = Path(data_folder_path) / str(data_file)
        data_map = create_data_map(data_file_path)
        data_maps.append(data_map)
    
    return data_maps

def load_hdf_dataset(dataset_name, index_list, f):
    """
    Load dataset from HDF5 file.

    Args:
        dataset_name (str): Name of the dataset to load.
        f (h5py.File): Open HDF5 file object.
        index_list (list): List of indices specifying which height levels to include.

    Returns:
        np.array: Loaded dataset.
    """
    dataset = extract_data(np.squeeze(f[dataset_name]), index_list, dim = 2)
    return dataset

def prepare_hdf_dataset(data_file_path):
    """
    Return unprocessed data map (dataset_name : array) for HDF5 file. 

    Args:
        data_file_path (str): Path to the HDF5 file.

    Returns:
        dict: Data map.
    """
    
    data_map = {}
    with h5py.File(data_file_path, 'r') as f:
        qc_autoconv = np.squeeze(f[hdf_cloud_var_names[0]])
        threshold_values = int(HDF_THRESHOLD_PERCENTAGE * qc_autoconv.shape[0] * qc_autoconv.shape[1])
        index_list = find_nonzero_threshold(qc_autoconv, threshold_values, hdf=True)
        for i in range(len(hdf_cloud_var_names)):
            data_map[hdf_cloud_var_names[i]] = load_hdf_dataset(hdf_cloud_var_names[i], index_list, f)
        data_map[turb_var_names[0]] = load_hdf_dataset(turb_var_names[0], index_list, f)
    return data_map

def create_deep_dataset(datamap, log_map):
    '''
    Extract arrays for a deep model.

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
    
    input_data = np.stack(data_list[:-1], axis=1) #exclude target data
    input_data = min_max_normalize(input_data, (0))
    target_data = min_max_normalize(data_list[-1])

    return torch.FloatTensor(input_data), torch.FloatTensor(target_data).unsqueeze(1)

def create_deep_dataset_subset(data_map, log_map, percent):
    """
    Use subset of data for faster training
    """
    input_data, target_data = create_deep_dataset(data_map, log_map)
    p = torch.randperm(len(input_data))

    subset_size = int(percent * len(input_data))

    input_data_subset = input_data[p][:subset_size]
    target_data_subset = target_data[p][:subset_size]

    return input_data_subset, target_data_subset

def save_data_info(inputs, targets, model_folder_path, model_name):
    """
    Save data information (mean, min, max) to JSON files. 
    Needed for rescaling model outputs to real-life interpretable values. 

    Args:
        inputs (np.array): Input data array.
        targets (np.array): Target data array.
        model_folder_path (str): Path to the folder to save JSON files.
        model_name (str): Name of the model to use in filenames.
    """
    input_data_map = {}
    
    target_data_map = {}
    non_zero_mask = targets != 0
    masked_targets = targets[non_zero_mask]
    target_data_map['mean'] = np.mean(targets).item()
    target_data_map['std'] = np.std(targets).item()
    target_data_map['min'] = np.min(masked_targets).item()
    target_data_map['max'] = np.max(masked_targets).item()
    
    qc = inputs[:, 0]
    nc = inputs[:, 1]
    tke_sgs = inputs[:, 2]
    masked_qc = qc[qc !=0]
    masked_nc = nc[nc != 0]
    masked_tke_sgs = tke_sgs[tke_sgs != 0]
    input_data_map['qc'] = {'min': np.min(masked_qc).item(), 'max': np.max(masked_qc).item()}
    input_data_map['nc'] = {'min': np.min(masked_nc).item(), 'max': np.max(masked_nc).item()}
    input_data_map['tke_sgs'] = {'min': np.min(masked_tke_sgs).item(), 'max': np.max(masked_tke_sgs).item()}

    with open(Path(model_folder_path) / f'{model_name}_target_data_map.json', 'w') as f:
        json.dump(target_data_map, f)

    with open(Path(model_folder_path) / f'{model_name}_input_data_map.json', 'w') as f:
        json.dump(input_data_map, f)
        
def save_data_info_mean_std(inputs, targets, model_folder_path, model_name):
    """
    Save data information (mean, std) to JSON files. 
    Needed for rescaling model outputs to real-life interpretable values. 

    Args:
        inputs (np.array): Input data array.
        targets (np.array): Target data array.
        model_folder_path (str): Path to the folder to save JSON files.
        model_name (str): Name of the model to use in filenames.
    """
    input_data_map = {}
    target_data_map = {}

    target_data_map['mean'] = np.mean(targets).item()
    target_data_map['std'] = np.std(targets).item()
    
    qc = inputs[:, 0]
    nc = inputs[:, 1]
    tke_sgs = inputs[:, 2]
    
    input_data_map['qc'] = {'mean': np.mean(qc).item(), 'std': np.std(qc).item()}
    input_data_map['nc'] = {'mean': np.mean(nc).item(), 'std': np.std(nc).item()}
    input_data_map['tke_sgs'] = {'mean': np.mean(tke_sgs).item(), 'std': np.std(tke_sgs).item()}

    with open(Path(model_folder_path) / f'{model_name}_target_data_map.json', 'w') as f:
        json.dump(target_data_map, f)

    with open(Path(model_folder_path) / f'{model_name}_input_data_map.json', 'w') as f:
        json.dump(input_data_map, f)