import xarray as xr
import netCDF4
import numpy as np
import torch
import h5py
from pathlib import Path
import os
import json

from sklearn.metrics import r2_score

#Open file, initialize data arrays 
nc_cloud_var_names = ['qc_autoconv_cloud', 'nc_autoconv_cloud','auto_cldmsink_b_cloud']
hdf_cloud_var_names = ['qc_autoconv', 'nc_autoconv', 'auto_cldmsink_b']
turb_var_names = ['tke_sgs']
NC_QC_THRESHOLD = 1e-6

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

def remove_outliers_std(arr):
    """
    Remove values that are more than 3 standard deviations away from the mean.

    Args:
        arr (np.array): Input NumPy array to check for outliers.

    Returns:
        np.array: Mask of the same shape as `arr` where outliers are marked as False.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    
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

def destandardize_single(data, mean, std):
    """
    Destandardize data using mean and standard deviation.

    Args:
        data (np.array): Standardized data to destandardize.
        mean (float): Mean of the original data.
        std (float): Standard deviation of the original data.

    Returns:
        np.array: Destandardized data.
    """
    return data * std + mean

def destandardize_output(model_folder_path, model_name, data):
    with open(Path(model_folder_path) / f'{model_name}_data_info.json', 'r') as f:
        data_map = json.load(f)
    
    target_data = data_map['targets']
    mean = target_data['mean']
    std = target_data['std']

    return destandardize_single(data, mean, std)

def find_nonzero_threshold(dataset, threshold):
    """
    Return list of indices of data that are more than threshold. 

    Args:
        dataset (ndarray): Dataset to search through.
        threshold (float): Threshold value used for marking index_list

    Returns:
        list: Mask of indices of height levels meeting the criteria.
    """
    return dataset >= threshold

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
    turb_ds = xr.open_dataset(data_file_path, group='DiagnosticState/profiles')

    mask = find_nonzero_threshold(np.array(cloud_ds[nc_cloud_var_names[0]]).flatten(), NC_QC_THRESHOLD)
    data_map = {}
    for i in range(len(nc_cloud_var_names)):
        data = np.array(cloud_ds[nc_cloud_var_names[i]]).flatten()
        data_map[nc_cloud_var_names[i]] = data[mask]
    for i in range(len(turb_var_names)):
        data = np.array(turb_ds[turb_var_names[i]]).flatten()
        data_map[turb_var_names[i]] = data[mask]
    return data_map

def filter_data(qc):
    return qc >= THRESHOLD

def create_test_data_map_nc(data_file_path):
    cloud_ds = xr.open_dataset(data_file_path, group='DiagnosticsClouds/profiles')
    data_map = {}
    filter = filter_data(np.array(cloud_ds[nc_cloud_var_names[0]]).squeeze().flatten())
    for i in range(len(nc_cloud_var_names)):
        data = np.array(cloud_ds[nc_cloud_var_names[i]])
        data = data.squeeze().flatten()
        data = data[filter]
        data = log_ignore_zero(data)
        data_map[nc_cloud_var_names[i]] = data
    turb_ds = xr.open_dataset(data_file_path, group='DiagnosticState/profiles')
    for i in range(len(turb_var_names)):
        data = np.array(turb_ds[turb_var_names[i]])
        data = data.squeeze().flatten()
        data = data[filter]
        data = log_ignore_zero(data)
        data_map[turb_var_names[i]] = data
    
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
        index_list = find_nonzero_threshold(qc_autoconv)
        for i in range(len(hdf_cloud_var_names)):
            data_map[hdf_cloud_var_names[i]] = load_hdf_dataset(hdf_cloud_var_names[i], index_list, f)
        data_map[turb_var_names[0]] = load_hdf_dataset(turb_var_names[0], index_list, f)
    return data_map

def create_h5_dataset(datamap, log_map):
    '''
    Extract arrays for a model from datamap obtained from h5 file

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

def create_h5_dataset_subset(data_map, log_map, percent):
    """
    Use subset of data for faster training
    """
    input_data, target_data = create_h5_dataset(data_map, log_map)
    p = torch.randperm(len(input_data))

    subset_size = int(percent * len(input_data))

    input_data_subset = input_data[p][:subset_size]
    target_data_subset = target_data[p][:subset_size]

    return input_data_subset, target_data_subset

def load_data(data_folder_path, model_folder_path, model_name):
    '''
    Load data for specific type of model from data folder path
    '''
    from Models.MLP.MLPDataUtils import create_MLP_dataset
    from Models.GEN import GENModel, Icosphere
    from Models.DeepMLP.DeepMLPModel import DeepMLP
    def create_MLP_dataset_wrapper():
        return create_MLP_dataset(data_folder_path, model_name, model_folder_path)

    def create_deep_dataset_wrapper():
        log_map = {
            'qc_autoconv': True,
            'nc_autoconv': False,
            'tke_sgs': True,
            'auto_cldmsink_b': True}
        data_file_name = '00d-03h-00m-00s-000ms.h5'
        data_map = prepare_hdf_dataset(Path(data_folder_path) / data_file_name)
        return create_h5_dataset_subset(data_map, log_map, percent = 1)
    
    model_data_loaders = {
        'MLP3': create_MLP_dataset_wrapper,
        'GEN': create_deep_dataset_wrapper,
        'DeepMLP': create_MLP_dataset_wrapper
    }

    if model_name not in model_data_loaders:
        raise ValueError(f"Unsupported model_name: {model_name}")

    data = model_data_loaders[model_name]()

    return data

def create_model_dataset(data_folder_path, model_folder_path, model_name):
    """
    Returns train/val dataset and saves test dataset as .pth file depending on model. 
    """
    data = load_data(data_folder_path, model_folder_path, model_name)

    input_data, target_data = data
    train_val_dataset = torch.utils.data.TensorDataset(input_data, target_data)
        
    return train_val_dataset

def save_data_info(inputs, targets, model_folder_path, model_name, dataset_name=''):
    """
    Save data information (mean, min, max, std) to a single JSON file. 
    Needed for rescaling model outputs to real-life interpretable values. 

    Args:
        inputs (np.array): Input data array.
        targets (np.array): Target data array.
        model_folder_path (str): Path to the folder to save the JSON file.
        model_name (str): Name of the model to use in the filename.
        dataset_name (str): Optional dataset name to include in the filename.
    """
    data_info = {
        'inputs': {},
        'targets': {}
    }
    
    # Process targets
    non_zero_mask = targets != 0
    masked_targets = targets[non_zero_mask]
    data_info['targets']['mean'] = np.mean(targets).item()
    data_info['targets']['std'] = np.std(targets).item()
    data_info['targets']['min'] = np.min(masked_targets).item()
    data_info['targets']['max'] = np.max(masked_targets).item()
    
    # Process inputs
    qc = inputs[:, 0]
    nc = inputs[:, 1]
    tke_sgs = inputs[:, 2]
    
    masked_qc = qc[qc != 0]
    masked_nc = nc[nc != 0]
    masked_tke_sgs = tke_sgs[tke_sgs != 0]
    
    data_info['inputs']['qc'] = {
        'mean': np.mean(qc).item(),
        'std': np.std(qc).item(),
        'min': np.min(masked_qc).item(),
        'max': np.max(masked_qc).item()
    }
    data_info['inputs']['nc'] = {
        'mean': np.mean(nc).item(),
        'std': np.std(nc).item(),
        'min': np.min(masked_nc).item(),
        'max': np.max(masked_nc).item()
    }
    data_info['inputs']['tke_sgs'] = {
        'mean': np.mean(tke_sgs).item(),
        'std': np.std(tke_sgs).item(),
        'min': np.min(masked_tke_sgs).item(),
        'max': np.max(masked_tke_sgs).item()
    }

    # Save all data to a single JSON file
    file_path = Path(model_folder_path) / f'{model_name}{dataset_name}_data_info.json'
    with open(file_path, 'w') as f:
        json.dump(data_info, f)

def pred_metrics(predictions, targets):
    """
    Calculate various metrics for model predictions.

    Args:
        predictions (np.array): Model predictions.
        targets (np.array): Target values.

    Returns:
        dict: Dictionary containing various metrics.
    """
    metrics = {}
    metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
    metrics['MAE'] = np.mean(np.abs(predictions- targets))
    metrics['r2'] = r2_score(targets, predictions)
    return metrics 
