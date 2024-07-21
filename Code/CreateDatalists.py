import xarray as xr
import numpy as np
import os

#Open file, initialize data arrays
data_folder_file_path = '../../../Data' #Path to data folder from file in Models/CNNs
file = 'ena25jan2023.nc'
cloud_ds = xr.open_dataset(data_folder_file_path + '/' + file, group = 'DiagnosticsClouds/profiles')
var_name = ['qc_autoconv_cloud', 'nc_autoconv_cloud', 'qr_autoconv_cloud', 'nr_autoconv_cloud', 'auto_cldmsink_b_cloud']
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
    data = min_max_normalize(data)
    return data.tolist()

def create_data_map(index_list):
    '''
    Create map to store all data arrays
    '''
    data_map = {}
    for i in range(len(var_name)):
        data_map[var_name[i]] = prepare_dataset(cloud_ds[var_name[i]], log_list[i], index_list)
    return data_map