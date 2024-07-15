import netCDF4
import torch
import xarray as xr
import numpy as np
import json

#Open file, initialize data arrays
data_folder_file_path = '/Users/HP/Documents/GitHub/Machine-Learning-Cloud-Microphysics/Data'
file = 'ena25jan2023.nc'
cloud_ds = xr.open_dataset(data_folder_file_path + '/' + file, group = 'DiagnosticsClouds/profiles')
var_name = ['qc_autoconv_cloud', 'nc_autoconv_cloud', 'qr_autoconv_cloud', 'nr_autoconv_cloud', 'auto_cldmsink_b_cloud']
log_list = [False, False, True, True, True]

THRESHOLD_VALUES = 0.90 * 721
THRESHOLD = 1e-6

def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def find_nonzero_threshold(dataset, num_values):
    index_list = []
    for i in range(dataset.shape[1]):
        array = dataset.isel(z = i).data
        count = np.count_nonzero(array >= THRESHOLD)
        if count > num_values:
            index_list.append(i)
    return index_list

def extract_data(dataset, index_list):
    data_list = []
    for index in index_list:
        data_list.append(dataset[:, index])
    data_array = np.array(data_list)
    return data_array

def prepare_dataset(dataset, log, index_list):
    dataset_copy = dataset.values
    if log:
        dataset_copy = np.log(dataset_copy, where = dataset_copy > 0)
    data = extract_data(dataset_copy, index_list)
    data = min_max_normalize(data)
    return data.tolist()

def dump_data(filePath, index_list):
    data_map = {}
    for i in range(len(var_name)):
        data_map[var_name[i]] = prepare_dataset(cloud_ds[var_name[i]], log_list[i], index_list)
    data_map['time'] = cloud_ds['time'].values.tolist()
    height_map = {}
    for i in index_list:
        height_map[str(i)] = cloud_ds[var_name[0]].time.values[i]
    with open(filePath + '/data.json', 'w') as f:
        json.dump(data_map, f)
    with open(filePath + '/height.json', 'w') as f:
        json.dump(height_map, f)
    
def create_dataset(dataFolder):
    with open(dataFolder + '/data.json', 'r') as f:
        data_map = json.load(f)
    qc_autoconv_cloud = np.array(np.transpose(data_map['qc_autoconv_cloud']))
    nc_autoconv_cloud = np.array(np.transpose(data_map['nc_autoconv_cloud']))
    qr_autoconv_cloud = np.array(np.transpose(data_map['qr_autoconv_cloud']))
    nr_autoconv_cloud = np.array(np.transpose(data_map['nr_autoconv_cloud']))
    auto_cldmsink_b_cloud = np.array(np.transpose(data_map['auto_cldmsink_b_cloud']))

    input_data = np.stack((qc_autoconv_cloud, nc_autoconv_cloud, 
                            qr_autoconv_cloud, nr_autoconv_cloud), axis = 1)
    target_data = auto_cldmsink_b_cloud

    return torch.FloatTensor(input_data), torch.FloatTensor(target_data)

def main():
    input_data, target_data = create_dataset('/Users/HP/Documents/GitHub/Machine-Learning-Cloud-Microphysics/Data')
    print(input_data.shape)
    print(target_data.shape)

if __name__ == "__main__":
    main()