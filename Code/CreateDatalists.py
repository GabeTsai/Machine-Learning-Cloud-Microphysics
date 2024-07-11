import netCDF4
import xarray as xr
import numpy as np
import json

#Open file, initialize data arrays
file = '../Data/ena25jan2023.nc'
cloud_ds = xr.open_dataset(file, group = 'DiagnosticsClouds/profiles')
var_name = ['qc_autoconv_cloud', 'nc_autoconv_cloud', 'qr_autoconv_cloud', 'nr_autoconv_cloud', 'auto_cldmsink_b_cloud']
NONZERO_THRESHOLD = 550

def find_nonzero(dataset):
    map = {}
    for i in range(270):
        array = dataset.isel(z = i).data
        nonzero_count = np.count_nonzero(array)
        if (nonzero_count):
            map[i] = nonzero_count
    return map

def extract_nonzero_data(dataset):
    map = find_nonzero(dataset)
    filtered_list = [index for index, count in map.items() if count >= NONZERO_THRESHOLD]
    data_list = []
    for index in filtered_list:
        data_list.append(dataset.isel(z = index).data.tolist())
    return data_list

def dump_data(filePath):
    data_map = {}
    for name in var_name:
        data_map[name] = extract_nonzero_data(cloud_ds[name])
    with open(filePath + '/data.json', 'w') as f:
        json.dump(data_map, f)

def main():
    dump_data('../Data')

if __name__ == "__main__":
    main()