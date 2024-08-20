import numpy as np
import torch
import torch.nn as nn
from Train import choose_model, test_best_config
from Models.MLP.MLPDataUtils import concat_data
from Visualizations import *
from DataUtils import * 

def create_MLP_test_dataset_nc(data_file_path, model_name, model_folder_path, data_name):
    data_map = create_test_data_map_nc(data_file_path)
    inputs, targets = concat_data([data_map], model_name, model_folder_path, data_name = data_name)
    return torch.FloatTensor(inputs), torch.FloatTensor(targets)

def create_test_dataset_hdf5(data_file_name):
    log_map = {
        'qc_autoconv': True,
        'nc_autoconv': False,
        'tke_sgs': True,
        'auto_cldmsink_b': True}
    data_map = prepare_hdf_dataset(data_file_name)
    inputs, targets = create_deep_dataset_subset(data_map, log_map, percent = 1)
    return torch.FloatTensor(inputs), torch.FloatTensor(targets)

def main():
    model_name = 'MLP3'
    dataset_name = 'ENAh5'
    data_file_path = f'../Data/Test/{dataset_name}_stats.nc'
    denorm_log_pred = True
    model_folder_path = f'../SavedModels/{model_name}'
    data_file_name = '/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/Data/00d-03h-00m-00s-000ms.h5'
    
    model_file_name = 'best_model_MLP3hl164hl232lr0.000324132680419563weight_decay7.0845415052502345e-06batch_size256max_epochs1000.pth'
    # inputs, targets = create_MLP_test_dataset_nc(data_file_path, model_name, model_folder_path, dataset_name)
    inputs, targets = create_test_dataset_hdf5(data_file_name)
    histogram_single(inputs[:, 0], 'qc', model_name, f'{dataset_name}qc', '../Visualizations/')
    histogram_single(inputs[:, 1], 'nc', model_name, f'{dataset_name}nc', '../Visualizations/')
    histogram_single(inputs[:, 2], 'tke', model_name, f'{dataset_name}tke', '../Visualizations/')
    histogram_single(targets, 'targets', model_name, f'{dataset_name}targets', '../Visualizations/')
    test_dataset = torch.utils.data.TensorDataset(inputs, targets)
    
    test_loss, predictions, true_values = test_best_config(test_dataset, model_name, model_file_name, model_folder_path)
    # if denorm_log_pred:
    #     predictions, true_values = denormalize_predictions(model_folder_path, f'{model_name}_{dataset_name}', predictions, true_values, test_dataset, delog = False)
    density_plot(np.array(predictions), np.array(true_values), model_name, f'{dataset_name}_log')
    

if __name__ == "__main__":
    main()