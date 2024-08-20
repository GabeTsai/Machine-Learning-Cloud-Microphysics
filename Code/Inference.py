import numpy as np
import torch
import torch.nn as nn
from Train import choose_model, test_best_config
from Models.MLP.MLPDataUtils import concat_data
from DataUtils import * 

def create_MLP_test_dataset_nc(data_file_path, model_name, model_folder_path, data_name):
    data_map = create_test_data_map_nc(data_file_path)
    inputs, targets = concat_data([data_map], model_name, model_folder_path, data_name = data_name)
    return torch.FloatTensor(inputs), torch.FloatTensor(targets).unsqueeze(1)

def main():
    model_name = 'MLP3'
    data_file_name = 'dycoms_stats.nc'
    model_folder_path = f'../SavedModels/{model_name}'
    data_file_path = f'../Data/Test/{data_file_name}'
    inputs, targets = create_MLP_test_dataset_nc(data_file_path, model_name, model_folder_path, 'dycoms')
    test_dataset = torch.utils.data.TensorDataset(inputs, targets)
    test_loss, predictions, true_values = test_best_config(test_dataset, model_name, model_file_name, model_folder_path)

if __name__ == "__main__":
    main()