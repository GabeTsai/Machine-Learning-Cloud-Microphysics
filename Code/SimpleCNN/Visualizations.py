import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Model import SimpleCNN
from DataUtils import create_dataset, min_max_denormalize
import xarray as xr
import seaborn as sns
from pathlib import Path
import json

NUM_HEIGHT_LEVELS = 28
model_folder_path = '../../SavedModels/SimpleCNN'
data_folder = '../../Data'
data_file = 'ena25jan2023.nc'

with open(Path(model_folder_path) / 'heightSimpleCNN.json', 'r') as f:
        height_map = json.load(f)

def load_model(model_path):
    '''
    Load model from model_path
    '''
    model = SimpleCNN(height_dim = NUM_HEIGHT_LEVELS)
    model.load_state_dict(torch.load(model_path))
    return model

def predict(model, fold, inputs, targets, test_indices):
    '''
    Predict values using model and test_loader. Return unlogged values.
    '''

    select_inputs = inputs[test_indices[f'Fold {fold}']]
    select_targets = targets[test_indices[f'Fold {fold}']]
    
    model.eval()
    predictions = model(torch.FloatTensor(select_inputs))
    true_values = select_targets

    criterion = nn.MSELoss()
    loss = criterion(predictions, true_values)
    print(f'Loss for fold {fold}: {loss.item()}')

    return predictions.detach().numpy(), true_values.numpy()

def prepare_pred(predictions, true_values, true_data):
    '''
    Denormalize and unlog the predictions and true values (ignore 0 values)
    '''
    log_values = np.log(true_data, out=np.zeros_like(true_data, dtype=np.float64), where = (true_data > 0))
    log_values = np.nan_to_num(log_values, nan = 0)
    target_min = np.min(log_values)
    target_max = np.max(log_values)

    predictions = np.exp(min_max_denormalize(predictions, target_min, target_max), where = predictions > 0)
    true_values = np.exp(min_max_denormalize(true_values, target_min, target_max), where = true_values > 0)
    return predictions, true_values

def plot_pred_single_sample_time(true_values, predicted_values, i, time, ax):
    '''
    Plot predicted and true values for a single sample using scatterplot
    '''
    true_value_slice = true_values[i]
    predicted_value_slice = predicted_values[i]
    height_list = []
    for _, height in height_map.items():
        height_list.append(height)
    
    # fig, ax = plt.subplots(figsize=(10, 10))  # Create a Figure and an Axes
    ax.scatter(height_list, true_value_slice, label='True values', color='red', s = 10)
    ax.scatter(height_list, predicted_value_slice, label='Predicted values', color='blue', s = 10)

    ax.set_title('Time ' + str(time) + 's', fontsize = 'small')
    ax.set_xlabel('Height level (m)', fontsize = 'small')
    ax.set_ylabel('auto_cldmsink_b_cloud (kg/kg/s)', fontsize = 'small')
    ax.tick_params(axis='both', which='major', labelsize= 'small')
    ax.tick_params(axis='both', which='minor', labelsize= 'small')

    ax.legend(fontsize = 'small')

def plot_pred_single_sample_height(true_values, predicted_values, i, ax, time_list):
    '''
    Plot predicted and true values for a single sample using scatterplot
    '''
    true_value_slice = true_values[:, i]
    predicted_value_slice = predicted_values[:, i]
    
    height = height_map[str(i)]

    ax.scatter(time_list, true_value_slice, label='True values', color='red', s = 10)
    ax.scatter(time_list, predicted_value_slice, label='Predicted values', color='blue', s = 10)

    ax.set_title('Height level ' + str(i), fontsize = 'small')
    ax.set_xlabel('Time (s)', fontsize = 'small')
    ax.set_ylabel('auto_cldmsink_b_cloud (kg/kg/s)', fontsize = 'small')
    ax.tick_params(axis='both', which='major', labelsize= 'small')
    ax.tick_params(axis='both', which='minor', labelsize= 'small')

    ax.legend(fontsize = 'small')
    plt.show()

def plot_multiple_samples(true_values, predicted_values, model, time_list, time_indices):
    '''
    Plot predicted and true values for multiple samples using scatterplot
    '''
    fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize = (30, 20))

    axs = axs.flatten()
    for i, ax in enumerate(axs):
        time = time_list[time_indices[i * 5]]
        plot_pred_single_sample_time(true_values, predicted_values, i * 5, model, time, ax)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    plt.show()

def heat_map(true_values, predicted_values):
    '''
    Plot a heatmap of the true and predicted values
    '''
    diff = np.transpose(np.abs(true_values - predicted_values))
    plt.figure(figsize=(10, 10))
    plt.imshow(diff, cmap='coolwarm', interpolation = 'none')
    plt.colorbar(label = 'Difference (kg/kg/s)')
    plt.xlabel('Time')
    plt.ylabel('Height level')
    plt.gca().invert_yaxis()
    plt.show()

def main():
    fold = 3

    cloud_ds = xr.open_dataset(Path(data_folder) / data_file, group = 'DiagnosticsClouds/profiles')

    with open(Path(model_folder_path) / 'test_indices.json', 'r') as f:
        test_indices = json.load(f)  # Load the test indices dictionary

    with open (Path(data_folder) / 'dataSimpleCNN.json', 'r') as f:
        data_map = json.load(f) # Load the data map
    
    time_list = data_map['time'] #Load the time list to convert indices to time
    model = load_model(f'../../SavedModels/SimpleCNN/SimpleCNN_{fold}.pth')

    inputs, targets = create_dataset(data_folder)   

    predictions, true_values = predict(model, fold, inputs, targets, test_indices)
    fig, ax = plt.subplots(figsize=(10, 10))  # Create a Figure and an Axes

    print(predictions.shape)
    # heat_map(true_values, predictions)
    auto_cldmsink_b_cloud = cloud_ds['auto_cldmsink_b_cloud'].values
    predictions, true_values = prepare_pred(predictions, true_values, auto_cldmsink_b_cloud)
    
    time_indices = test_indices[f'Fold {fold}']
    # plot_pred_single_sample_time(true_values, predictions, 0, time_list[time_indices[0]], ax)
    plot_pred_single_sample_height(true_values, predictions, 6, ax, time_list)
    # plot_multiple_samples(true_values, predictions, model, time_list, time_indices)
    
    
if __name__ == "__main__":
    main()
    
