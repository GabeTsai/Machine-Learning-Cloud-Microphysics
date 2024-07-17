import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Model import SimpleCNN, LargerCNN
from DataUtils import create_dataset, min_max_denormalize
import xarray as xr
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score
import json

NUM_HEIGHT_LEVELS = 30

def predict(model, fold, inputs, targets, test_indices):
    '''
    Predict values using model and test_loader. Return unlogged values.
    '''

    select_inputs = inputs[test_indices[f'Fold {fold}']]
    select_targets = targets[test_indices[f'Fold {fold}']]
    
    predictions = model(torch.FloatTensor(select_inputs))
    true_values = select_targets

    criterion = nn.MSELoss()
    loss = criterion(predictions, true_values)
    print(f'Loss for fold {fold}: {loss.item()}')
    true_values_np = true_values.detach().numpy()
    predictions_np = predictions.detach().numpy()
    r2 = r2_score(true_values_np, predictions_np, multioutput='uniform_average')
    print(f'Averaged R2 score for fold {fold}: {r2}')
    return predictions_np, true_values_np

def prepare_pred(predictions, true_values, true_data):
    '''
    Denormalize and unlog the predictions and true values (ignore 0 values)
    '''
    log_values = np.log(true_data, out=np.zeros_like(true_data, dtype=np.float64), where = (true_data > 0))
    log_values = np.nan_to_num(log_values, nan = 0)
    target_min = np.min(log_values)
    target_max = np.max(log_values)

    predictions = np.exp(min_max_denormalize(predictions, target_min, target_max), 
                         out=np.zeros_like(predictions, dtype=np.float64), where = predictions > 0)
    true_values = np.exp(min_max_denormalize(true_values, target_min, target_max), 
                         out = np.zeros_like(true_values, dtype = np.float64), where = true_values > 0)
    return predictions, true_values

def plot_pred_single_sample_time(true_values, predicted_values, i, time, height_map, ax):
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

def plot_pred_single_sample_height(true_values, predicted_values, i, time_list, time_indices, height_map, ax):
    '''
    Plot predicted and true values for a single sample using scatterplot
    '''
    true_value_slice = true_values[:, i]
    
    predicted_value_slice = predicted_values[:, i]
    print(height_map)
    height = 0
    count = 0
    time_list_plot = []
    for _, h in height_map.items():
        if count == i:
            height = h
            break
        count += 1
    for time in time_indices:
        time_list_plot.append(time_list[time])
    ax.scatter(time_list_plot, true_value_slice, label='True values', color='red', s = 10)
    ax.scatter(time_list_plot, predicted_value_slice, label='Predicted values', color='blue', s = 10)

    ax.set_title('Height level ' + str(height), fontsize = 'small')
    ax.set_xlabel('Time (s)', fontsize = 'small')
    ax.set_ylabel('auto_cldmsink_b_cloud (kg/kg/s)', fontsize = 'small')
    ax.tick_params(axis='both', which='major', labelsize= 'small')
    ax.tick_params(axis='both', which='minor', labelsize= 'small')

    ax.legend(fontsize = 'small')
    plt.show()

def plot_multiple_samples(true_values, predicted_values, model, time_list, height_map, time_indices):
    '''
    Plot predicted and true values for multiple samples using scatterplot
    '''
    fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize = (30, 20))

    axs = axs.flatten()
    for i, ax in enumerate(axs):
        time = time_list[time_indices[i * 5]]
        plot_pred_single_sample_time(true_values, predicted_values, i * 5, time, height_map, ax)
    
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
    plt.colorbar(label = 'Difference (log min-max normalized)')
    plt.xlabel('Time')
    plt.ylabel('Height level')
    plt.gca().invert_yaxis()
    plt.show()

def main():
    fold = 4
    model_name = 'SimpleCNN'
    model_folder_path = f'../../SavedModels/{model_name}'
    data_folder = '../../Data'
    data_file = 'ena25jan2023.nc'

    with open(Path(data_folder) / 'height.json', 'r') as f:
        height_map = json.load(f)
    cloud_ds = xr.open_dataset(Path(data_folder) / data_file, group = 'DiagnosticsClouds/profiles')

    with open(Path(model_folder_path) / 'test_indices.json', 'r') as f:
        test_indices = json.load(f)  # Load the test indices dictionary

    with open (Path(data_folder) / 'data.json', 'r') as f:
        data_map = json.load(f) # Load the data map

    time_list = data_map['time'] #Load the time list to convert indices to time
    model = SimpleCNN(NUM_HEIGHT_LEVELS, torch.zeros(30, ))
    model.load_state_dict(torch.load(Path(model_folder_path) / f'{model_name}_{fold}.pth'))
    inputs, targets = create_dataset(data_folder)   

    predictions, true_values = predict(model, fold, inputs, targets, test_indices)
    fig, ax = plt.subplots(figsize=(10, 10))  # Create a Figure and an Axes

    # heat_map(true_values, predictions)
    auto_cldmsink_b_cloud = cloud_ds['auto_cldmsink_b_cloud'].values
    predictions, true_values = prepare_pred(predictions, true_values, auto_cldmsink_b_cloud)
    
    time_indices = test_indices[f'Fold {fold}']
    # plot_pred_single_sample_time(true_values, predictions, 0, time_list[time_indices[0]], ax)
    # plot_pred_single_sample_height(true_values, predictions, 6, time_list, time_indices, height_map, ax)
    # plot_multiple_samples(true_values, predictions, model, time_list, height_map, time_indices)
    
    
if __name__ == "__main__":
    main()
    
