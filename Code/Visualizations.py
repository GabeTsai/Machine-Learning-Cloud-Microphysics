import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from Models.CNNs.CNNModels import SimpleCNN, LargerCNN
from Models.CNNs.CNNDataUtils import create_CNN_dataset, min_max_denormalize
from Models.MLP.MLPModel import MLP

import xarray as xr
from pathlib import Path
import json

NUM_HEIGHT_LEVELS = 30 #parameter specific to CNNs

def predict(model, test_dataset):
    '''
    Predict values from test TensorDataset
    '''
    inputs, targets = test_dataset[:]
    predictions = model(inputs).squeeze()
    true_values = targets.squeeze()

    true_values_np = true_values.detach().numpy()
    predictions_np = predictions.detach().numpy()
    
    return predictions_np, true_values_np

def get_loss(predictions, targets):
    criterion = nn.MSELoss()
    loss = criterion(torch.FloatTensor(predictions), torch.FloatTensor(targets))
    print(f'Mean Squared Error: {loss.item()}')

def predict_fold(model, fold, inputs, targets, test_indices):
    '''
    Predict values using model and test_loader. 
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

def denormalize_delog(arr, arr_min, arr_max):
    '''
    Denormalize and unlog the array
    '''
    return np.exp(min_max_denormalize(arr, arr_min, arr_max), 
                  out=np.zeros_like(arr, dtype=np.float64), where = arr > 0)

def prepare_pred(predictions, true_values, min, max, denormalize = True):
    '''
    Denormalize and unlog the predictions and true values (ignore 0 values)
    '''
    if denormalize:
        predictions = denormalize_delog(predictions, min, max)
        true_values = denormalize_delog(true_values, min, max)
    
    print(np.median(predictions)) 
    print(np.median(true_values)) 
    print(np.max(predictions))  
    print(np.max(true_values))   
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
    ax.scatter(true_value_slice, height_list, label='True values', color='red', s = 10)
    ax.scatter(predicted_value_slice, height_list, label='Predicted values', color='blue', s = 10)

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

def setup_CNN_Fold_predictions(model, data_folder, data_file, fold, model_folder_path, model_name):
    '''
    Setup the CNN model and data for predictions
    '''
    with open(Path(data_folder) / 'height.json', 'r') as f:
        height_map = json.load(f)
    cloud_ds = xr.open_dataset(Path(data_folder) / data_file, group = 'DiagnosticsClouds/profiles')

    with open(Path(model_folder_path) / 'test_indices.json', 'r') as f:
        test_indices = json.load(f)  # Load the test indices dictionary

    with open (Path(data_folder) / 'data.json', 'r') as f:
        data_map = json.load(f) # Load the data map

    time_list = data_map['time'] #Load the time list to convert indices to time

    model.load_state_dict(torch.load(Path(model_folder_path) / f'{model_name}_{fold}.pth'))
    inputs, targets = create_CNN_dataset(data_folder)   

    return model, cloud_ds, test_indices, time_list, height_map, inputs, targets

def plot_CNN_Fold_predictions(predictions, true_values, cloud_ds, test_indices, fold, model, time_list, height_map, plot_type):
    fig, ax = plt.subplots(figsize=(10, 10))  # Create a Figure and an Axes
    if plot_type == 'heat_map':
        heat_map(true_values, predictions)
    else:
        auto_cldmsink_b_cloud = cloud_ds['auto_cldmsink_b_cloud'].values
        predictions, true_values = prepare_pred(predictions, true_values, auto_cldmsink_b_cloud)
        time_indices = test_indices[f'Fold {fold}']
        if plot_type == 'single_sample_time':
            plot_pred_single_sample_time(true_values, predictions, 0, time_list[time_indices[0]], height_map, ax)
        elif plot_type == 'single_sample_height':
            plot_pred_single_sample_height(true_values, predictions, 0, time_list, time_indices, height_map, ax)
        elif plot_type == 'multiple_samples':
            plot_multiple_samples(true_values, predictions, model, time_list, height_map, time_indices)
        else:
            print('Invalid plot type')

def plot_losses(train_losses, val_losses, model_name):
    '''
    Plot the training and validation losses
    '''
    plt.plot(train_losses, label = 'Training loss')
    plt.plot(val_losses, label = 'Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Training and Validation Losses')
    plt.savefig(Path(f'../Visualizations/{model_name}/TrainingLosses.png'))

def setup_predictions(model, model_folder_path, model_name, denormalize = True):
    '''
    Prepare predictions for plotting and analysis
    '''
    with open(Path(model_folder_path) / f'{model_name}_target_data_map.json', 'r') as f:
        target_data_map = json.load(f)
    test_dataset = torch.load(Path(model_folder_path) / f'{model_name}_test_dataset.pth')
    
    min_val = target_data_map['min']
    max_val = target_data_map['max']
    model.load_state_dict(torch.load(Path(model_folder_path) / f'{model_name}.pth'))
    predictions, true_values = predict(model, test_dataset)
    predictions, true_values = prepare_pred(predictions, true_values, min_val, max_val, denormalize)
    return predictions, true_values
    
def scatter_plot(predicted_values, true_values, model_name):
    '''
    Plot a scatter plot of the true and predicted values.
    '''
    plt.figure(figsize=(5, 5))
    plt.scatter(predicted_values, true_values, s = 5)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.xlim(0, np.max(true_values))
    plt.ylim(0, np.max(true_values))
    plt.title('Predicted vs True values')
    plt.savefig(Path(f'../Visualizations/{model_name}/{model_name}ScatterPlot.png'))

def density_plot(predicted_values, true_values, model_name):
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection = 'scatter_density')
    density = ax.scatter_density(predicted_values, true_values, cmap = white_viridis, dpi = 30)
    density.set_clim(0, 50) 
    fig.colorbar(density, label = 'Number of points per pixel') 
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.xlim(0, np.max(true_values))
    plt.ylim(0, np.max(true_values))
    plt.title('Predicted vs True values')
    plt.savefig(Path(f'../Visualizations/{model_name}/{model_name}DensityPlot.png'))

def histogram(predicted_values, true_values, model_name):
    '''
    Plot a histogram of the true and predicted values.
    '''
    plt.hist(predicted_values, bins = 20, alpha = 0.5, label = 'Predicted values')
    plt.hist(true_values, bins = 20, alpha = 0.5, label = 'True values')
    plt.xlabel('auto_cldmsink_b_cloud (kg/kg/s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted and True values')
    plt.legend()
    plt.savefig(Path(f'../Visualizations/{model_name}/{model_name}Histogram.png'))

def compare_eq_vs_ml(predictions, true_values, model_folder_path, model_name):
    '''
    Compare ML model predictions with Khairoutdinov & Kogan, 2000 paramaterized equation. 
    '''

    test_dataset = torch.load(Path(model_folder_path) / f'{model_name}_test_dataset.pth')
    inputs, targets = test_dataset[:]
    with open(Path(model_folder_path) / f'{model_name}_input_data_map.json', 'r') as f:
        input_data_map = json.load(f)
    qc_min = input_data_map['qc']['min']
    qc_max = input_data_map['qc']['max']
    nc_min = input_data_map['nc']['min']
    nc_max = input_data_map['nc']['max']

    qc = inputs[:, 0]
    nc = inputs[:, 1]

    qc = min_max_denormalize(qc, qc_min, qc_max)
    nc = min_max_denormalize(nc, nc_min, nc_max)

    eq_autoconv_rate = 13.5 * np.power(qc, 2.47) * np.power(nc, -1.1) #KK2000 equation
    print(eq_autoconv_rate)
    print(predictions)
    criterion = nn.MSELoss()
    print(f'Mean Squared Error for ML model: {criterion(torch.FloatTensor(predictions), torch.FloatTensor(true_values))}')
    print(f'Mean Squared Error for KK2000 equation: {criterion(torch.FloatTensor(eq_autoconv_rate), torch.FloatTensor(true_values))}')

    # plt.scatter(true_values, true_values, label = 'True values', s = 5)
    # plt.scatter(predictions, predictions, label = 'ML predictions', s = 5)
    # plt.scatter(eq_autoconv_rate, eq_autoconv_rate, label = 'KK2000 predictions', s = 5)

def main():
    fold = 9
    model_name = 'MLP'
    model_folder_path = f'../SavedModels/{model_name}'
    data_folder = '../../Data'
    data_file = 'ena25jan2023.nc'

    model = MLP(4, torch.zeros(()))
    predictions, true_values = setup_predictions(model, model_folder_path, model_name, True)
    get_loss(predictions, true_values)
    scatter_plot(predictions, true_values, model_name)
    density_plot(predictions, true_values, model_name)
    compare_eq_vs_ml(predictions, true_values, model_folder_path, model_name)
    # model = LargerCNN(NUM_HEIGHT_LEVELS, torch.zeros(30, ))
    
    # model, cloud_ds, test_indices, time_list, height_map, inputs, targets = setup_CNN_Fold_predictions(model, data_folder, data_file, fold, model_folder_path, model_name)

    # predictions, true_values = predict_fold(model, fold, inputs, targets, test_indices)
    # plot_CNN_Fold_predictions(predictions, true_values, cloud_ds, test_indices, fold, model, time_list, height_map, 'multiple_samples')
    
    
if __name__ == "__main__":
    main()
    
