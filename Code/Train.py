import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import numpy as np
from pathlib import Path
from Models.CNNs.CNNModels import SimpleCNN, LargerCNN
from Models.MLP.MLPModel import MLP
from Models.CNNs.CNNDataUtils import create_CNN_dataset
from Models.MLP.MLPDataUtils import create_MLP_dataset
from Models.LSTM.LSTMDataUtils import create_LSTM_dataset
from Models.LSTM.LSTMModel import LSTM
from Visualizations import plot_losses
from sklearn.model_selection import KFold
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust this as needed
os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "50"

torch.manual_seed(99)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ray.init(num_gpus = 1)

def decay_lr(optimizer, decay_rate):
    '''
    Decay learning rate by decay_rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

def reset_weights(model):
    '''
    Reset weights to avoid weight leakage
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def save_checkpoint(epoch, model, model_name, optimizer, folder, fold):
    """
    Save model checkpoint

    :param epoch: epoch number
    :param model: model object
    :param optimizer: optimizer object
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    filename = f'{model_name}{fold}.pth.tar'
    torch.save(state, Path(folder) / filename)

def choose_model(model_name, input_dim, target_dim, output_bias, config):
    '''
    Choose model based on model name
    '''
    if model_name == 'MLP2':
        return MLP(input_dim[0], output_bias, config["hl1"], config["hl2"])
    else:
        raise ValueError('Model name not recognized.')
    
def define_hyperparameter_search_space(model_name, device):
    '''
    Define hyperparameter search space. Adjust as necessary.
    '''
    if str(device) == 'cuda':
        batch_size_list = [128,256,512,1024]
    else:
        batch_size_list = [16, 32, 64]
    if model_name == 'MLP2':
        return {
        "hl1": tune.sample_from(lambda _: 2 ** np.random.randint(5, 8)),
        "hl2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 6)),
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice(batch_size_list),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        }
    else:
        return {}
    
def train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, decay_lr_at, start_epoch, model_folder_path, model_name, fold):
    '''
    Train the model on a single train/test/val split until validation loss has not reached a new minimum in early_stop epochs.

    :param model: model to train
    :param criterion: loss function
    :param optimizer: optimizer
    :param dataset: tensor dataset
    :param batch_size: batch size
    :param epochs: number of epochs to train on
    :param model_folder_path: folder to save model
    :param model_name: name of model
    '''

    train_losses = []
    val_losses = []
    min_val_loss = np.inf
    epoch = start_epoch
    early_stop_counter = 0
    while early_stop_counter < early_stop:
        model.train()
        epoch_loss = 0.0
        if epoch in decay_lr_at:
            decay_lr(optimizer, 0.1)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_epoch_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_epoch_loss += loss.item()
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch} || training loss: {avg_train_loss} || validation loss: {avg_val_loss}')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model = model.state_dict()
            early_stop_counter = 0
        
        early_stop_counter += 1
        epoch += 1
        # if epoch % 5000 == 0:
        #     plot_losses(train_losses, val_losses, model_name)
    print(f'Min validation loss: {min_val_loss}')
    # plot_losses(train_losses, val_losses, model_name)
    return min_val_loss, best_model

def train_k_fold(config, dataset, model_name, model_folder_path, num_folds = 5):
    '''
    Train model using k-fold cross validation for a particular hyperparameter configuration.
    '''
    kfold = KFold(n_splits = num_folds, shuffle = True)

    val_losses = []
    input, target = dataset[0]

    _, targets = dataset[:]
    output_bias = torch.mean(targets)

    best_model = None
    best_loss = np.inf

    for fold, (train_i, val_i) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold}--------------------------------')
        # reset_weights(model)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_i)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_i)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size= int(config["batch_size"]), sampler=train_subsampler, num_workers = 4, pin_memory = False)

        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size = int(config["batch_size"]), sampler=val_subsampler, num_workers = 4, pin_memory = False)

        model = choose_model(model_name, input.shape, target.shape, output_bias, config).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])

        start_epoch = 0
        early_stop = 500
        decay_lr_at = []

        val_loss, model_state = train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, decay_lr_at, start_epoch, model_folder_path, model_name, fold)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model_state  # Update best model

    torch.save(best_model, Path(model_folder_path) / 'best_model.pth')  # Save the best model state
    mean_val_loss = np.mean(val_losses)
    session.report({"mean_val_loss": mean_val_loss})

def load_data(data_folder_path, model_folder_path, model_name):
    '''
    Load data from data folder path
    '''
    input_data = None
    target_data = None
    if model_name == 'MLP2':
        input_data, target_data = create_MLP_dataset(data_folder_path, model_folder_path, include_qr_nr = False)
    return input_data, target_data

def main():
    
    data_folder_path = '../Data/NetCDFFiles'
    checkpoint_path = None
    model_name = 'MLP2' 
    model_folder_path = f'../SavedModels/{model_name}'
    seq_length = 8

    max_num_epochs = 500

    input_data, target_data = load_data(data_folder_path, model_folder_path, model_name)

    dataset = torch.utils.data.TensorDataset(input_data, target_data)

    test_percentage = 0.1 # 10% of data used for testing
    test_size = int(len(input_data) * test_percentage)
    k_fold_train_size = len(input_data) - test_size

    k_fold_dataset, test_dataset = torch.utils.data.random_split(dataset, [k_fold_train_size, test_size])
    
    config = define_hyperparameter_search_space(model_name, device)
    

    scheduler = ASHAScheduler(
        metric = "mean_val_loss",
        mode = "min", 
        max_t = max_num_epochs,
        grace_period = 1,
        reduction_factor = 2
    )

    # analysis = tune.run(
    #     tune.with_parameters(train_k_fold, dataset = k_fold_dataset, 
    #                          model_name = model_name, model_folder_path = model_folder_path, ), # Pass in dataset for k-fold training
    #     config = config, 
    #     num_samples = 10,
    #     scheduler = scheduler,
    #     verbose = 1,
    #     resources_per_trial = {"cpu": 16, "gpu": 1},
    #     max_concurrent_trials = 1  # Limit to 4 concurrent trials
    # )
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_k_fold, dataset=k_fold_dataset,
                                 model_name=model_name, model_folder_path=model_folder_path),
            resources = {"cpu": 12, "gpu": 0},
        ),
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=10,
            max_concurrent_trials=1  # Limit to 1 concurrent trial
        ), 
        run_config=ray.air.config.RunConfig(
            name = "tune_k_fold",
            verbose=1
        )
    )

    # Run the tuning
    analysis = tuner.fit()
    best_result = analysis.get_best_result()
    best_config = best_result.config
    best_metrics = best_result.metrics
    print("Best hyperparameters found were: ", best_config)
    print("Best validation loss found was: ", best_metrics['mean_val_loss'])

    with open(f'{model_folder_path}/best_config.txt', 'w') as f:
        f.write(f"Best hyperparameters: {best_config}\n")
        f.write(f"Best validation loss: {best_metrics['mean_val_loss']}\n")
        #run the config on the test dataset

if __name__ == '__main__':
    main()