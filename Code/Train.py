import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import ExponentialLR, LinearLR

import numpy as np
from pathlib import Path
from Models.MLP.MLPModel import *
from Models.GEN import *
from Models.DeepMLP.DeepMLPModel import DeepMLP
from Models import Layers
from DataUtils import create_model_dataset, save_data_info

from sklearn.model_selection import KFold
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import os
import json
import wandb
wandb.require("core")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(3407) #is all you need
torch.cuda.manual_seed(3407)

# ray.init(num_gpus = 1)
def decay_lr(optimizer, decay_rate):
    """
    Decay learning rate by decay_rate.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be decayed.
        decay_rate (float): The factor by which to decay the learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

def reset_weights(model):
    """
    Reset weights to avoid weight leakage.

    Args:
        model (torch.nn.Module): The model whose weights will be reset.
    """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def choose_model(model_name, input_dim, output_bias, config):
    """
    Choose model based on model name.

    Args:
        model_name (str): The name of the model to choose.
        input_dim (tuple): The input dimensions.
        output_bias (bool): Whether to use output bias.
        config (dict): The configuration dictionary.

    Returns:
        torch.nn.Module: The chosen model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name == 'MLP3':
        return MLPBase(input_dim[0], output_bias, config["hl1"], config["hl2"])
    elif model_name == 'GEN':
        latent_dim = config["latent_dim"]
        encoder = MLP([input_dim[0], latent_dim, latent_dim], activation = nn.SiLU())
        node_mapper = MLPNodeStateMapper(latent_dim, latent_dim)
        processor = Processor(latent_dim)
        pooling_layer = GlobalPooling()
        decoder = MLP([latent_dim, latent_dim, 1], activation = nn.SiLU(), output_bias = output_bias)
        return GEN(encoder, node_mapper, processor, pooling_layer, decoder)
    elif model_name == 'DeepMLP':
        return DeepMLP(input_dim[0], config["latent_dim"], 1, output_bias, num_blocks = config["num_blocks"])
    else:
        raise ValueError('Model name not recognized.')
    
def define_hyperparameter_search_space(model_name, mode, device):
    """
    Define hyperparameter search space. Adjust as necessary.
    
    Args:
        model_name (str): The name of the model.
        mode (str): Type of hyperparameters to tune
        config (dict): Model architecture config
        device (torch.device): The device to use.
    
    Returns:
        dict: The hyperparameter search space.
    """
    
    if 'MLP' in model_name and 'Deep' not in model_name:
        return {
        "hl1": tune.sample_from(lambda _: 2 ** np.random.randint(5, 8)),
        "hl2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 6)),
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([128,256,512,1024]),
        "max_epochs": 1000
        }
    elif model_name == 'GEN':
        return {
            "latent_dim": 512,
            "max_lr": tune.loguniform(1e-4, 3e-3),
            "weight_decay": tune.loguniform(1e-3, 1e-1),
            "batch_size": 64,
            "max_epochs": 10
        }
    elif model_name == 'DeepMLP':
        if mode == 'architecture':
            return {
                    "latent_dim": tune.randint(32, 256),
                    "num_blocks": tune.choice([2,3,4,5]),
                    "max_lr": 3e-4,
                    "min_lr_factor": 1e-3,
                    "inc_lr_perc": 0.05,
                    "hold_lr_perc": 0.4,
                    "gamma": 0.99997, #decrease lr by factor of 0.75 every 10000 step
                    "weight_decay": 0,
                    "batch_size": 32,
                    "max_epochs": 50
                    }
        elif mode == 'lr': 
            return {
                    "min_lr_factor": tune.choice([1, 0.1, 0.01, 1e-3]),
                    "inc_lr_perc": tune.uniform(0.05, 0.15),
                    "max_lr": tune.loguniform(1e-4, 1e-3),
                    "hold_lr_perc": tune.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                    "gamma": tune.uniform(0.99993, 0.99997),
                    "weight_decay": tune.loguniform(0.001, 0.01),
                    "max_epochs": 100
            }
        else:
            raise ValueError('Invalid mode')
    else:
        return {}

def train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, start_epoch, model_name, config, scheduler = None):
    '''
    Train the model on a single train/test/val split until validation loss has not reached a new minimum in early_stop epochs.

    :param model: model to train
    :param criterion: loss function
    :param optimizer: optimizer
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param early_stop: number of epochs to stop if no improvement
    :param max_num_epochs: maximum number of epochs to train
    :param start_epoch: starting epoch
    :param model_name: name of model
    '''

    def train_epoch_default():
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    if scheduler:
        max_lr = get_lr(optimizer)
        total_num_steps = len(train_loader) * config["max_epochs"]
        print(total_num_steps)
        inc_lr_until = int(config["inc_lr_perc"] * total_num_steps)
        inc_amount = max_lr/inc_lr_until
        hold_lr_lim = int(config["hold_lr_perc"] * total_num_steps)
        linear_scheduler = LinearLR(optimizer, start_factor = config["min_lr_factor"], total_iters = inc_lr_until)
    
    num_steps = 0
    
    def train_epoch_deep(epoch):
        nonlocal num_steps
        nonlocal linear_scheduler
        nonlocal max_lr
        model.train()
        epoch_loss = 0.0
        
        batch = 0
        accum_batch_loss = 0
        num_batch = 100
        for inputs, targets in train_loader:
            #Handle learning rate adjustments
            if scheduler:
                if num_steps < inc_lr_until:
                    linear_scheduler.step()
                elif num_steps < hold_lr_lim:
                    pass
                else: 
                    scheduler.step()
            lr = get_lr(optimizer)
            wandb.log({f"Lr": lr})

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            accum_batch_loss += loss.item()

            if (batch + 1) % num_batch == 0:
                print(f'Epoch {epoch}: Averaged train loss over {num_batch} batches {int(batch/num_batch)}: {accum_batch_loss/num_batch}')
                wandb.log({f"Train loss over {num_batch} batches": accum_batch_loss/num_batch})
                accum_batch_loss = 0
            batch += 1
            num_steps += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) #clip gradients to -1 to 1
        return epoch_loss / len(train_loader)

    def validate_epoch_deep(epoch):
        model.eval()
        val_epoch_loss = 0.0

        batch = 0
        accum_batch_loss = 0
        num_batch = 100
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_epoch_loss += loss.item()
                accum_batch_loss += loss.item()
                if (batch + 1) % num_batch == 0:
                    print(f'Epoch {epoch}: Averaged val loss over {num_batch} batches {int(batch/num_batch)}: {accum_batch_loss/num_batch}')
                    wandb.log({f"Val loss over {num_batch} batches": accum_batch_loss/num_batch})
                    accum_batch_loss = 0
                batch += 1
        return val_epoch_loss / len(val_loader)

    def validate_epoch_default():
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_epoch_loss += loss.item()
        return val_epoch_loss / len(val_loader)

    # Choose the appropriate functions
    if 'GEN' or 'DeepMLP' in model_name:
        train_epoch = train_epoch_deep
        validate_epoch = validate_epoch_deep
    else:
        train_epoch = train_epoch_default
        validate_epoch = validate_epoch_default
    
    train_losses = []
    val_losses = []
    min_val_loss = np.inf
    best_model = None
    epoch = start_epoch
    early_stop_counter = 0

    while epoch < config["max_epochs"]:

        avg_train_loss = train_epoch(epoch)
        train_losses.append(avg_train_loss)

        avg_val_loss = validate_epoch(epoch)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch} || training loss: {avg_train_loss} || validation loss: {avg_val_loss}')
        wandb.log({"Train epoch loss": avg_train_loss})
        wandb.log({"Val epoch loss": avg_val_loss})
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        wandb.log({"Min val epoch loss": min_val_loss})
        session.report({"mean_val_loss": min_val_loss})
        epoch += 1

    wandb.finish()
    return min_val_loss, best_model

def train_single_split(config, dataset, model_name, model_folder_path, num_workers, mode, collate_fn=None):
    '''
    Train on a single train-valid split (provided dataset is big enough) for a particular hyperparameter config.
    '''
    if 'LSTM' in model_name:
        input, target, _ = dataset[0]
        _, targets, _ = zip(*dataset)
        output_bias = torch.mean(torch.tensor(targets))
    else:
        input, target = dataset[0]
        _, targets = dataset[:]
        output_bias = torch.mean(targets)

    model = choose_model(model_name, input.shape, output_bias, config).to(device)

    val_percent = 1/9 #Since we already reserved 10 percent of data for testing 
    val_size = int(val_percent * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size= int(config["batch_size"]), shuffle = True, num_workers = num_workers, pin_memory = True, collate_fn = collate_fn)
    
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size = int(config["batch_size"]), shuffle = True, num_workers = num_workers, pin_memory = True, collate_fn = collate_fn)

    criterion = nn.MSELoss()
    if mode == 'architecture':
        optimizer = torch.optim.AdamW(model.parameters(), lr = config["max_lr"])
        scheduler = None
    elif mode == 'lr':
        optimizer = torch.optim.AdamW(model.parameters(), lr = config["max_lr"], weight_decay = config["weight_decay"])
        scheduler = ExponentialLR(optimizer, gamma = config["gamma"])
    else:
        raise ValueError('Invalid mode')

    start_epoch = 0
    early_stop = int(config["max_epochs"] * 0.1)

    wandb.init(
      project=f"{model_name}",
      config = config
      )

    wandb.log(config)
    val_loss, best_model = train_single(model, criterion, optimizer, train_loader, val_loader, early_stop,
                                        start_epoch, model_name, config, scheduler = scheduler)

    model_key = ''.join(f'{key}{str(item)}' for key, item in config.items()) # Create unique file key based on model config
    model_data = {
        'model_state_dict': best_model, #model weights
        'config': config #model hyperparams
    }
    save_path = Path(model_folder_path) / f'best_model_{model_name}{model_key}.pth'
    torch.save(model_data, save_path)  # Save the best model state

def train_k_fold(config, dataset, model_name, model_folder_path, num_workers, num_folds = 10):
    '''
    Train model using k-fold cross validation for a particular hyperparameter configuration.
    '''
    kfold = KFold(n_splits = num_folds, shuffle = True)

    input, target = dataset[0]
    _, targets = dataset[:]
    output_bias = torch.mean(targets)

    best_loss = np.inf
    best_model = None
    for fold, (train_i, val_i) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold}--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_i)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_i)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size= int(config["batch_size"]), sampler=train_subsampler, num_workers = num_workers, pin_memory = True)

        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size = int(config["batch_size"]), sampler=val_subsampler, num_workers = num_workers, pin_memory = True)

        model = choose_model(model_name, input.shape, output_bias, config).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = config["max_lr"], weight_decay = config["weight_decay"])
        scheduler = ExponentialLR(optimizer, gamma = config["gamma"])

        start_epoch = 0
        early_stop = int(config["max_epochs"] * 0.1)

        wandb.init(
        project=f"{model_name}",
        config = config
        )

        wandb.log(config)
        wandb.log({"fold": fold})

        val_loss, best_model_fold = train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, 
                                                start_epoch, model_name, config, scheduler = scheduler)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = best_model_fold  # Update best model

    model_key = ''.join(f'{key}{str(item)}' for key, item in config.items()) # Create unique file key based on model config
    model_data = {
        'model_state_dict': best_model, #model weights
        'config': config #model hyperparams
    }
    save_path = Path(model_folder_path) / f'best_model_k_fold_{model_name}{model_key}.pth'
    torch.save(model_data, save_path)  # Save the best model state

def configure_bias(checkpoint):
    '''
    Adds extra dimension to output bias of saved model checkpoint. Workaround.
    PyTorch copies the bias as a scalar tensor but we feed in a tensor of dim ([1]) 
    '''
    # last_layer_bias_key = None
    # for layer in checkpoint.keys():
    #     if 'fc' in layer and 'bias' in layer:
    #         last_layer_bias_key = layer
    # checkpoint[last_layer_bias_key] = checkpoint[last_layer_bias_key].unsqueeze(0)
    last_layer_bias_key = None
    for key in checkpoint.keys():
        if 'bias' in key:
            last_layer_bias_key = key

    if last_layer_bias_key:
        checkpoint[last_layer_bias_key] = checkpoint[last_layer_bias_key].unsqueeze(0)
    return checkpoint

def test_best_config(test_dataset, model_name, model_file_name, model_folder_path):
    '''
    Test the best model configuration on a given test dataset and compute the loss.

    This function loads a pre-trained model from a specified file, configures it,
    and evaluates its performance on the provided test dataset using Mean Squared Error (MSE) loss.

    :param test_dataset: A dataset object that provides test data. It should return tuples of (input, target).
    :param model_name: The name of the model architecture to use. This should match one of the options in `choose_model`.
    :param model_file_name: The filename of the pre-trained model checkpoint to load.
    :param model_folder_path: The directory path where the model checkpoint file is located.
    :return: A tuple containing the test loss, the model outputs, and the target values.
    '''
    input, target = test_dataset[0]
    model_data = torch.load(Path(model_folder_path) / f'{model_file_name}',  map_location=torch.device('cpu'))
    checkpoint = model_data['model_state_dict']
    checkpoint = configure_bias(checkpoint)
    model = choose_model(model_name, input.shape, torch.zeros(1), model_data['config']).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():
        inputs, targets = test_dataset[:]
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets.unsqueeze(1))
    print(f'Test loss: {test_loss}')
    return test_loss, outputs, targets

def tune_model(data_folder_path, model_folder_path, model_name, mode = 'architecture', arch_config = None, single_split = True):
    wandb.login()
    train_func = train_k_fold

    if single_split: 
        train_func = train_single_split
    
    cpus = os.cpu_count()
    num_processes = 4
    num_gpus = torch.cuda.device_count()
    num_workers = 16 if cpus/num_processes > 16 else int(cpus/num_processes)

    train_val_dataset = create_model_dataset(data_folder_path, model_folder_path, model_name)
    config = define_hyperparameter_search_space(model_name, mode, device)
    if arch_config and mode == 'lr':
        arch_config.update(config)
        config = arch_config

    scheduler = ASHAScheduler(
        metric = "mean_val_loss",
        mode = "min", 
        grace_period = 2,
        brackets = 2,
        reduction_factor = 4
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func, dataset=train_val_dataset,
                                 model_name=model_name, model_folder_path=model_folder_path, 
                                 num_workers = num_workers, mode = mode),
            resources = {"cpu": num_workers, "gpu": num_gpus/num_processes},
        ),
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=100,
            max_concurrent_trials=num_processes
        ), 
        run_config=ray.air.config.RunConfig(
            name = "tune_model",
            verbose=1
        )
    )

    # Run the tuning
    results = tuner.fit()
    best_result = results.get_best_result(metric="mean_val_loss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics
    print("Best hyperparameters found were: ", best_config)
    print("Best validation loss found was: ", best_metrics['mean_val_loss'])

    best_settings_map = {
        "config": best_config,
        "loss": best_metrics['mean_val_loss']
    }
    with open (f'{model_folder_path}/best_config_{mode}_{model_name}.json', 'w') as f:
        json.dump(best_settings_map, f)

def tune_best_arch(model_name, data_folder_path, model_folder_path, config_name):
    with open (f'{model_folder_path}/{config_name}.json', 'r') as f:
        run_details = json.load(f)
    best_model_arch_config = run_details["config"]
    train_val_dataset = create_model_dataset(data_folder_path, model_folder_path, model_name)
    tune_model(data_folder_path, model_folder_path, model_name, mode = 'lr', arch_config = best_model_arch_config, single_split = True)
    
def train_best_config(model_name, data_folder_path, model_folder_path, config_name):
    with open (f'{model_folder_path}/{config_name}.json', 'r') as f:
        best_config = json.load(f)
    train_val_dataset = create_model_dataset(data_folder_path, model_folder_path, model_name)
    train_single_split(best_config["config"], train_val_dataset, model_name, model_folder_path, num_workers = 16, mode = 'lr')

def k_fold_best_config(model_name, data_folder_path, model_folder_path, config_name):
    with open (f'{model_folder_path}/{config_name}.json', 'r') as f:
        best_config = json.load(f)
    train_val_dataset = create_model_dataset(data_folder_path, model_folder_path, model_name)
    num_workers = 16 if os.cpu_count() > 16 else os.cpu_count()
    train_k_fold(best_config["config"], train_val_dataset, model_name, model_folder_path, num_workers = num_workers, num_folds = 10)

def main():
    data_folder_path = '../Data/NetCDFFiles'
    checkpoint_path = None
    model_name = 'DeepMLP'
    model_folder_path = f'/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/SavedModels/{model_name}'
    
    # tune_model_arch(data_folder_path, model_folder_path, model_name, single_split = True)
    # tune_best_arch(model_name, data_folder_path, model_folder_path, 'best_config_8_28_24_b')

    # train_best_config(model_name, data_folder_path, model_folder_path, 'best_config_lr_8_29_24')
    k_fold_best_config(model_name, data_folder_path, model_folder_path, 'best_config_lr_8_29_24')
    
    # test_dataset = torch.load(f'{model_folder_path}/{model_name}_test_dataset.pth',  map_location=torch.device('cpu'))
    # model_file_name = '/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/SavedModels/LSTM/best_model_LSTMhidden_dim64num_layers2lr0.00019361424297303123weight_decay2.450336447031607e-06batch_size256max_epochs500.pth'
    # test_best_config(test_dataset, model_name, model_file_name, model_folder_path)
    
if __name__ == '__main__':
    main()
