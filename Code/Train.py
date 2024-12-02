import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR, LinearLR

import numpy as np
from pathlib import Path
from MLPModel import *
from DeepMLPModel import DeepMLP, EnsembleDeepMLP
import Layers
from DataUtils import create_model_dataset, save_data_info
from Visualizations import histogram
import config

from sklearn.model_selection import KFold
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air.config import FailureConfig
import os
import json
import wandb

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(3407) #is all you need
torch.cuda.manual_seed(3407)

def reset_weights(model):
    """
    Reset weights to avoid weight leakage.

    Args:
        model (torch.nn.Module): The model whose weights will be reset.
    """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def choose_model(model_name, input_dim, output_dim, output_bias, config, **kwargs):
    """
    Choose model based on model name.

    Args:
        model_name (str): The name of the model to choose.
        input_dim (tuple): The input dimensions.
        output_bias (bool): Whether to use output bias.
        config (dict): The configuration dictionary.
        **kwargs: model folder

    Returns:
        torch.nn.Module: The chosen model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available:
        raise ValueError("cuda_available is False")

    if 'Ensemble' in model_name:
        ensemble_models = nn.ModuleList()
        model_folder_path = kwargs.get("model_folder_path", None)
        ensemble_folder = os.path.join(model_folder_path, "EnsembleModels")
        ensemble_configs = os.path.join(model_folder_path, "EnsembleConfigs")

        for model_filename, config_file_name in zip(sorted(os.listdir(ensemble_folder)), sorted(os.listdir(ensemble_configs))):
            full_model_config_path = os.path.join(ensemble_configs, config_file_name)
            full_model_path = os.path.join(ensemble_folder, model_filename)
            
            with open(full_model_config_path, 'r') as f:
                model_config = json.load(f)["config"]
            
            model = DeepMLP(input_dim[0], model_config["latent_dim"], 1, output_bias, num_blocks=model_config["num_blocks"])
            
            checkpoint = torch.load(full_model_path, map_location=device)
            
            state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(state_dict)

            model = model.to(device)
            
            ensemble_models.append(model)
        
        # Return the ensemble model, ensuring it's on the correct device
        return EnsembleDeepMLP(ensemble_models, config["latent_dim"], output_dim, output_bias, num_blocks=config["num_blocks"], 
                                activation = nn.LeakyReLU(), freeze_ensemble_weights=False).to(device)
    elif 'DeepMLP' in model_name:
        return DeepMLP(input_dim, config["latent_dim"], output_dim, output_bias, num_blocks = config["num_blocks"], activation = nn.LeakyReLU())
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
    elif 'DeepMLP' in model_name or 'Ensemble' in model_name:
        #DeepMLP tuning has two modes: architecture and learning rate.
        #Tune the architecture first and save the best model
        #Then, use the best model as a starting point to tune lr related parameters.
        if mode == 'architecture':
            return {
                    "latent_dim": tune.randint(32, 256),
                    "num_blocks": tune.choice([i for i in range(3, 16)]),
                    "max_lr": 1e-4,
                    "batch_size": 1024,
                    "max_epochs": 50
                    }
        elif mode == 'lr': 
            return {
                    "min_lr_factor": tune.choice([1e-12, 1]),
                    "inc_lr_perc": tune.uniform(0.01, 0.10),
                    "max_lr": tune.loguniform(1e-5, 1e-3),
                    "hold_lr_perc": tune.choice([i / 10 for i in range(2, 11)]),
                    "gamma": tune.uniform(0.99993, 0.99997),
                    "weight_decay": tune.loguniform(0.001, 0.01),
                    "max_epochs": 100
            }
        else:
            raise ValueError('Invalid mode')
    else:
        return {}

def train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, start_epoch, 
                 model_name, hyperparam_config, scheduler = None, checkpoint_name = None):
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

    if checkpoint_name: 
        model = load_checkpoint(model_name, checkpoint_name, f'{config.MODEL_FOLDER_PATH}/{model_name}')

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    if scheduler:
        max_lr = get_lr(optimizer)
        total_num_steps = len(train_loader) * hyperparam_config["max_epochs"]
        inc_lr_until = int(hyperparam_config["inc_lr_perc"] * total_num_steps)
        inc_amount = max_lr/inc_lr_until
        hold_lr_lim = int(hyperparam_config["hold_lr_perc"] * total_num_steps) + inc_lr_until
        linear_scheduler = LinearLR(optimizer, start_factor = hyperparam_config["min_lr_factor"], total_iters = inc_lr_until)
    
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

    # Choose the appropriate functions
    if 'DeepMLP' in model_name or 'Ensemble' in model_name: 
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

    while epoch < hyperparam_config["max_epochs"]:
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

def train_single_split(hyperparam_config, dataset, model_name, model_folder_path, dataset_name, num_workers, mode, checkpoint_name, collate_fn=None):
    '''
    Train on a single train-valid split (provided dataset is big enough) for a particular hyperparameter config. 
    '''
    train_dataset, val_dataset = dataset[0], dataset[1]
    if dataset_name != 'all':
        region_id = config.REGION_TAG_MAP[dataset_name]
        
        # Filter training dataset
        train_regions = train_dataset.tensors[2]
        train_mask = (train_regions == region_id)
        train_inputs = train_dataset.tensors[0][train_mask]
        train_targets = train_dataset.tensors[1][train_mask]
        print(train_inputs.shape)
        train_dataset = TensorDataset(train_inputs, train_targets)  # Create new dataset

        # Filter validation dataset
        val_regions = val_dataset.tensors[2]
        val_mask = (val_regions == region_id)
        val_inputs = val_dataset.tensors[0][val_mask]
        val_targets = val_dataset.tensors[1][val_mask]
        print(val_inputs.shape)
        val_dataset = TensorDataset(val_inputs, val_targets)  # Create new dataset
    
    input_dim, target_dim = config.NUM_INPUTS, config.NUM_OUTPUTS
    targets = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
    output_bias = torch.mean(targets)

    model = choose_model(model_name, input_dim, target_dim, output_bias, hyperparam_config, model_folder_path = model_folder_path).to(device)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size= int(hyperparam_config["batch_size"]), shuffle = True, num_workers = num_workers, pin_memory = True, collate_fn = collate_fn)
    
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size = int(hyperparam_config["batch_size"]), shuffle = True, num_workers = num_workers, pin_memory = True, collate_fn = collate_fn)

    criterion = nn.MSELoss()
    if mode == 'architecture':
        optimizer = torch.optim.AdamW(model.parameters(), lr = hyperparam_config["max_lr"])
        scheduler = None
    elif mode == 'lr':
        optimizer = torch.optim.AdamW(model.parameters(), lr = hyperparam_config["max_lr"], weight_decay = hyperparam_config["weight_decay"])
        scheduler = ExponentialLR(optimizer, gamma = hyperparam_config["gamma"])
    else:
        raise ValueError('Invalid mode')

    start_epoch = 0
    early_stop = int(hyperparam_config["max_epochs"] * 0.1)

    wandb.init(
      project=f"{model_name}{mode}{dataset_name}",
      config = hyperparam_config
      )

    wandb.log(hyperparam_config)
    val_loss, best_model = train_single(model, criterion, optimizer, train_loader, val_loader, early_stop,
                                        start_epoch, model_name, hyperparam_config, scheduler = scheduler, checkpoint_name = checkpoint_name)

    model_key = ''.join(f'{key}{str(item)}' for key, item in hyperparam_config.items()) # Create unique file key based on model config
    model_data = {
        'model_state_dict': best_model, #model weights
        'config': hyperparam_config #model hyperparams
    }
    save_path = Path(model_folder_path) / f'best_model_{model_name}_{dataset_name}_{model_key}.pth'
    torch.save(model_data, save_path)  # Save the best model states

def train_k_fold(hyperparam_config, dataset, model_name, model_folder_path, num_workers, num_folds = 10):
    '''
    Train model using k-fold cross validation for a particular hyperparameter configuration.
    '''
    kfold = KFold(n_splits = num_folds, shuffle = True)
    
    input_dim, target_dim = config.NUM_INPUTS, config.NUM_OUTPUTS
    merged_dataset = ConcatDataset([dataset[0], dataset[1]])

    best_loss = np.inf
    best_model = None
    for fold, (train_i, val_i) in enumerate(kfold.split(merged_dataset)):
        print(f'Fold {fold}--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_i)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_i)

        train_loader = torch.utils.data.DataLoader(
            merged_dataset, batch_size= int(hyperparam_config["batch_size"]), sampler=train_subsampler, num_workers = num_workers, pin_memory = True)

        val_loader = torch.utils.data.DataLoader(
            merged_dataset, batch_size = int(hyperparam_config["batch_size"]), sampler=val_subsampler, num_workers = num_workers, pin_memory = True)

        all_targets = []
        for _, targets in train_loader:
            all_targets.append(targets)
        all_targets = torch.cat(all_targets, dim=0)
        output_bias = torch.mean(all_targets)

        model = choose_model(model_name, input_dim, target_dim, output_bias, hyperparam_config, model_folder_path = model_folder_path).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = hyperparam_config["max_lr"], weight_decay = hyperparam_config["weight_decay"])
        scheduler = ExponentialLR(optimizer, gamma = hyperparam_config["gamma"])

        start_epoch = 0
        early_stop = int(hyperparam_config["max_epochs"] * 0.1)

        wandb.init(
        project=f"{model_name}",
        config = hyperparam_config
        )

        wandb.log(hyperparam_config)
        wandb.log({"fold": fold})

        val_loss, best_model_fold = train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, 
                                                start_epoch, model_name, hyperparam_config, scheduler = scheduler)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = best_model_fold  # Update best model

    model_key = ''.join(f'{key}{str(item)}' for key, item in hyperparam_config.items()) # Create unique file key based on model config
    model_data = {
        'model_state_dict': best_model, #model weights
        'config': hyperparam_config #model hyperparams
    }
    save_path = Path(model_folder_path) / f'best_model_k_fold_{model_name}{model_key}.pth'
    torch.save(model_data, save_path)  # Save the best model state

def configure_bias(checkpoint):
    '''
    Adds extra dimension to output bias of saved model checkpoint. Workaround.
    PyTorch copies the bias as a scalar tensor but we feed in a tensor of dim ([1]) 
    '''
    
    last_layer_bias_key = None

    for key in checkpoint.keys():
        if 'bias' in key:
            last_layer_bias_key = key
    
    if last_layer_bias_key:
        checkpoint[last_layer_bias_key] = checkpoint[last_layer_bias_key]

    return checkpoint

def load_checkpoint(model_name, model_file_name, model_folder_path):
    input_dim, target_dim = config.NUM_INPUTS, config.NUM_OUTPUTS
    model_data = torch.load(Path(model_folder_path) / f'{model_file_name}',  map_location=torch.device('cpu'))
    checkpoint = model_data['model_state_dict']

    model = choose_model(model_name, input_dim, target_dim, torch.zeros(1), model_data['config'], model_folder_path = model_folder_path).to(device)

    if model_name == "Ensemble":
        meta_learner_checkpoint = {key: value for key, value in checkpoint.items() if "meta_learner" in key}
        model.load_state_dict(meta_learner_checkpoint, strict = False)
    else:
        model.load_state_dict(checkpoint)
    
    return model

def test_best_config(test_dataset, model_name, model_file_name, model_folder_path, region = None):
    '''
    Test the best model configuration on a given test dataset and compute the loss.

    This function loads a pre-trained model from a specified file, configures it,
    and evaluates its performance on the provided test dataset using Mean Squared Error (MSE) loss.

    :param model_name: The name of the model architecture to use. This should match one of the options in `choose_model`.
    :param model_file_name: The filename of the pre-trained model checkpoint to load.
    :param model_folder_path: The directory path where the model checkpoint file is located.
    :param region (str, optional): Specify if you want inference on a particular region
    :return: A tuple containing the test loss, the model outputs, and the target values.
    '''
    model = load_checkpoint(model_name, model_file_name, model_folder_path)
        
    model.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():
        inputs, targets = [], []
        
        inputs = test_dataset.tensors[0]
        targets = test_dataset.tensors[1]
        if region:
            regions = test_dataset.tensors[2]
            region_id = config.REGION_TAG_MAP[region]
            mask = (regions == region_id)
            
            inputs = inputs[mask]
            targets = targets[mask]
            

        # Concatenate the inputs and targets
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
    print(f'Test loss: {test_loss}')
    return test_loss, outputs, targets

def tune_model(data_folder_path, model_folder_path, model_name, mode = 'architecture', dataset_name = "", arch_config = None, single_split = True):
    wandb.login()
    train_func = train_k_fold

    if single_split: 
        train_func = train_single_split

    train_val_dataset = create_model_dataset(data_folder_path, model_folder_path, model_name, dataset_name = dataset_name)
    hyperparam_config = define_hyperparameter_search_space(model_name, mode, device)

    if arch_config and mode == 'lr':
        arch_config.update(hyperparam_config)
        hyperparam_config = arch_config

    scheduler = ASHAScheduler(
        metric = "mean_val_loss", 
        mode = "min", 
        grace_period = config.GRACE_PERIOD,
        brackets = config.BRACKETS,
        reduction_factor = config.REDUCTION_FACTOR
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func, dataset=train_val_dataset,
                                 model_name=model_name, model_folder_path=model_folder_path, 
                                 dataset_name = dataset_name, 
                                 num_workers = config.NUM_WORKERS, mode = mode, checkpoint_name = None),
            resources = {"cpu": config.NUM_WORKERS, "gpu": config.NUM_GPUS/config.NUM_PROCESSES},
        ),
        param_space=hyperparam_config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=config.NUM_SAMPLES,
            max_concurrent_trials=config.NUM_PROCESSES
        ), 
        run_config=ray.air.config.RunConfig(
            name = "tune_model",
            verbose=1, 
            failure_config = FailureConfig(max_failures=100) 
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
    with open (f'{model_folder_path}/best_config_{mode}_{model_name}_{dataset_name}.json', 'w') as f:
        json.dump(best_settings_map, f)

def tune_best_arch(model_name, data_folder_path, model_folder_path, config_name, dataset_name):
    with open (f'{model_folder_path}/{config_name}.json', 'r') as f:
        run_details = json.load(f)
    best_model_arch_config = run_details["config"]
    tune_model(data_folder_path, model_folder_path, model_name, mode = 'lr', dataset_name = dataset_name, arch_config = best_model_arch_config, single_split = True)
    
def train_best_config(model_name, data_folder_path, model_folder_path, dataset_name, config_name, checkpoint_name = None, mode = 'lr'):
    with open (f'{model_folder_path}/{config_name}.json', 'r') as f:
        best_config = json.load(f)
    train_val_dataset = create_model_dataset(data_folder_path, model_folder_path, model_name, dataset_name = dataset_name)
    train_single_split(best_config["config"], train_val_dataset, model_name, model_folder_path, 
                    dataset_name, num_workers = config.NUM_WORKERS, mode = mode, checkpoint_name = checkpoint_name)

def k_fold_best_config(model_name, data_folder_path, model_folder_path, config_name, dataset_name):
    with open (f'{model_folder_path}/{config_name}.json', 'r') as f:
        best_config = json.load(f)
    train_val_dataset = create_model_dataset(data_folder_path, model_folder_path, model_name, dataset_name = dataset_name)
    num_workers = 16 if os.cpu_count() > 16 else os.cpu_count()
    train_k_fold(best_config["config"], train_val_dataset, model_name, model_folder_path, num_workers = num_workers, num_folds = 10)

def main():
    data_name = 'all'
    data_folder_path = f'../Data/{data_name}'
    checkpoint_path = None
    model_type = 'DeepMLP'
    model_folder_path = f'{config.MODEL_FOLDER_PATH}/{model_type}'
    # tune_model(config.DATA_FOLDER_PATH, model_folder_path, model_type, dataset_name = data_handle, single_split = True)
    # tune_best_arch(model_type, f"{data_folder_path}", model_folder_path, config_name = f"best_config_architecture_{model_type}_{data_handle}", dataset_name = data_handle)
    # train_best_config(model_type, data_folder_path, model_folder_path, data_name, config.BEST_MODEL_CONFIG_NAME, mode = 'lr') 
    # tune_model_arch(data_folder_path, model_folder_path, model_name, single_split = True)
    # tune_best_arch(model_name, data_folder_path, model_folder_path, 'best_config_8_28_24_b')
    # k_fold_best_config(model_type, data_folder_path, model_folder_path, config_name = 'best_config_lr_DeepMLP_11_1_24_kfold', dataset_name = "")
    for region in config.REGIONS:
        train_best_config(model_type, f"{data_folder_path}", model_folder_path, region, config.BEST_MODEL_CONFIG_NAME, mode = 'lr')
    # train_best_config(model_type, f"{data_folder_path}/{dataset_name}", model_folder_path, dataset_name, f'best_config_{dataset_name}')
    # k_fold_best_config(model_name, data_folder_path, model_folder_path, 'best_config_lr_8_29_24')
    
    
if __name__ == '__main__':
    main()
