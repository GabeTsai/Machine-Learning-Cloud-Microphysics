import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import numpy as np
from pathlib import Path
from Models.CNNs.CNNModels import SimpleCNN, LargerCNN
from Models.MLP.MLPModel import MLP
from Models.MLP.MLPDataUtils import create_MLP_dataset
from Models.LSTM.LSTMDataUtils import *
from Models.LSTM.LSTMModel import LSTM
from Models.GEN import *

from Visualizations import plot_losses
from sklearn.model_selection import KFold
import pandas as pd
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import os
import json
import wandb
wandb.require("core")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust this as needed

torch.manual_seed(3407) #is all you need
torch.cuda.manual_seed(3407)

# ray.init(num_gpus = 1)
print(device)
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

def save_checkpoint(epoch, model, model_name, optimizer, folder, fold):
    """
    Save model checkpoint.

    Args:
        epoch (int): The epoch number.
        model (torch.nn.Module): The model object.
        model_name (str): The name of the model.
        optimizer (torch.optim.Optimizer): The optimizer object.
        folder (str): The folder where the checkpoint will be saved.
        fold (int): The fold number.
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    filename = f'{model_name}{fold}.pth.tar'
    torch.save(state, Path(folder) / filename)

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
    if 'MLP' in model_name:
        return MLP(input_dim[0], output_bias, config["hl1"], config["hl2"])
    elif model_name == 'LSTM':
        return LSTM(input_dim[1], config["hidden_dim"], config["num_layers"], output_bias) #inputs: (max_seq_len, num_features)
    elif model_name == 'GEN':
        latent_dim = config["latent_dim"]
        encoder = MLPEncoder([input_dim[0], latent_dim, latent_dim, latent_dim])
        node_mapper = MLPNodeStateMapper(latent_dim, latent_dim)
        processor = Processor(latent_dim)
        pooling_layer = GlobalPooling()
        decoder = MLPDecoder([latent_dim, latent_dim, 1], output_bias)
        return GEN(encoder, node_mapper, processor, pooling_layer, decoder)
    else:
        raise ValueError('Model name not recognized.')
    
def define_hyperparameter_search_space(model_name, device):
    """
    Define hyperparameter search space. Adjust as necessary.
    
    Args:
        model_name (str): The name of the model.
        device (torch.device): The device to use.
    
    Returns:
        dict: The hyperparameter search space.
    """
    
    if 'MLP' in model_name:
        if str(device) == 'cuda':
            batch_size_list = [128,256,512,1024]
        else:
            batch_size_list = [16, 32, 64]
        return {
        "hl1": tune.sample_from(lambda _: 2 ** np.random.randint(5, 8)),
        "hl2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 6)),
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice(batch_size_list),
        "max_epochs": 1000
        }
    elif model_name == 'LSTM': 
        return {
            "hidden_dim": tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
            "num_layers": tune.choice([1, 2, 3, 4]),
            "lr": tune.loguniform(1e-5, 1e-3),
            "weight_decay": tune.loguniform(1e-6, 1e-4),
            "batch_size": tune.choice([128, 256, 512]),
            "max_epochs": 500
        }
    elif model_name == 'GEN':
        return {
            "latent_dim": 512,
            "lr": 1e-6,
            "weight_decay": tune.loguniform(1e-3, 1e-1),
            "batch_size": 32,
            "max_epochs": 10
        }
    else:
        return {}

def load_data(data_folder_path, model_folder_path, model_name):
    '''
    Load data for specific type of model from data folder path
    '''
    def create_MLP_dataset_wrapper():
        return create_MLP_dataset(data_folder_path, model_name, model_folder_path, include_qr_nr=False)

    def create_LSTM_dataset_wrapper():
        return create_LSTM_dataset(data_folder_path, model_folder_path, model_name)

    def create_GEN_dataset_wrapper():
        log_map = {
            'qc_autoconv': True,
            'nc_autoconv': False,
            'tke_sgs': True,
            'auto_cldmsink_b': True}
        data_file_name = '00d-03h-00m-00s-000ms.h5'
        data_map = prepare_hdf_dataset(Path(data_folder_path) / data_file_name)
        return create_GEN_dataset_subset(data_map, log_map, percent = 0.2)
    
    model_data_loaders = {
        'MLP2': create_MLP_dataset_wrapper,
        'MLP3': create_MLP_dataset_wrapper,
        'LSTM': create_LSTM_dataset_wrapper,
        'GEN': create_GEN_dataset_wrapper
    }

    if model_name not in model_data_loaders:
        raise ValueError(f"Unsupported model_name: {model_name}")

    data = model_data_loaders[model_name]()

    if data[1] is None:
        return data[0]  # Only input_data and target_data for models like MLP2
    else:
        return data  # input_data, target_data, and length_data for models like LSTM

def train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, max_num_epochs, decay_lr_at, start_epoch, model_folder_path, model_name):
    '''
    Train the model on a single train/test/val split until validation loss has not reached a new minimum in early_stop epochs.

    :param model: model to train
    :param criterion: loss function
    :param optimizer: optimizer
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param early_stop: number of epochs to stop if no improvement
    :param max_num_epochs: maximum number of epochs to train
    :param decay_lr_at: list of epochs to decay learning rate
    :param start_epoch: starting epoch
    :param model_folder_path: folder to save model
    :param model_name: name of model
    :param is_lstm: boolean flag indicating if the model is an LSTM
    '''

    def train_epoch_lstm():
        model.train()
        epoch_loss = 0.0
        for inputs, targets, lengths in train_loader:
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.cpu()
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

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

    def train_epoch_GEN(epoch):
        model.train()
        epoch_loss = 0.0
        
        batch = 0
        accum_batch_loss = 0
        num_batch = 100
        print(len_train_loader)
        for inputs, targets in train_loader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) #clip gradients to -1 to 1
        return epoch_loss / len(train_loader)

    def validate_epoch_GEN(epoch):
        model.eval()
        val_epoch_loss = 0.0

        batch = 0
        accum_batch_loss = 0
        num_batch = 1
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_epoch_loss += loss.item()
                if batch % num_batch == 1:
                    print(f'Epoch {epoch}: Averaged val loss over {num_batch} batches {int(batch/len(train_loader))}: {accum_batch_loss/num_batch}')
                    wandb.log({f"Val loss over {num_batch} batches": accum_batch_loss/num_batch})
                    accum_batch_loss = 0
                batch += 1
        return val_epoch_loss / len(val_loader)

    def validate_epoch_lstm():
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for inputs, targets, lengths in val_loader:
                inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.cpu()
                outputs = model(inputs, lengths)
                loss = criterion(outputs, targets)
                val_epoch_loss += loss.item()
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
    if 'LSTM' in model_name:
        train_epoch = train_epoch_lstm
        validate_epoch = validate_epoch_lstm
    elif 'GEN' in model_name:
        train_epoch = train_epoch_GEN
        validate_epoch = validate_epoch_GEN
    else:
        train_epoch = train_epoch_default
        validate_epoch = validate_epoch_default
    
    train_losses = []
    val_losses = []
    min_val_loss = np.inf
    best_model = None
    epoch = start_epoch
    early_stop_counter = 0

    while early_stop_counter < early_stop and epoch < max_num_epochs:
        if epoch in decay_lr_at:
            decay_lr(optimizer, 0.1)

        avg_train_loss = train_epoch(epoch)
        train_losses.append(avg_train_loss)

        avg_val_loss = validate_epoch(epoch)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch} || training loss: {avg_train_loss} || validation loss: {avg_val_loss}')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        epoch += 1

    print(f'Min validation loss: {min_val_loss}')
    wandb.finish()
    return min_val_loss, best_model

def train_single_split(config, dataset, model_name, model_folder_path, num_workers, collate_fn=None):
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

    best_model = None

    model = choose_model(model_name, input.shape, output_bias, config).to(device)

    val_percent = 0.2
    val_size = int(val_percent * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size= int(config["batch_size"]), num_workers = num_workers, pin_memory = True, collate_fn = collate_fn)
    
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size = int(config["batch_size"]), num_workers = num_workers, pin_memory = True, collate_fn = collate_fn)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])

    start_epoch = 0
    early_stop = int(config["max_epochs"]/5)
    decay_lr_at = []

    wandb.init(
      project=f"{model_name}",
      config = config
      )

    wandb.log(config)
    val_loss, best_model = train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, 
                                        config["max_epochs"], decay_lr_at, start_epoch, model_folder_path, model_name)

    model_key = ''.join(f'{key}{str(item)}' for key, item in config.items()) # Create unique file key based on model config
    model_data = {
        'model_state_dict': best_model, #model weights
        'config': config #model hyperparams
    }
    save_path = Path(model_folder_path) / f'best_model_{model_name}{model_key}.pth'
    torch.save(model_data, save_path)  # Save the best model state
    session.report({"mean_val_loss": val_loss})

def train_k_fold(config, dataset, model_name, model_folder_path, num_workers, num_folds = 5):
    '''
    Train model using k-fold cross validation for a particular hyperparameter configuration.
    '''
    kfold = KFold(n_splits = num_folds, shuffle = True)

    val_losses = []
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

        model = choose_model(model_name, input.shape, target.shape, output_bias, config).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])

        start_epoch = 0
        early_stop = 100
        decay_lr_at = []

        val_loss, model_state = train_single(model, criterion, optimizer, train_loader, val_loader, early_stop, config["max_epochs"], decay_lr_at, start_epoch, model_folder_path, model_name)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model_state  # Update best model

    model_key = ''.join(f'{key}{str(item)}' for key, item in config.items()) # Create unique file key based on model config
    model_data = {
        'model_state_dict': best_model, #model weights
        'config': config #model hyperparams
    }
    save_path = Path(model_folder_path) / f'best_model_{model_name}{model_key}.pth'
    torch.save(model_data, save_path)  # Save the best model state
    mean_val_loss = np.mean(val_losses)
    session.report({"mean_val_loss": mean_val_loss})

def configure_bias(checkpoint):
    '''
    Adds extra dimension to output bias of saved model checkpoint. Workaround.
    PyTorch copies the bias as a scalar tensor but we feed in a tensor of dim ([1]) 
    '''
    last_layer_bias_key = None
    for layer in checkpoint.keys():
        if 'fc' in layer and 'bias' in layer:
            last_layer_bias_key = layer
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
    model = choose_model(model_name, input.shape, target.shape, torch.zeros(1), model_data['config']).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():
        inputs, targets = test_dataset[:]
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
    print(f'Test loss: {test_loss}')
    return test_loss, outputs, targets

def create_datasets(data_folder_path, model_folder_path, model_name):
    """
    Returns train/val dataset and saves test dataset as .pth file depending on model. 
    """
    data = load_data(data_folder_path, model_folder_path, model_name)
    if model_name == 'LSTM': 
        seqs, target_data, length_data = data
        test_seqs, test_target_data, test_len_data = create_LSTM_test_data(seqs, target_data, length_data)
        eq_inputs, eq_targets = seq_to_single(test_seqs, test_target_data)
        eq_test_dataset = torch.utils.data.TensorDataset(eq_inputs, eq_targets) 

        torch.save(eq_test_dataset, Path(model_folder_path)/ f'{model_name}_eq_test_dataset.pth')

        train_val_dataset = VariableSeqLenDataset(seqs, target_data, length_data)
        test_dataset = VariableSeqLenDataset(test_seqs, test_target_data, test_len_data)

        torch.save(eq_test_dataset, Path(model_folder_path)/ f'{model_name}_test_dataset.pth')

    else:
        input_data, target_data = data
        dataset = torch.utils.data.TensorDataset(input_data, target_data)

        test_percentage = 0.1 # 10% of data used for testing
        test_size = int(len(input_data) * test_percentage)
        k_fold_train_size = len(input_data) - test_size

        train_val_dataset, test_dataset = torch.utils.data.random_split(dataset, [k_fold_train_size, test_size])
        
        torch.save(test_dataset, Path(model_folder_path)/ f'{model_name}_test_dataset.pth')
    return train_val_dataset

def tune_model(data_folder_path, model_folder_path, model_name, single_split = False):
    wandb.login()
    train_func = train_k_fold
    if single_split: 
        train_func = train_single_split
    model_folder_path = f'/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/SavedModels/{model_name}'
    # model_folder_path = f'/Users/HP/Documents/GitHub/Machine-Learning-Cloud-Microphysics/SavedModels/{model_name}'

    cpus = os.cpu_count()
    num_processes = 1
    num_gpus = 1
    num_workers = 16 if cpus/num_processes > 16 else cpus/num_processes

    train_val_dataset = create_datasets(data_folder_path, model_folder_path, model_name)
    config = define_hyperparameter_search_space(model_name, device)
    
    scheduler = ASHAScheduler(
        metric = "mean_val_loss",
        mode = "min", 
        grace_period = 1,
        reduction_factor = 3
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func, dataset=train_val_dataset,
                                 model_name=model_name, model_folder_path=model_folder_path, 
                                 num_workers = num_workers),
            resources = {"cpu": num_workers, "gpu": num_gpus/num_processes},
        ),
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=1,
            max_concurrent_trials=1
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

    with open(f'{model_folder_path}/best_config.txt', 'w') as f:
        f.write(f"Best hyperparameters: {best_config}\n")
        f.write(f"Best validation loss: {best_metrics['mean_val_loss']}\n")

    best_settings_map = {
        "config": best_config,
        "loss": best_metrics['mean_val_loss']
    }
    with open (f'{model_folder_path}/best_config_{model_name}.json', 'w') as f:
        json.dump(best_settings_map, f)

def main():
    data_folder_path = '../Data/'
    checkpoint_path = None
    model_name = 'GEN'
    single_split = True
    
    tune_model(data_folder_path, checkpoint_path, model_name, single_split)
    # test_dataset = torch.load(f'{model_folder_path}/{model_name}_test_dataset.pth',  map_location=torch.device('cpu'))
    # model_file_name = '/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/SavedModels/LSTM/best_model_LSTMhidden_dim64num_layers2lr0.00019361424297303123weight_decay2.450336447031607e-06batch_size256max_epochs500.pth'
    # test_best_config(test_dataset, model_name, model_file_name, model_folder_path)
    
if __name__ == '__main__':
    main()
