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
from Visualizations import plot_losses
from sklearn.model_selection import KFold
import json

torch.manual_seed(99)

def reset_weights(model):
    '''
    Reset weights to avoid weight leakage
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train(model, criterion, optimizer, train_loader, epochs, model_folder_path, track_loss = False):
    '''
    Train model on training dataset, or for one fold of data. Credit to @lucasew
    '''
    if track_loss:
        losses = []
    for epoch in range(0, epochs):
        model.train()

        #Iterate over DataLoader
        current_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, targets = data
            
            #Zero gradients
            optimizer.zero_grad()

            #Perform forward pass
            outputs = model(inputs)

            #Compute loss
            loss = criterion(outputs, targets)

            #Perform backward pass
            loss.backward()

            #Optimize
            optimizer.step()

            current_loss += loss.item()
            total_loss += loss.item()
            if i % 10 == 0:
                print(f'Averaged loss over 10 batches after minibatch {i}: {current_loss / 10}')
                current_loss = 0.0
                
        avg_train_loss = total_loss / len(train_loader)
        if track_loss:
            losses.append(loss.item())
        print(f'Average training loss for epoch {epoch}: {avg_train_loss}')
    if track_loss:
        return losses, avg_train_loss
    else:
        return avg_train_loss
        
def test(model, criterion, test_loader, model_path):
    '''
    Test model using k-fold cross validation for one fold. Credit to @lucasew
    '''
    
    model.eval()
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        current_loss = 0.0
        for i, data in enumerate(test_loader):
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            current_loss += loss.item()
    
    return current_loss/len(test_loader)

def k_fold_training(model, criterion, optimizer, dataset, batch_size, k_folds, epochs, model_folder_path, model_name):
    kfold = KFold(n_splits=k_folds, shuffle=True)

    test_results = {}
    train_results = {}
    saved_test_indices = {}

    for fold, (train_i, test_i) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold}--------------------------------')
        reset_weights(model)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_i)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_i)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler)

        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_subsampler)

        avg_train_loss = train(model, criterion, optimizer, train_loader, epochs, model_folder_path)
        train_results[f'{model_name}_{fold}.pth'] = avg_train_loss
        print(model)
        torch.save(model.state_dict(), Path(model_folder_path) / f'{model_name}_{fold}.pth')
        print(f'Finished training for fold {fold}')
        avg_test_loss = test(model, criterion, test_loader, Path(model_folder_path) / f'{model_name}_{fold}.pth')
        print(f'Averaged loss for fold {fold}: {avg_test_loss}')

        test_results[f'{model_name}_{fold}.pth'] = avg_test_loss
        saved_test_indices[f'Fold {fold}'] = test_i.tolist()  # Save the test indices

    k_fold_metrics(train_results)
    k_fold_metrics(test_results)
    with open(Path(model_folder_path) / 'test_results.json', 'w') as f:
        json.dump(test_results, f)
    with open(Path(model_folder_path) / 'test_indices.json', 'w') as f:
        json.dump(saved_test_indices, f)  # Save the test indices dictionary    

def k_fold_metrics(results):
    '''
    Display individual model losses and average loss over all models.
    '''
    total_loss = 0
    for model_name, loss in results.items():
        print(f'{model_name} loss: {loss}')
        total_loss += loss
    print(f'Average loss: {total_loss / len(results)}')

def decay_lr(optimizer, decay_rate):
    '''
    Decay learning rate by decay_rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

def save_checkpoint(epoch, model, model_name, optimizer, folder):
    """
    Save model checkpoint

    :param epoch: epoch number
    :param model: model object
    :param optimizer: optimizer object
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    filename = f'{model_name}.pth.tar'
    torch.save(state, Path(folder) / filename)

def train_single(model, criterion, optimizer, dataset, batch_size, early_stop, decay_lr_at, start_epoch, model_folder_path, data_folder_path, model_name):
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
    train_percent = 0.8
    val_percent = 0.1

    train_size = int(train_percent * len(dataset))
    val_size = int(val_percent * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    torch.save(test_dataset, Path(model_folder_path) / f'{model_name}_test_dataset.pth')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

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
        for i, data in enumerate(train_loader):
            inputs, targets = data
            targets = targets.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        for i, data in enumerate(val_loader):
            inputs, targets = data
            targets = targets.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
        avg_val_loss = epoch_loss / len(val_loader)
        print(f'Epoch {epoch} || training loss: {avg_train_loss} || validation loss: {avg_val_loss}')
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            print(f'Saving model for lower avg min validation loss: {min_val_loss}')
            save_checkpoint(epoch, model, model_name, optimizer, model_folder_path)
            early_stop_counter = 0
        val_losses.append(avg_val_loss)
        early_stop_counter += 1
        epoch += 1
        if epoch % 5000 == 0:
            plot_losses(train_losses, val_losses, model_name)
    print(f'Min validation loss: {min_val_loss}')
    plot_losses(train_losses, val_losses, model_name)

def main():
    data_folder_path = '../Data/NetCDFFiles'
    checkpoint_path = None
    model_name = 'MLP2'
    model_folder_path = f'../SavedModels/{model_name}'
    input_data, target_data = create_MLP_dataset(data_folder_path, model_folder_path)
    print(input_data.shape)
    print(target_data.shape)
    
    #Change model specifics below
    #CNN Parameters 
    # model = LargerCNN(input_data.shape[2], torch.full(input_data.shape[2], torch.mean(target_data)))
    if checkpoint_path is None:
        model = MLP(input_data.shape[1], torch.mean(target_data))
        learning_rate = 3e-5
        weight_decay = 1e-5
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        start_epoch = 0
    else:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    dataset = torch.utils.data.TensorDataset(input_data, target_data)

    epochs = 2000
    batch_size = 64
    k_folds = 10
    early_stop = 500

    decay_lr_at = []

    # k_fold_training(model, criterion, optimizer, dataset, batch_size, k_folds, epochs, model_folder_path, model_name)
    train_single(model, criterion, optimizer, dataset, batch_size, 
                 early_stop, decay_lr_at, start_epoch, model_folder_path, data_folder_path, model_name)
    

if __name__ == '__main__':
    main()