import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import KFold

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