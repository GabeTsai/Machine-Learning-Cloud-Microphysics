import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import numpy as np
from pathlib import Path
from Model import SimpleCNN, LargerCNN
from DataUtils import create_dataset
from sklearn.model_selection import KFold
import json

NUM_HEIGHT_LEVELS = 24
torch.manual_seed(99)

def reset_weights(model):
    '''
    Reset weights to avoid weight leakage
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train(model, criterion, optimizer, train_loader, epochs, model_folder_path):
    '''
    Train model using k-fold cross validation for one fold. Credit to @lucasew
    '''
    
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
            # if i % 10 == 0:
            #     print(f'Averaged loss over 10 batches after minibatch {i}: {current_loss / 10}')
            #     current_loss = 0.0

        avg_train_loss = total_loss / len(train_loader)
        print(f'Average training loss for epoch {epoch}: {avg_train_loss}')
    
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

def test_folds(targets, dataset, kfold):
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        train_targets = targets[train_idx].numpy()
        test_targets = targets[test_idx].numpy()
        print(f'Fold {fold} - Train target mean: {np.mean(train_targets)}, Test target mean: {np.mean(test_targets)}')
        print(f'Fold {fold} - Train target var: {np.var(train_targets)}, Test target var: {np.var(test_targets)}')

def metrics(results):
    '''
    Display individual model losses and average loss over all models.
    '''
    total_loss = 0
    for model_name, loss in results.items():
        print(f'{model_name} loss: {loss}')
        total_loss += loss
    print(f'Average loss: {total_loss / len(results)}')

def main():
    k_folds = 10
    epochs = 2000
    batch_size = 8
    input_data, target_data = create_dataset('../../Data')
    model_name = 'SimpleCNN'
    shape = (input_data.shape[2], )
    model = SimpleCNN(input_data.shape[2], torch.full(shape, torch.mean(target_data)))
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, weight_decay = 1e-5)
    dataset = torch.utils.data.TensorDataset(input_data, target_data)
    kfold = KFold(n_splits = k_folds, shuffle = True)
    
    model_folder_path = f'../../SavedModels/{model_name}'
    
    test_results = {}
    train_results = {}
    saved_test_indices = {}

    for kfold, (train_i, test_i) in enumerate(kfold.split(dataset)):
        print(f'Fold {kfold}--------------------------------')
        reset_weights(model)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_i)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_i)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size, sampler = train_subsampler)
        
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size = batch_size, sampler = test_subsampler)
        
        avg_train_loss = train(model, criterion, optimizer, train_loader, epochs, model_folder_path)
        train_results[f'{model_name}_{kfold}.pth'] = avg_train_loss
        print(model)
        torch.save(model.state_dict(), Path(model_folder_path) / f'{model_name}_{kfold}.pth')
        print(f'Finished training for fold {kfold}')
        avg_test_loss = test(model, criterion, test_loader, Path(model_folder_path) / f'{model_name}_{kfold}.pth')
        print(f'Averaged loss for fold {kfold}: {avg_test_loss}')

        test_results[f'{model_name}_{kfold}.pth'] = avg_test_loss
        saved_test_indices[f'Fold {kfold}'] = test_i.tolist()  # Save the test indices
    metrics(train_results)
    metrics(test_results)
    with open(Path(model_folder_path) / 'test_results.json', 'w') as f:
        json.dump(test_results, f)
    with open(Path(model_folder_path) / 'test_indices.json', 'w') as f:
        json.dump(saved_test_indices, f)  # Save the test indices dictionary    

if __name__ == '__main__':
    main()