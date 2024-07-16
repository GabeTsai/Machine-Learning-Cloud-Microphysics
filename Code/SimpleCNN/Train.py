import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from pathlib import Path
from Model import SimpleCNN
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
        print(f'Epoch {epoch}--------------------------------')

        #Iterate over DataLoader
        current_loss = 0.0
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
            if i % 10 == 0:
                print(f'Averaged loss over 10 batches after minibatch {i}: {current_loss / 10}')
                current_loss = 0.0
        
def test(model, criterion, test_loader, model_path):
    '''
    Test model using k-fold cross validation for one fold. Credit to @lucasew
    '''
    
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model_performances = {}
    with torch.no_grad():
        current_loss = 0.0
        for i, data in enumerate(test_loader):
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            current_loss += loss.item()
    
    return current_loss/len(test_loader)

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
    k_folds = 5
    epochs = 500
    input_data, target_data = create_dataset('../../Data')
    model = SimpleCNN(input_data.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    dataset = torch.utils.data.TensorDataset(input_data, target_data)
    kfold = KFold(n_splits = k_folds, shuffle = True)

    model_folder_path = '../../SavedModels/SimpleCNN'
    
    results = {}
    saved_test_indices = {}
    for kfold, (train_i, test_i) in enumerate(kfold.split(dataset)):
        print(f'Fold {kfold}--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_i)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_i)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size = 8, sampler = train_subsampler)
        
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size = 8, sampler = test_subsampler)
        
        reset_weights(model)

        train(model, criterion, optimizer, train_loader, epochs, model_folder_path)
        torch.save(model.state_dict(), Path(model_folder_path) / f'SimpleCNN_{kfold}.pth')
        print(f'Finished training for fold {kfold}')
        avg_test_loss = test(model, criterion, test_loader, Path(model_folder_path) / f'SimpleCNN_{kfold}.pth')
        print(f'Averaged loss for fold {kfold}: {avg_test_loss}')

        results[f'SimpleCNN_{kfold}.pth'] = avg_test_loss
        print(test_i)
        saved_test_indices[f'Fold {kfold}'] = test_i.tolist()  # Save the test indices

    metrics(results)
    with open(Path(model_folder_path) / 'results.json', 'w') as f:
        json.dump(results, f)
    with open(Path(model_folder_path) / 'test_indices.json', 'w') as f:
        json.dump(saved_test_indices, f)  # Save the test indices dictionary    

if __name__ == '__main__':
    main()