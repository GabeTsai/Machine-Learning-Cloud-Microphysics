import netCDF4
import torch
import xarray as xr
import numpy as np
import json
import sys
sys.path.append('../../')
from DataUtils import * # Import functions and variables from CreateDataLists.py
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from Visualizations import histogram
from pathlib import Path
from sklearn.preprocessing import RobustScaler, QuantileTransformer

np.random.seed(42)
class VariableSeqLenDataset(Dataset):
    """
    Args:
        sequences (list of np.ndarray): List of sequences (each sequence is an array of shape (seq_length, num_features)).
        targets (np.ndarray): Array of targets.
        lengths (np.ndarray): Array of sequence lengths.
    """
    def __init__(self, sequences, targets, lengths):
        self.sequences = [torch.FloatTensor(seq) for seq in sequences]
        self.targets = torch.FloatTensor(targets)
        self.lengths = torch.LongTensor(lengths)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.lengths[idx]

def collate_fn(batch):
    """
    Custom collate function to handle dataloading. We sort sequences for memory efficiency.
    :param batch: batched LSTM input sequences, targets and seq lengths

    :return: sorted padded sequences
    """
    sequences, targets, lengths = zip(*batch)

    # Sort by length (descending)
    lengths = torch.LongTensor(lengths)
    lengths, perm_idx = lengths.sort(0, descending=True)
    sequences = [sequences[i] for i in perm_idx]
    targets = torch.stack([targets[i] for i in perm_idx])

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)

    return padded_sequences, targets, lengths

def prepare_dataset_LSTM(data_map, min_seq_length, max_seq_length, max_zeros=100):
    '''
    Return list of seq_length min to max seq_length and target datasets for LSTM model. 
    '''
    seq_intervals = [1, 5, 10, 15, 30]
    qc_autoconv_cloud = np.array(data_map['qc_autoconv_cloud'])
    nc_autoconv_cloud = np.array(data_map['nc_autoconv_cloud'])
    auto_cldmsink_b_cloud = np.array(data_map['auto_cldmsink_b_cloud'])

    inputs = []
    targets = []
    lengths = []
    for seq_length in range(min_seq_length, max_seq_length + 1):  # for all seq lengths,
        for seq_int in seq_intervals:
            for i in range(0, len(qc_autoconv_cloud[0]) - seq_length - seq_int, seq_int):  # apply a sliding window of size seq_length to extract input sequences
                qc_seq = qc_autoconv_cloud[:, i:i + seq_length]
                nc_seq = nc_autoconv_cloud[:, i:i + seq_length]
                target = auto_cldmsink_b_cloud[:, i + seq_length]

                for height_level in range(qc_seq.shape[0]):  # Iterate over each height level/batch_dimension
                    sequence = np.stack((qc_seq[height_level], nc_seq[height_level]), axis=-1)  # (seq_length, num_features)
                    inputs.append((sequence))  # Append each height level sequence separately
                    targets.append(target[height_level])  # Append the target for the corresponding height level
                    lengths.append(seq_length)  # Store the sequence length
    assert(len(inputs) == len(targets) == len(lengths))
    return inputs, targets, lengths

def create_LSTM_test_data(input_list, target_array, len_array):
    '''
    Partition data into 90-10 split. Modify input arrays, return test data arrays.
    '''
    total_size = len(input_list)
    test_size = int(0.1 * total_size)
    tv_size = total_size - test_size

    # Randomly select indices for the test set
    all_indices = np.arange(total_size)
    np.random.shuffle(all_indices)
    test_indices = all_indices[:test_size]
    tv_indices = all_indices[test_size:]

    # Extract test data
    test_list = [input_list[i] for i in test_indices]
    test_target_array = target_array[test_indices]
    test_len_array = len_array[test_indices]

    # Create new arrays for training/validation data
    tv_list = [input_list[i] for i in tv_indices]
    tv_target_array = target_array[tv_indices]
    tv_len_array = len_array[tv_indices]

    # Replace input arrays with training/validation data
    input_list = tv_list
    target_array = tv_target_array
    len_array = tv_len_array

    assert len(input_list) == len(target_array) == len(len_array)
    return test_list, test_target_array, test_len_array
    
def seq_to_single(input_list, target_array):
    '''
    Convert sequential inputs back to single inputs and outputs for KK2000 equation. 
    Returns single inputs and corresponding outputs as tensors. 
    '''
    sinput_list = []
    starget_list = []
    for i in range(1, len(input_list)): 
        seq = input_list[i]
        sinput_list.append(seq[-1, :]) #get the last values of the input sequence, corresponding to
        starget_list.append(target_array[i - 1]) #the previous target
    sinput_arr = np.vstack(sinput_list)
    return torch.FloatTensor(sinput_arr), torch.FloatTensor(np.array(starget_list))

def concat_data(data_maps, model_folder_path, min_seq_length, max_seq_length, model_name):
    input_list = []
    target_list = []
    len_list = []

    for data_map in data_maps:
        inputs, targets, lengths = prepare_dataset_LSTM(data_map, min_seq_length, max_seq_length)
        input_list = input_list + inputs
        target_list = target_list + targets
        len_list = len_list + lengths

    # Convert lists to NumPy arrays
    input_array = np.array(input_list, dtype=object)  # dtype=object for variable-length sequences
    target_array = np.array(target_list)
    len_array = np.array(len_list)

    # Mask out outliers
    filter_mask = remove_outliers(target_array)
    target_array = target_array[filter_mask]
    input_array = input_array[filter_mask]
    len_array = len_array[filter_mask]

    assert input_array.shape[0] == target_array.shape[0] == len_array.shape[0]

    # Concatenate the inputs
    seq_shapes = [seq.shape for seq in input_array]
    num_rows = sum(shape[0] for shape in seq_shapes)
    merged_input_arr = np.vstack(input_array)
    
    save_data_info(merged_input_arr, target_array, model_folder_path, model_name)
    merged_input_arr = min_max_normalize(merged_input_arr, dims = (0))
    input_list = []
    start = 0
    for shape in seq_shapes:
        end = start + shape[0]
        sliced_arr = merged_input_arr[start:end, :]
        input_list.append(torch.FloatTensor(sliced_arr.reshape(shape)))
        start = end

    return input_list, target_array, len_array

def create_LSTM_dataset(data_folder_path, model_folder_path, model_name):
    min_seq_length = 3 
    max_seq_length = 6
    data_maps = prepare_datasets(data_folder_path)
    inputs, targets, lengths = concat_data(data_maps, model_folder_path, min_seq_length, max_seq_length, model_name)
    return inputs, targets, lengths

def main():
    # data_folder_path = '../../../Data/NetCDFFiles'
    # model_folder_path = '../../../SavedModels/LSTM'
    
    # inputs, targets, lengths = create_LSTM_dataset(data_folder_path, model_folder_path, 'LSTM')
    # print(len(inputs))
    # print(targets.shape)
    # print(lengths.shape)
    # histogram(targets, targets, 'LSTM', '../../../Visualizations')
    test()


if __name__ == "__main__":
    main()