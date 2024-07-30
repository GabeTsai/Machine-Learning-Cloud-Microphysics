import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_bias):
        '''
        LSTM-based neural network model for sequence-to-one prediction.

        :param input_dim (int): The number of features in the input sequence.
        :param hidden_dim (int): The number of features in the hidden state of the LSTM.
        :param num_layers (int): The number of recurrent layers in the LSTM.
        :param output_bias (torch.Tensor): The initial bias for the output layer. Initialized to mean of output data
        '''
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first= True) #(batch, seq_len, num_features)
        self.fc = nn.Linear(hidden_dim, 1)
        self.fc.bias = torch.nn.Parameter(output_bias)

    def forward(self, x):
        # Move h0 and c0 to the same device as x
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device).requires_grad_()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # only take the last output (many to one)
        return out