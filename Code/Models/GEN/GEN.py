import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Transform input data to latent vector

    Args:
        layer_dims (list): List of dimensions for each layer in the encoder
    """
    def __init__(self, layer_dims):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class Decoder(nn.Module):
    """
    Transform latent vector obtained from mesh processor to output

    Args:
        layer_dims (list): List of dimensions for each layer in the decoder
        output_bias (torch.Tensor): The initial bias for the output layer. Initialized to mean of output data
    """
    def __init__(self, layer_dims, output_bias):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        self.layers[-1].bias = torch.nn.Parameter(output_bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GEN(nn.Module):
    def __init__(self):
        super(GEN, self).__init__()
    
    def representation_fn(self, x, node_pos):
        '''
        Representation function to map input data to latent space
        '''
        pass