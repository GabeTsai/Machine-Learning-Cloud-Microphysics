import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from IcosphereMesh import IcosphereMesh

class MLPEncoder(nn.Module):
    """
    MLP to transform input data to latent vector

    Args:
        layer_dims (list): List of dimensions for each layer in the encoder
    """
    def __init__(self, layer_dims):
        super(MLPEncoder, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.encoder_MLP = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder_MLP(x)

class MLPNodeStateMapper(nn.Module):
    """
    MLP to transform latent vector + positional embedding to node states 

    Args:
        latent_dim (int): Dimension of latent vector
        pos_emb_dim (int): Dimension of positional embedding
    """
    def __init__(self, latent_dim, pos_emb_dim):
        super(MLPNodeStateMapper, self).__init__()
        self.pos_emb_dim = pos_emb_dim
        self.positional_embedding_mlp = nn.Linear(3, pos_emb_dim)
        self.node_state_mlp = nn.Linear(latent_dim + pos_emb_dim, latent_dim)

    def forward(self, x, node_pos):
        pos_emb = self.positional_embedding_mlp(node_pos)
        print(pos_emb.shape)
        combined = torch.cat((x.expand(pos_emb.shape[0], -1), pos_emb), dim=1)
        print(combined.shape)
        node_states = self.node_state_mlp(combined)
        return node_states

class MLPDecoder(nn.Module):
    """
    MLP to transform latent vector obtained from mesh processor to output

    Args:
        layer_dims (list): List of dimensions for each layer in the decoder
        output_bias (torch.Tensor): The initial bias for the output layer. Initialized to mean of output data
    """
    def __init__(self, layer_dims, output_bias=None):
        super(MLPDecoder, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.decoder_MLP = nn.Sequential(*layers)
        if output_bias is not None:
            self.decoder_MLP[-1].bias = torch.nn.Parameter(output_bias)

    def forward(self, x):
        return self.decoder_MLP(x)

# class GATMapper(nn.Module):
#     """
#     Generates node states from latent vector using GAT
#     """
#     def __init__(self, latent_dim, node_pos, pos_emb_dim, n_heads):
#         super(GATMapper, self).__init__()
#         self.node_pos = node_pos
#         self.gat = nn.GATConv(latent_dim, node_pos, heads=n_heads)
#         self.positional_embedding_mlp = nn.Linear(3, pos_emb_dim)

#     def forward(self, x, )

class GEN(nn.Module):
    def __init__(self):
        super(GEN, self).__init__()
    
