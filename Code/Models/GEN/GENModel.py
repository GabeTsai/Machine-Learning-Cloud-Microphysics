import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from .Icosphere import IcosphereMesh, IcosphereTetrahedron

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPEncoder(nn.Module):
    """
    MLP to transform input data to latent vector

    Args:
        layer_dims (list): List of dimensions for each layer in the encoder
    """
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.LayerNorm(layer_dims[i + 1]))
                layers.append(nn.SiLU())
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
    def __init__(self, latent_dim, pos_emb_dim, pos_dim = 3):
        super().__init__()
        self.pos_emb_dim = pos_emb_dim
        self.positional_embedding_mlp = nn.Sequential(
            nn.Linear(pos_dim, pos_emb_dim),
            nn.LayerNorm(pos_emb_dim),
            nn.SiLU()
        )
        self.node_state_mlp = nn.Sequential(
            nn.Linear(latent_dim + pos_emb_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, node_pos):
        pos_emb = self.positional_embedding_mlp(node_pos) # (num_nodes, pos_emb_dim)
        x = x.unsqueeze(0).expand(node_pos.size(0), x.size(0)) # (N, latent_dim)
        combined = torch.cat((x, pos_emb), dim=1) # (N, latent_dim + pos_emb_dim)
        node_states = self.node_state_mlp(combined)
        node_states = node_states + x
        return node_states

class NodeUpdateMLP(nn.Module):
    
class Processor(nn.Module):

    def __init__(self, hidden_dim, num_rounds = 16):
        super(Processor, self).__init__()
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_rounds)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_rounds)])
        self.num_rounds = num_rounds
    
    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weight
        
        for i in range(self.num_rounds):
            residual = x
            x = self.layer_norms[i](self.convs[i](x, edge_index, edge_weight = edge_weights))
            x = x + residual #add residual connection

        return x
    
class GlobalPooling(nn.Module):
    """
    Aggregates node states to a single vector using a global pooling mechanism.

    Args:
        method (str): The pooling method to use. Options are 'mean' and 'max'

    Returns:
        torch.Tensor: The aggregated latent vector
    """
    def __init__(self, method='mean'):
        super(GlobalPooling, self).__init__()
        self.method = method

    def forward(self, x, batch):
        if self.method == 'mean':
            return pyg_nn.global_mean_pool(x, batch)
        elif self.method == 'max':
            return pyg_nn.global_max_pool(x, batch)
        else:
            raise ValueError(f"Pooling method {self.method} not supported.")
        
class MLPDecoder(nn.Module):
    """
    MLP to transform latent vector obtained from mesh processor to output

    Args:
        layer_dims (list): List of dimensions for each layer in the decoder
        output_bias (torch.Tensor): The initial bias for the output layer. Initialized to mean of output data
    """
    def __init__(self, layer_dims, output_bias=None):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.SiLU())
        self.decoder_MLP = nn.Sequential(*layers)
        if output_bias is not None:
            self.decoder_MLP[-1].bias = torch.nn.Parameter(output_bias)

    def forward(self, x):
        return self.decoder_MLP(x)

class GEN(nn.Module):
    """
    GEN model

    Args:
        encoder (nn.Module): Encoder module
        node_mapper (nn.Module): Node state mapper module
        processor (nn.Module): Processor module
        pooling_layer (nn.Module): Global pooling layer
        decoder (nn.Module): Decoder module
        num_rounds (int): Number of message passing rounds

    Returns:
        torch.Tensor: The output of the GEN model
    """
    def __init__(self, encoder, node_mapper, processor, pooling_layer, decoder, num_refine = 3):
        super(GEN, self).__init__()
        self.encoder = encoder
        self.node_mapper = node_mapper
        self.processor = processor
        self.pooling_layer = pooling_layer
        self.decoder = decoder
        self.mesh = IcosphereTetrahedron(num_refine)
        self.node_pos = torch.FloatTensor(self.mesh.vertices).to(device)
        self.edge_index = torch.LongTensor(self.mesh.edges).to(device)
        self.edge_weights = nn.Parameter(torch.FloatTensor(self.mesh.edge_weights)).to(device)

    def forward(self, x):
        #Encode input data to latent vector
        latent_vectors = self.encoder(x)
        
        #Map latent vector and node positions to node states for each batch element
        node_states = []
        batch_indices = []
        #Store node states for each latent vector and corresponding batch indices
        for i, latent_vector in enumerate(latent_vectors):
            node_states.append(self.node_mapper(latent_vector, self.node_pos))
            batch_indices.append(torch.full((self.node_pos.size(0),), i, dtype = torch.long, device = device))
        
        node_states = torch.cat(node_states, dim=0)  # (B * N, latent_dim)
        batch_indices = torch.cat(batch_indices, dim=0)  # (B * N,)

        # Prepare data object for GCN
        data = Data(x=node_states, edge_index=self.edge_index, edge_weight = self.edge_weights, batch = batch_indices)

        #Process node states with message passing
        processed_node_states = self.processor(data)

        #Aggregate node states into a single latent vector
        global_latent_vector = self.pooling_layer(processed_node_states, data.batch)

        output = self.decoder(global_latent_vector)
        return output

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

