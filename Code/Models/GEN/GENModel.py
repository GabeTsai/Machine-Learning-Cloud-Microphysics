import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_scatter import scatter
from Icosphere import IcosphereMesh, IcosphereTetrahedron
import sys
sys.path.append('../')
from Layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.positional_embedding_mlp = MLP([pos_dim, pos_emb_dim, pos_emb_dim, pos_emb_dim], nn.SiLU())
        self.node_state_mlp = MLP([latent_dim + pos_emb_dim, latent_dim, latent_dim, latent_dim], nn.SiLU()) 

    def forward(self, x, node_pos):
        pos_emb = self.positional_embedding_mlp(node_pos) # (num_nodes, pos_emb_dim)
        x = x.unsqueeze(0).expand(node_pos.size(0), x.size(0)) # (N, latent_dim)
        combined = torch.cat((x, pos_emb), dim=1) # (N, latent_dim + pos_emb_dim)
        node_states = self.node_state_mlp(combined)
        node_states = node_states + x
        return node_states
    
class Processor(nn.Module):

    def __init__(self, node_update_mlp, edge_update_mlp, num_rounds = 16):
        super(Processor, self).__init__()
        self.node_update_mlps = nn.ModuleList([node_update_mlp for _ in range(num_rounds)])
        self.edge_update_mlps = nn.ModuleList([edge_update_mlp for _ in range(num_rounds)])
        self.num_rounds = num_rounds
    
    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_attr
        
        for i in range(self.num_rounds):
            #Update edge states
            residual_edge = edge_features
            edge_features = self.edge_update_mlps[i](torch.cat((x[edge_index[0]], x[edge_index[1]], edge_features), axis = 1)) # (num_edges, latent_dim)
            edge_features = edge_features + residual_edge #add residual connection

            #Aggregate edge features
            agg_edge_features = scatter(edge_features, edge_index[1], dim = 0, dim_size = x.size(0), reduce = 'sum') # (num_nodes, latent_dim)

            #Update node states
            residual_node = x
            x = self.node_update_mlps[i](torch.cat((x , agg_edge_features), axis = 1)) # (num_nodes, latent_dim)
            x = x + residual_node #add residual connection

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

class GEN(nn.Module):
    """
    GEN model

    Args:
        input_dim (int): Dimension of the input features
        latent_dim (int): Dimension of the latent space
        num_refine (int, optional): Number of refinement steps (default: 3)
        num_rounds (int, optional): Number of message passing rounds (default: 16)

    Returns:
        torch.Tensor: The output of the GEN model
    """
    def __init__(self, input_dim, latent_dim, num_refine = 3, num_rounds = 16):
        super(GEN, self).__init__()
        
        self.mesh = IcosphereTetrahedron(num_refine)
        self.node_pos = torch.FloatTensor(self.mesh.vertices).to(device)
        self.edge_index = torch.LongTensor(self.mesh.edges).to(device)
        self.edge_features = torch.FloatTensor(self.mesh.edge_feat).to(device)
        self.encoder = MLP([input_dim, latent_dim, latent_dim, latent_dim], nn.SiLU())
        self.node_mapper = MLPNodeStateMapper(latent_dim, latent_dim)
        self.edge_mapper = MLP([4, latent_dim, latent_dim, latent_dim], nn.SiLU())
        self.processor = self.init_processor(latent_dim, num_rounds) 
        self.pooling_layer = GlobalPooling()
        self.decoder = MLP([latent_dim, latent_dim, latent_dim, 1], activation = nn.SiLU())

    def init_processor(self, hidden_dim, num_rounds):
        """
        Initialize the processor module with the given hidden dimension and number of rounds

        Args:
            hidden_dim (int): Hidden dimension for the processor module
            num_rounds (int): Number of message passing rounds
        """
        node_update_mlp = MLP([2 * hidden_dim, hidden_dim, hidden_dim, hidden_dim], nn.SiLU())
        edge_update_mlp = MLP([3 * hidden_dim, hidden_dim, hidden_dim, hidden_dim], nn.SiLU())
        return Processor(node_update_mlp, edge_update_mlp, num_rounds)

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
        edge_features = self.edge_mapper(self.edge_features) # (num_edges, latent_dim)
        batch_indices = torch.cat(batch_indices, dim=0)  # (B * N,)
        
        # Prepare data object for GCN
        data = Data(x=node_states, edge_index=self.edge_index, edge_attr = edge_features, batch = batch_indices)

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

