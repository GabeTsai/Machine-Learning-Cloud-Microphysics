import torch
import sys
sys.path.append('../')
from Models.Layers import *

class Processor(nn.Module):
    def __init__(self, latent_dim, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLP([latent_dim, latent_dim, latent_dim, latent_dim], activation=nn.SiLU(), use_layer_norm=False, use_batch_norm=True)
            for _ in range(num_blocks)
        ])
        self.act_f = nn.SiLU()
    def forward(self, x):
        for block in self.blocks:
            residual = x
            x = block(x)
            x += residual #residual connection
        return x

class DeepMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, output_bias, num_blocks):
        super().__init__()
        torch.manual_seed(3407) #is all you need
        self.encoder = MLP([input_dim, latent_dim])
        self.processor = Processor(latent_dim, num_blocks)
        self.decoder = MLP([latent_dim, output_dim], output_bias = output_bias)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.processor(x)
        x = self.decoder(x)
        return x