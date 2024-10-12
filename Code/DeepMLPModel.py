import torch
import sys
from Layers import *

class Processor(nn.Module):
    def __init__(self, latent_dim, num_blocks, activation):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLP([latent_dim, latent_dim, latent_dim, latent_dim], activation = activation, use_layer_norm=False, use_batch_norm=True)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            residual = x
            x = block(x)
            x += residual #residual connection
        return x

class DeepMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, output_bias, num_blocks, activation = nn.ReLU()):
        super().__init__()
        torch.manual_seed(3407) #is all you need
        self.encoder = MLP([input_dim, latent_dim])
        self.processor = Processor(latent_dim, num_blocks, activation)

        if output_bias is not None and output_bias.shape != torch.Size([output_dim]):
            output_bias = output_bias.view(output_dim)
            
        self.decoder = MLP([latent_dim, output_dim], output_bias = output_bias)        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.processor(x)
        x = self.decoder(x)
        return x
    
class EnsembleDeepMLP(nn.Module):
    def __init__(self, models, latent_dim, output_dim, output_bias, num_blocks, freeze_ensemble_weights = True):
        super().__init__()
        torch.manual_seed(3407) #is all you need
        self.models = models
        self.meta_learner = DeepMLP(len(self.models), latent_dim, output_dim, output_bias, num_blocks)
        if (freeze_ensemble_weights):
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False

    def forward(self, x):
        meta_learner_inputs = []

        for i in range(len(self.models)):
            i_out = self.models[i](x)
            meta_learner_inputs.append(i_out)

        meta_learner_inputs = torch.cat(meta_learner_inputs, dim = 1)

        return self.meta_learner(meta_learner_inputs)
    
        