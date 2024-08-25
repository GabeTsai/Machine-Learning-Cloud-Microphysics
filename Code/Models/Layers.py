import torch
from torch import nn

class MLP(nn.Module):
    """
    General-purpose MLP class for encoding, decoding, or other transformations.

    Args:
        layer_dims (list): List of dimensions for each layer in the MLP.
        activation (nn.Module): Activation function to use between layers. Default is ReLU.
        use_layer_norm (bool): Whether to apply LayerNorm after each linear layer (except the last one).
        output_bias (torch.Tensor): The initial bias for the output layer, if needed. Used for decoders.
    """
    def __init__(self, layer_dims, activation=nn.ReLU(), use_layer_norm=False, use_batch_norm = True, output_bias=None):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(layer_dims[i + 1]))
                elif use_batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(activation)
        self.mlp = nn.Sequential(*layers)

        # Optionally set the output bias
        if output_bias is not None:
            self.mlp[-1].bias = torch.nn.Parameter(output_bias)

    def forward(self, x):
        return self.mlp(x)
