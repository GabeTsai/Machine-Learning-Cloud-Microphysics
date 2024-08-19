import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from GENModel import *

# def testMLPNodeStateMapper():
#     '''
#     Test MLPNodeStateMapper forward pass
#     '''
#     latent_dim = 64
#     pos_emb_dim = 32
#     node_state_mapper = MLPNodeStateMapper(latent_dim, pos_emb_dim)
#     x = torch.randn(5, latent_dim) #5 random latent vectors
#     node_pos = torch.randn(3, 3) #3 random points in 3D space
#     out = node_state_mapper(x, node_pos)
#     assert out.shape == (5, 3, latent_dim), f"Expected output shape (5, 3, {latent_dim}), got {out.shape}"
#     print('MLPNodeStateMapper test passed')

def testGENModel():
    '''
    Test GENModel forward pass
    '''
    model = GEN(3, 64)
    data = torch.randn(5, 3)
    out = model(data)
    print(out) #Since output bias is 1, output should be close to 1
    assert out.shape == (5, 1), f"Expected output shape (5, 1), got {out.shape}"

def main():
    # testMLPNodeStateMapper()
    testGENModel()

if __name__ == "__main__":
    main()