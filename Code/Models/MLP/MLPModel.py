import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_bias, hl1, hl2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.fc3 = nn.Linear(hl2, 1)
        if output_bias is not None:
            self.fc3.bias = torch.nn.Parameter(output_bias)
        else:
            nn.init.zeros_(self.fc3.bias)  # Ensure the bias is initialized 
            
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
