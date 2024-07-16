import torch
import torch.nn as nn
import numpy as np
import json

class SimpleCNN(nn.Module):
    def __init__(self, height_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(32 * height_dim, 64)
        self.fc2 = nn.Linear(64, height_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.relu(x)
        return x

