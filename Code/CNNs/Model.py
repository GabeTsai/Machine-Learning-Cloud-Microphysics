import torch
import torch.nn as nn
import numpy as np
import json

class SimpleCNN(nn.Module):
    def __init__(self, height_dim, output_bias):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size = 3, padding = 1) #(B, 4, 24)
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1) #(B, 8, 24)
        self.conv3 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1) #(B, 16, 24)
        self.fc1 = nn.Linear(32 * height_dim, 64) # (B, 32, 24) -> (B, 64)
        self.fc2 = nn.Linear(64, height_dim) # (B, 64) -> (B, 24)
        self.fc2.bias = torch.nn.Parameter(output_bias)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.relu(x)
        return x

class LargerCNN(nn.Module):
    def __init__(self, height_dim, output_bias):
        super(LargerCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1) #(B, 8, height_dim)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1) #(B, 16, height_dim)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1) #(B, 32, height_dim)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1) #(B, 64, height_dim)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1) #(B, 128, height_dim)
        self.fc1 = nn.Linear(128 * height_dim, 128) # (B, 128*height_dim) -> (B, 128)
        self.fc2 = nn.Linear(128, height_dim) # (B, 128) -> (B, height_dim)
        self.fc2.bias = torch.nn.Parameter(output_bias)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(x)
        return x
