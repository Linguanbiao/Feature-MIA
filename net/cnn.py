import torch
import torch.nn as nn


class cnnNet(nn.Module):
    def __init__(self):
        super(cnnNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=16*7*7, out_features=128),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
