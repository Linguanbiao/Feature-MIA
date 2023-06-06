import torch.nn as nn
import torch


class LeaksCNN(nn.Module):
    def __init__(self):
        super(LeaksCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=2, stride=1),
            nn.ReLU()
        )
        self.maxp1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0, stride=1),
            nn.ReLU()
        )
        self.maxp2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=32*3*3, out_features=128),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10)

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
