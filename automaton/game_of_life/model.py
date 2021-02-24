import torch
from torch import nn

class NeuralAutomaton(nn.Module):
    def __init__(self):
        super(NeuralAutomaton, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, state):
        if state.ndim == 2:
            state = state.unsqueeze(0)
        return self.layers(state.unsqueeze(1))[:,0]