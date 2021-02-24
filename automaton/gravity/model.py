import torch
from torch import nn

class NeuralAutomatonCollector(nn.Module):
    def __init__(self, n_channels=16, n_col_filters=16, n_col_layers=3, n_col_reapeat=16, n_update_filters=16, n_update_layers=4):
        super(NeuralAutomatonCollector, self).__init__()
        self.n_channels = n_channels
        self.n_col_reapeat = n_col_reapeat

        # input projection layer
        self.in_proj = nn.Conv2d(n_channels, n_col_filters, kernel_size=3, padding=1)

        # collector network
        layers = []
        for _ in range(n_col_layers):
            layers += [nn.Conv2d(in_channels=n_col_filters, out_channels=n_col_filters, kernel_size=3, padding=2, dilation=2), nn.ReLU()]
        self.collector = nn.Sequential(*layers)

        # update network
        layers = []
        for i in range(n_update_layers):
            n_in = n_channels + n_col_filters if i == 0 else n_update_filters
            n_out = n_update_filters if i < n_update_layers - 1 else n_channels
            layers += [nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=3, padding=1), nn.ReLU()]
        self.update = nn.Sequential(*layers[:-1])


    def forward(self, x):
        rep = self.in_proj(x)
        for _ in range(self.n_col_reapeat):
            rep = self.collector(rep)
        return x + self.update(torch.cat([x, rep], dim=1))