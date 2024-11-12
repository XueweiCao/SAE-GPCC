import torch
import torch.nn as nn


class StackedAutoEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super(StackedAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256, device=device), nn.ReLU(),
            nn.Linear(256, output_dim, device=device)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 256, device=device), nn.ReLU(),
            nn.Linear(256, input_dim, device=device), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class SimpleAutoEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 50, device=device), nn.ReLU(),
            nn.Linear(50, output_dim, device=device)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 50, device=device), nn.ReLU(),
            nn.Linear(50, input_dim, device=device), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def get_stacked_model(device, input_dim=600, output_dim=100):
    model = StackedAutoEncoder(device, input_dim, output_dim)
    return model


def get_simple_model(device, input_dim=100, output_dim=20):
    model = SimpleAutoEncoder(device, input_dim, output_dim)
    return model
