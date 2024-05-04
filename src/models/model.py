import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class FluxLimiter(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers, act):
        super().__init__()

        # Set activation function
        if act == "ReLU":
            activation = nn.ReLU
        elif act == "Tanh":
            activation = nn.Tanh
        else:
            raise ValueError(f"No activation function type: {act}")
        
        self.start = nn.Sequential(*[
                        nn.Linear(n_input, n_hidden),
                        activation()])
        self.hidden = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(n_hidden, n_hidden),
                            activation()]) for _ in range(n_layers-1)])
        self.end = nn.Linear(n_hidden, n_output)

    def forward(self, r):
        phi = self.start(r)
        phi = self.hidden(phi)
        phi = self.end(phi)

        phi = F.sigmoid(phi)
        phi = (1 - phi) * utils.minmod_torch(r) + phi * utils.superbee_torch(r)

        return phi