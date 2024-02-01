import torch
import torch.nn as nn
import torch.nn.functional as F

class FluxLimiter(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        activation = nn.Tanh
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
        
        # Exert hard contraints
        phi = (r - 1) * phi + 1
        phi = F.relu(phi)
        phi = torch.where(r <= 0, torch.zeros_like(phi), phi)

        return phi