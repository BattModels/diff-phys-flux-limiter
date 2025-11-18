import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class FluxLimiter(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers, act):
        super().__init__()

        # Set activation function
        if act == "relu":
            activation = nn.ReLU
        elif act == "tanh":
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
        # phi = phi * torch.maximum(torch.tensor([0.]), torch.minimum(torch.tensor([2.]), 2*r))

        return phi
    
class SymmetricFluxLimiter(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers, act):
        super().__init__()

        # Set activation function
        if act == "relu":
            activation = nn.ReLU
        elif act == "tanh":
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
        sharpness_ctrl = 1000
        is_r_gt_one = F.sigmoid(sharpness_ctrl * (r - 1.0))
        is_r_lt_one = 1. - is_r_gt_one
        one_over_r_gt_one = is_r_gt_one * (1.0 / (r + 1e-8))

        r_mod = is_r_lt_one * r + one_over_r_gt_one 

        phi = self.start(r_mod)
        phi = self.hidden(phi)
        phi = self.end(phi)

        phi = F.sigmoid(phi)
        phi = (1 - phi) * utils.minmod_torch(r_mod) + phi * utils.superbee_torch(r_mod)

        phi = is_r_lt_one * phi + is_r_gt_one * r * phi
        return phi