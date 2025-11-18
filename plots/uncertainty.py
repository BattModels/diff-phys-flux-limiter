import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter
from utils import fvm_solver, utils

plt.style.use('science')
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":8,
    })

device = 'cpu'


# This is the script for plots:
# "uncertainty.pdf" (Fig. 13 in the paper)

# Model
fig, ax = plt.subplots()
for (model_name, label) in zip(['model_linear_seed0.pt', 'model_linear_seed1.pt', 'model_linear_seed2.pt', 'model_linear_seed3.pt', 'model_linear_relu.pt'], [r'\#1', r'\#2', r'\#3', r'\#4', r'\#5']):
    model = FluxLimiter(1,1,64,5,act="relu") #
    model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    model = model.to(device)

    def neural_flux_limiter(r):
        model.eval()
        with torch.no_grad():
            phi = model(torch.Tensor(r).view(-1, 1).to(device))
        return phi.numpy().squeeze()

    model.eval()
    r_min = -3
    r_max = 10
    n_eval = 1000
    r_eval = np.linspace(r_min, r_max, n_eval)
    with torch.no_grad():
        preds = model(torch.Tensor(r_eval).view(-1,1))

    ax.plot(r_eval, preds.cpu(), label=label)
ax.set_xlabel('$r$')
ax.set_ylabel('$\phi(r)$')
ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.legend(loc='upper left')
fig.savefig(f'figures/paper/limiter_uncertainty.pdf')

