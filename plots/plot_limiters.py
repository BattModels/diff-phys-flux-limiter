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
# "fl_comparison_relu.pdf" (Fig. 7 in the paper)
# "learned_limiter_euler.pdf" (Fig. 8 in the paper)


# Fig. 7

# Model
model = FluxLimiter(1,1,64,5,act="relu") #
model.load_state_dict(torch.load("model_linear_relu.pt", map_location=torch.device('cpu')))
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
fig, ax = plt.subplots()
ax.plot(r_eval, utils.minmod(r_eval), label="Minmod")
ax.plot(r_eval, utils.superbee(r_eval), label="Superbee")
ax.plot(r_eval, utils.vanLeer(r_eval), label="van Leer")
ax.plot(r_eval, preds.cpu(), label="DPFL")
ax.plot(r_eval, utils.piecewise_linear_limiter(r_eval), label="Piecewise \n linear")
ax.set_xlabel('$r$')
ax.set_ylabel('$\phi(r)$')
ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.legend(loc='upper left')
fig.savefig('figures/paper/fl_comparison_relu.pdf')



# Fig. 8

model = FluxLimiter(1,1,64,5,act="tanh") #
model.load_state_dict(torch.load("model_euler_best.pt"))
model = model.to(device)

def neural_flux_limiter(r):
    model.eval()
    with torch.no_grad():
        phi = model(torch.Tensor(r).view(-1, 1).to(device))
    return phi.numpy().squeeze()

model.eval()
r_min = -2
r_max = 10
n_eval = 1000
r_eval = np.linspace(r_min, r_max, n_eval)
with torch.no_grad():
    preds = model(torch.Tensor(r_eval).view(-1,1))
fig, ax = plt.subplots()
ax.plot(r_eval, utils.minmod(r_eval), label="Minmod")
ax.plot(r_eval, utils.superbee(r_eval), label="Superbee")
ax.plot(r_eval, utils.vanLeer(r_eval), label="van Leer")
ax.plot(r_eval, preds.cpu(), label="DPFL-Sod")
ax.set_xlabel('$r$')
ax.set_ylabel('$\phi(r)$')
ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.legend()
fig.savefig('figures/paper/learned_limiter_euler.pdf')