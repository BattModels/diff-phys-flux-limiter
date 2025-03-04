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


# This is the script for plots:
# "learned_limiter_linear_relu.pdf" (Fig. 4(a))
# "check_fl_symm_linear_relu.pdf" (Fig. 4(b))

device = 'cpu'

# Model
model = FluxLimiter(1,1,64,5,act="relu") #sometimes 32 n_hidden
# model.load_state_dict(torch.load("model_euler.pt"))
model.load_state_dict(torch.load("model_linear_relu.pt", map_location=torch.device(device)))
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
    # test_points = model(torch.tensor([[0,0], [0.5],[1.5],[3.0]]))
    # print(test_points)

# Fig. 1(a)
fig, ax = plt.subplots()
ax.plot(r_eval, utils.minmod(r_eval), label="Minmod", clip_on=False)
ax.plot(r_eval, utils.superbee(r_eval), label="Superbee", clip_on=False)
ax.plot(r_eval, utils.vanLeer(r_eval), label="van Leer", clip_on=False)
# ax.plot(r_eval, utils.koren(r_eval), label="Koren", clip_on=False)
ax.plot(r_eval, preds.cpu(), label="Neural flux limiter", clip_on=False)
ax.set_xlabel('$r$')
ax.set_ylabel('$\phi(r)$')
ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.legend()
fig.show()
fig.savefig('figures/paper/learned_limiter_linear_relu.pdf')
color = plt.gca().lines[2].get_color()

# Fig. 1(b)
fig, ax = plt.subplots()
ax.plot(r_eval, preds.cpu(), label="Neural flux limiter", clip_on=False)
with torch.no_grad():
    preds = model(torch.Tensor(1./r_eval).view(-1,1))
    preds = preds.squeeze()
ax.plot(r_eval, r_eval * preds.cpu().numpy(), '--', label="$r \phi(1/r)$", clip_on=False, color=color)
ax.set_xlabel('$r$')
ax.set_ylabel('$\phi(r)$')
ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.legend()
fig.show()
fig.savefig('figures/paper/check_fl_symm_linear_relu.pdf')