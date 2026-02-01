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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

fig, ax = plt.subplots()
for act in ["relu", "tanh", "sigmoid"]:
    model = FluxLimiter(1,1,64,5,act=act)
    model.load_state_dict(torch.load(f"model_linear_{act}.pt", map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    r_min = -2
    r_max = 10
    n_eval = 1000
    r_eval = np.linspace(r_min, r_max, n_eval)
    with torch.no_grad():
        preds = model(torch.Tensor(r_eval).view(-1,1).to(device))
    ax.plot(r_eval, preds.cpu(), label="DPFL " + act, clip_on=False)
ax.set_xlabel('$r$')
ax.set_ylabel('$\phi(r)$')
ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.legend()
fig.show()
fig.savefig('figures/paper/limiters_act.pdf')