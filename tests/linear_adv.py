import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter
from utils import fvm_solver, utils

device = 'cpu'

# Model
model = FluxLimiter(1,1,64,5,act="Tanh") #
model.load_state_dict(torch.load("model.pt"))
model = model.to(device)

def neural_flux_limiter(r):
    model.eval()
    with torch.no_grad():
        phi = model(torch.Tensor(r).view(-1, 1).to(device))
    return phi.numpy().squeeze()

x0, u0 = utils.wave_combination()

flux_limiters = {
    "Upwind": utils.FOU,
    "Lax-Wendroff": utils.LaxWendroff,
    "Minmod": utils.minmod,
    "Superbee": utils.superbee,
    "Van Leer": utils.vanLeer,
    "Koren": utils.koren,
    "Neural network": neural_flux_limiter,
}

fig, ax = plt.subplots()
ax.plot(x0, u0, label='Initial data', clip_on=False)
for (name, flux_limiter) in flux_limiters.items():
    u, _ = fvm_solver.solve_linear_advection_1D(u0=u0, T=8, a=1., dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
    ax.plot(x0, u, label=name, clip_on=False)
    print(f"{name}: {np.sum((u - u0)**2)/u.size}")
ax.legend()
fig.set_size_inches(10,5)
fig.savefig('wave_comb_test')

