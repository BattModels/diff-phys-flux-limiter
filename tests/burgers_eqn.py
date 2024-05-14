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

N = 101
T = 0.3                        # breaking time: 1/(2*np.pi)
x0, u0 = utils.sin_wave(N=N)
# x0, u0 = utils.square_wave(N=100)
# x0, u0 = utils.linear_spike(N=100)
# x0, u0 = utils.step(N=100)
# x0, u0 = utils.ramp(slope_ratio=5, N=100)


T_all, X, U = fvm_solver.solve_burgers_1D_exactly(Nx=N, T_end=T, dt=1e-3)

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
ax.plot(X, U[-1], label='Exact', clip_on=False)
for (name, flux_limiter) in flux_limiters.items():
    u = fvm_solver.solve_burgers_1D(u0=u0, T=T, dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
    ax.plot(x0, u, label=name, clip_on=False)
    print(f"{name}: {np.sum((u - U[-1])**2)/u.size}")
ax.legend()
# ax.set_xlim([0.475, 0.525])
fig.set_size_inches(10,5)
fig.savefig('burgers')

