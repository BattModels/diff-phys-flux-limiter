import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter
from utils import fvm_solver, utils

device = 'cpu'

# Model
model = FluxLimiter(1,1,64,5,act="relu")
model.load_state_dict(torch.load("model_linear_relu.pt", map_location=torch.device(device)))
model = model.to(device)

def neural_flux_limiter(r):
    model.eval()
    with torch.no_grad():
        phi = model(torch.Tensor(r).view(-1, 1).to(device))
    return phi.numpy().squeeze()

flux_limiters = {
    "Upwind": utils.FOU,
    "Lax-Wendroff": utils.LaxWendroff,
    "Minmod": utils.minmod,
    "Superbee": utils.superbee,
    "Van Leer": utils.vanLeer,
    "Koren": utils.koren,
    "MC": utils.MC,
    "DPFL (linear)": neural_flux_limiter,
    # "Piecewise linear": utils.piecewise_linear_limiter,
}

# Calculate errors for different number of cells
T = 0.4
n_datapoints = 4
n_cells_coarsest = 40
cell_counts = n_cells_coarsest * 2**np.arange(n_datapoints)

errors = np.zeros((len(flux_limiters), n_datapoints))
sol_list = []
for n, n_cells in enumerate(cell_counts):
    x0, u0 = utils.sin_wave(N=int(n_cells))
    # x0, u0 = utils.step(N=int(n_cells))

    # Solve for exact solution at time T
    T_all, X, U = fvm_solver.solve_burgers_1D_exactly(Nx=int(n_cells), T_end=T, dt=T)

    for i, (name, flux_limiter) in enumerate(flux_limiters.items()):
        u = fvm_solver.solve_burgers_1D(u0=u0, T=T, dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
        # errors[i, n] = np.sqrt(np.mean((u - U[-1])**2))
        errors[i, n] = np.mean(np.abs(u - U[-1]))
        # fig, ax = plt.subplots()
        # ax.plot(x0, U[-1], label='Exact', clip_on=False, color='gray')
        # ax.plot(x0, u, label=f't = {T} ({name})', clip_on=False)
        # ax.legend()
        # fig.savefig(f'convergence_{name}_N{int(n_cells)}.png', dpi=300)
        # plt.close(fig)

# Plot errors with number of cells
fig, ax = plt.subplots()
for i, (name, flux_limiter) in enumerate(flux_limiters.items()):
    ax.loglog(cell_counts, errors[i], label=name, clip_on=False)
    rate = np.log2(errors[i, -2] / errors[i, -1])
    print(f"{name}: {rate}")
ax.legend()
ax.set_xlabel('Number of cells')
ax.set_ylabel('$L_2$ error')
fig.savefig('convergence_rate', dpi=300)

# Print out table of errors and convergence rates
# compute convergence order
def conv_order(err):
    order = np.full_like(err, np.nan)
    for k in range(1, len(err)):
        order[k] = np.log2(err[k-1]/err[k])
    return order

orders = np.vstack([conv_order(errors[i]) for i in range(errors.shape[0])])

# ---- build table like journal format ----
rows = []
names = list(flux_limiters.keys())

for i, name in enumerate(names):
    for k, N in enumerate(cell_counts):
        rows.append({
            "Scheme": name,
            "Mesh": int(N),
            r"$L_2$": f"{errors[i,k]:.2e}",
            "Order": "-" if np.isnan(orders[i,k]) else f"{orders[i,k]:.2f}"
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))


