import h5py
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

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    # "Upwind": utils.FOU,
    # "Lax-Wendroff": utils.LaxWendroff,
    # "Minmod": utils.minmod,
    # "Superbee": utils.superbee,
    # "Van Leer": utils.vanLeer,
    # "Koren": utils.koren,
    "DPFL": neural_flux_limiter,
    # "Piecewise linear": utils.piecewise_linear_limiter,
}

path = '/home/chyhuang/research/flux_limiter/data/1D_Burgers_Sols_Nu0.001.hdf5'
with h5py.File(path, "r") as h5_file:
        # xcrd = np.array(h5_file["x-coordinate"], dtype=np.float32)
        # tcrd = np.array(h5_file["t-coordinate"], dtype=np.float32)
        data = np.array(h5_file["tensor"], dtype=np.float32)

i_sample = 999 #682  #9994 #9999
CG = 8
T = 0.2
u0 = data[i_sample, 0, ::CG]
u_exact = data[i_sample, int(T/0.01), ::CG]
x0 = np.linspace(1/128/2, 1- 1/128/2, 128)

fig, ax = plt.subplots()
ax.plot(x0, u0, label='Initial data', clip_on=False, color='gray')
ax.plot(x0, u_exact, label='t = 0.2 (Truth)', clip_on=False)
for (name, flux_limiter) in flux_limiters.items():
    u = fvm_solver.solve_burgers_1D(u0=u0, T=T, dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
    ax.plot(x0, u, label='t = 0.2 (' + name + ')', clip_on=False, linestyle='--')
    print(f"{name}: {np.sum((u - u_exact)**2)/u.size}")
ax.legend()
# fig.set_size_inches(10,5)
fig.savefig('figures/paper/burgers_sample.pdf')