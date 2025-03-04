import h5py
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import scienceplots

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter, SymmetricFluxLimiter
from utils import fvm_solver, utils

plt.style.use(['science', 'high-vis'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":12,
    })

# define global size of the figure
plt.rcParams['figure.figsize'] = [10, 5]

# Define your nine high-contrast colors
# get the color from the current style
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0], 'tab:gray', 'tab:brown', colors[3], colors[4], colors[5], colors[2], colors[1], 'tab:pink']
# colors = ['tab:gray', 'tab:blue', 'tab:brown', 'tab:orange', 
#           'tab:red', 'tab:purple', 'tab:green', 'tab:pink', 'tab:olive']

# Define nine different linestyles
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', (0, (3, 1, 1, 1, 1, 1)), '-']

# Set the color cycle globally
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) + mpl.cycler(linestyle=linestyles)

torch.manual_seed(3407)

device = 'cpu'

# Model
model = FluxLimiter(1,1,64,5,act="relu")
# model = SymmetricFluxLimiter(1,1,64,5,act="tanh") #
model.load_state_dict(torch.load("model_linear_relu.pt", map_location=torch.device(device)))
model = model.to(device)

def neural_flux_limiter(r):
    model.eval()
    with torch.no_grad():
        phi = model(torch.Tensor(r).view(-1, 1).to(device))
    return phi.numpy().squeeze()

flux_limiters = {
    "Upwind": utils.FOU,
    "LW": utils.LaxWendroff,
    "Minmod": utils.minmod,
    "Superbee": utils.superbee,
    "van Leer": utils.vanLeer,
    "Koren": utils.koren,
    "MC": utils.MC,
    "NN": neural_flux_limiter,
    # "Piecewise linear": utils.piecewise_linear_limiter,
}

path = '/home/chyhuang/research/flux_limiter/data/1D_Burgers_Sols_Nu0.001.hdf5'
with h5py.File(path, "r") as h5_file:
        # xcrd = np.array(h5_file["x-coordinate"], dtype=np.float32)
        # tcrd = np.array(h5_file["t-coordinate"], dtype=np.float32)
        data = np.array(h5_file["tensor"], dtype=np.float32)

CG = 8

# Shuffle the whole dataset
perm = torch.randperm(data.shape[0]).numpy()

# Fig. 5(a)

x0, u0 = utils.sin_wave(N=100)

fig, ax = plt.subplots()
axins = zoomed_inset_axes(ax, 6, loc=3)
ax.plot(x0, u0, label='Initial data', clip_on=False)
axins.plot(x0, u0)
for (name, flux_limiter) in flux_limiters.items():
    u, _ = fvm_solver.solve_linear_advection_1D(u0=u0, T=1, a=1., dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
    main_line, = ax.plot(x0, u, label=name, clip_on=False)
    axins.plot(x0, u)
    print(f"{name}: {np.sum(np.abs(u - u0)**2)/u.size}")
# ax.legend(loc=[0.68, 0.22])
ax.legend()
ax.set_yticks(np.arange(-1, 1.5, 0.5))
x1, x2, y1, y2 = 0.2, 0.3, 0.93, 1.02
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5", zorder=100)
# fig.set_size_inches(10,5)
fig.savefig('figures/paper/sample1_zoom_in.pdf')

# Fig. 5(b)

i_sample = 3545#8192+1024+12 #682  #9994  # bofore shuffling: sample2: 8, sample 3: 9999, after 3545, 6181

u0 = data[perm[i_sample], 0, ::CG]
x0 = np.linspace(1/128/2, 1- 1/128/2, 128)

fig, ax = plt.subplots()
# axins = zoomed_inset_axes(ax, 6, loc=3)
ax.plot(x0, u0, label='Initial data', clip_on=False)
for (name, flux_limiter) in flux_limiters.items():
    u, _ = fvm_solver.solve_linear_advection_1D(u0=u0, T=1, a=1., dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
    main_line, = ax.plot(x0, u, label=name, clip_on=False)
    # axins.plot(x0, u)
    print(f"{name}: {np.sum(np.abs(u - u0)**2)/u.size}")
ax.legend(loc=[0.28, 0.40])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
# fig.set_size_inches(10,5)
fig.savefig('figures/paper/sample2.pdf')

# Fig. 5(c)

i_sample = 6181#8192+1024+12 #682  #9994  # bofore shuffling: sample2: 8, sample 3: 9999, after 3545, 6181

u0 = data[perm[i_sample], 0, ::CG]
x0 = np.linspace(1/128/2, 1- 1/128/2, 128)

fig, ax = plt.subplots()
ax.plot(x0, u0, label='Initial data', clip_on=False)
for (name, flux_limiter) in flux_limiters.items():
    u, _ = fvm_solver.solve_linear_advection_1D(u0=u0, T=1, a=1., dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
    main_line, = ax.plot(x0, u, label=name, clip_on=False)
    # axins.plot(x0, u)
    print(f"{name}: {np.sum(np.abs(u - u0)**2)/u.size}")
ax.legend()
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
# fig.set_size_inches(10,5)
fig.savefig('figures/paper/sample3.pdf')


# Fig. 6(a)

x0, u0 = utils.square_wave(N=100)

fig, ax = plt.subplots()
ax.plot(x0, u0, label='Initial data', clip_on=False)
for (name, flux_limiter) in flux_limiters.items():
    u, _ = fvm_solver.solve_linear_advection_1D(u0=u0, T=1, a=1., dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
    main_line, = ax.plot(x0, u, label=name, clip_on=False)
    # axins.plot(x0, u)
    print(f"{name}: {np.sum(np.abs(u - u0)**2)/u.size}")
ax.legend()
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
# fig.set_size_inches(10,5)
fig.savefig('figures/paper/square_wave.pdf')

# Fig. 6(b)

x0, u0 = utils.wave_combination()

fig, ax = plt.subplots()
ax.plot(x0, u0, label='Initial data', clip_on=False)
for (name, flux_limiter) in flux_limiters.items():
    u, _ = fvm_solver.solve_linear_advection_1D(u0=u0, T=1, a=8., dx=x0[1]-x0[0], CFL=0.4, flux_limiter=flux_limiter)
    main_line, = ax.plot(x0, u, label=name, clip_on=False)
    # axins.plot(x0, u)
    print(f"{name}: {np.sum(np.abs(u - u0)**2)/u.size}")
ax.legend()
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
# fig.set_size_inches(10,5)
fig.savefig('figures/paper/wave_comb_test.pdf')