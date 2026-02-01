import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

def initial_condition(N=200, L=1):
    z = 0.3
    delta = 0.005
    beta = np.log(2)/(36*delta**2)
    
    def G(x, beta, z):
        return np.exp(-beta*(x-z)**2)
    
    dx = L / N
    x0 = (np.arange(N)+0.5)*dx - dx/2
    u0 = 1/6 * (G(x0,beta,z-delta) + G(x0,beta,z+delta) + 4*G(x0,beta,z)) * np.logical_and(x0 >= 0.2, x0 <= 0.4) \
         + 1 * np.logical_and(x0 >= 0.6, x0 <= 0.8)
    
    return x0, u0

# Model
model = FluxLimiter(1,1,64,5,act="relu")
model.load_state_dict(torch.load("model_linear_relu.pt", map_location=torch.device(device)))
model = model.to(device)

def neural_flux_limiter(r):
    model.eval()
    with torch.no_grad():
        phi = model(torch.Tensor(r).view(-1, 1).to(device))
    return phi.numpy().squeeze()

x0, u0 = initial_condition()

T = 2.0
a = 1.0

_, u_upwind_hist = fvm_solver.solve_linear_advection_1D(u0=u0, T=T, a=a, dx=x0[1]-x0[0], CFL=0.4, flux_limiter=utils.FOU)
_, u_lw_hist = fvm_solver.solve_linear_advection_1D(u0=u0, T=T, a=a, dx=x0[1]-x0[0], CFL=0.4, flux_limiter=utils.LaxWendroff)
n_frames = u_upwind_hist.shape[0]

# --- Set up animation figure for upwind---
fig, ax = plt.subplots()
line_upwind, = ax.plot(x0, u_upwind_hist[0], label="Upwind", clip_on=False)
line_init,   = ax.plot(x0, u0, "--", color="gray", label="Initial", clip_on=False)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.5, 1.5)
ax.minorticks_off()

def update_upwind(frame):
    line_upwind.set_ydata(u_upwind_hist[frame])
    return line_upwind,

anim = FuncAnimation(fig, update_upwind, frames=n_frames, interval=40, blit=True)
anim.save('figures/fou_animation.mp4', writer='ffmpeg', dpi=300)

# --- Set up animation figure for Lax-Wendroff---
fig, ax = plt.subplots()
line_lw,     = ax.plot(x0, u_lw_hist[0],     label="Lax-Wendroff", clip_on=False)
line_init,   = ax.plot(x0, u0, "--", color="gray", label="Initial", clip_on=False)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.5, 1.5)
ax.minorticks_off()

def update_lw(frame):
    line_lw.set_ydata(u_lw_hist[frame])
    return line_lw,

anim = FuncAnimation(fig, update_lw, frames=n_frames, interval=40, blit=True)
anim.save('figures/lw_animation.mp4', writer='ffmpeg', dpi=300)