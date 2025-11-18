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
    "font.size":10,
    })

device = 'cpu'

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

# Convert preds → numpy
phi_neural  = preds.cpu().numpy().squeeze()
phi_minmod  = utils.minmod(r_eval)
phi_superbee= utils.superbee(r_eval)
phi_vanleer = utils.vanLeer(r_eval)

fig, ax = plt.subplots(figsize=(4, 3))

mask = r_eval >= 0
ax.fill_between(
    r_eval[mask],
    phi_minmod[mask],
    phi_superbee[mask],
    color='gray',
    alpha=0.3,
    interpolate=True,
)

# plot the four curves
ax.plot(r_eval, phi_minmod,   color='tab:blue',   lw=1.5)
ax.plot(r_eval, phi_superbee, color='tab:green',  lw=1.5)
ax.plot(r_eval, phi_vanleer,  color='tab:orange', lw=1.5)
ax.plot(r_eval, phi_neural,   color='tab:red',    lw=1.5)

# horizontal guide‐lines at φ=1 and φ=2
ax.hlines(1, 0, 0.5, ls=(0, (5, 5)), color='black', lw=0.8)
ax.hlines(2, 0, 2,  ls=(0, (5, 5)), color='black', lw=0.8)
ax.vlines(
    1,        # x = 1
    0,        # y start = 0
    1,        # y end   = 1
    colors='black',
    ls=(0, (5, 5)),
    lw=0.8,
)

# move axes to cross at (0,0) & remove top/right spines
ax.tick_params(
    which='both',    # apply to both major and minor ticks
    top=False,       # turn off top ticks
    right=False      # turn off right ticks (if any remain)
)
# disable minor ticks entirely
ax.minorticks_off()


# Move spines to zero and hide top/right
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# set limits & ticks
ax.set_xlim(0, 4.5)
ax.set_ylim(0, 2.5)
ax.set_xticks([0,1,2,3,4])
ax.set_yticks([0,1,2])

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
hw = 0.02*(ymax-ymin)      # head width
hl = 0.02*(xmax-xmin)      # head length

# x‑axis arrow
ax.arrow(xmin, 0, xmax-xmin, 0,
         length_includes_head=True,
         head_width=hw, head_length=hl,
         fc='k', ec='k', clip_on=False)

# y‑axis arrow
ax.arrow(0, ymin, 0, ymax-ymin,
         length_includes_head=True,
         head_width=hw, head_length=hl,
         fc='k', ec='k', clip_on=False)

# Labels & annotations
ax.set_xlim(0,4)
ax.set_ylim(0,2.2)
ax.set_xticks([0,1,2,3,4])
ax.set_yticks([0,1,2])

# inline text labels
# ax.text(3.8, 2.02, r'$\phi=2$',  va='bottom')
# ax.text(3.8, 1.02, r'$\phi=1$',  va='bottom')
# ax.text(1.5, 1.5,  r'$\phi=r$',  ha='center', va='center', fontsize=8)
ax.text(3.0, 0.9, 'Minmod',   color='tab:blue',   ha='center', va='top')
ax.text(0.8, 1.20, 'Superbee', color='tab:green',  ha='center', va='bottom')
ax.text(4.35, 1.6, 'van Leer', color='tab:orange', ha='center', va='center')
# ax.text(3.0, 2.15, r'\textbf{DPFL}',   color='tab:red',    ha='center', va='center')
ax.text(3.0, 2.15, 'DPFL',   color='tab:red',    ha='center', va='center')

ax.text(2.5, 1.7, r'$2^{\rm{nd}}$ order TVD',   color='tab:gray',    ha='center', va='center')

ax.set_xlabel('$r$')
ax.set_ylabel(r'$\phi_{\theta}(r)$')
ax.set_aspect(1)
fig.savefig('figures/paper/limiter_remake.pdf')