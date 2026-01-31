import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter
from utils import fvm_solver, utils
from data.dataset import load_linear_adv_1d

torch.manual_seed(3407)

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model
model = FluxLimiter(1,1,64,5,act="relu") #
model.load_state_dict(torch.load("model_linear_relu.pt"))
model = model.to(device)

def neural_flux_limiter(r):
    return model(r)

flux_limiters = {
    "Upwind": utils.FOU_torch,
    "Lax-Wendroff": utils.LaxWendroff_torch,
    "Minmod": utils.minmod_torch,
    "Superbee": utils.superbee_torch,
    "Van Leer": utils.vanLeer_torch,
    "Koren": utils.koren_torch,
    "MC": utils.MC_torch,
    "Neural network": neural_flux_limiter,
    # "Piecewise linear": utils.piecewise_linear_limiter,
}

# Load data
path = '~/research/flux_limiter/data/'
path = os.path.expanduser(path)
flnm = '1D_linear_adv_beta1.0_8xCG.pt'
path = os.path.join(path, flnm)
train_loader, test_loader = load_linear_adv_1d(
    path=path,
    n_train=8192,
    n_test=1024,
    train_batch_size=1024,
    test_batch_size=1,
    )

model.eval()
with torch.no_grad():
    for (name, flux_limiter) in flux_limiters.items():
        mse_list = []
        for ibatch, sample in enumerate(test_loader):
            # Solve for the state until the end of time T
            u = fvm_solver.solve_linear_advection_1D_torch(
                u0=sample['x'].to(device),
                T=0.125,
                a=1.0,
                dx=1/(1024/8),
                CFL=0.4,
                model=flux_limiter)
            loss = torch.nn.MSELoss()(u, sample['y'].to(device))
            mse_list.append(loss.detach().cpu())

        mse_list = torch.stack(mse_list)
        mse_mean = mse_list.mean().item()
        mse_std = mse_list.std(unbiased=True).item()
        print(f"{name}:\nmse_mean: {mse_mean}, mse_std: {mse_std}")