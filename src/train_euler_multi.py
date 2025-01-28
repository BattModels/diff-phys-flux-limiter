import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.model import FluxLimiter, SymmetricFluxLimiter
from utils import fvm_solver, utils

from omegaconf import DictConfig, OmegaConf
import hydra

import wandb

torch.manual_seed(3407)

from scipy.optimize import fsolve
def shock_tube_func(p4, p1, p5, rho1, rho5, u1, u5, gamma):
    # Only solve the case where the solution is composed by:
    # left: expansion fan
    # mid: contact discontinuity
    # right: shock
    c1 = np.sqrt(gamma*p1/rho1)
    c5 = np.sqrt(gamma*p5/rho5)

    gm1 = gamma - 1
    gp1 = gamma + 1
    g2 = 2. * gamma

    f_expansion_fan = 2*c1/gm1 * ((p4/p1)**(gm1/g2)-1)
    f_shock = (p4 - p5) / (rho5*c5 * np.sqrt(gp1/g2 * p4/p5 + gm1/g2))

    u3 = u1 - f_expansion_fan
    u4 = u5 + f_shock

    return u4 - u3

def analytic_sod(left_state, right_state, npts=1000, t=0.2, L=1., gamma=1.4):
    # Define some auxiliary variables
    gm1 = gamma - 1
    gp1 = gamma + 1
    g2 = 2. * gamma

    # Define positions where we need to solve for the states
    # x_arr = np.linspace(0, L, npts)

    # This is another way to define the same mesh as that 
    # used in numerical solution so that we can compare the MSE loss
    dx = L / npts                      # Cell size
    x_arr = (np.arange(npts)+0.5)*dx   # Mesh

    rho1, u1, p1 = left_state
    rho5, u5, p5 = right_state

    p4 = fsolve(shock_tube_func, 0.5, (p1, p5, rho1, rho5, u1, u5, gamma))[0]

    # Compute post-shock density and velocity
    c5 = np.sqrt(gamma*p5/rho5)
    
    f_shock = (p4 - p5) / (rho5*c5 * np.sqrt(gp1/g2 * p4/p5 + gm1/g2))
    u4 = u5 + f_shock
    s = u5 - (p4 - p5) / (rho5*(u5 - u4))           # Shock speed
    rho4 = rho5 * (u5 - s) / (u4 - s)
    
    # Compute values at foot of the rarefaction
    p3 = p4
    u3 = u4
    rho3 = rho1 * (p3 / p1) ** (1. / gamma)         # Use the isentropic relation

    # Compute the position of each structure
    c1 = np.sqrt(gamma*p1/rho1)
    c3 = np.sqrt(gamma*p3/rho3)
    xi = L / 2                                      # Initial position of the barrier
    xsh = xi + s * t                                # Shock
    xcd = xi + u3 * t                               # Contact discontinuity
    xft = xi + (u3 - c3) * t                        # Foot of rarefaction
    xhd = xi + (u1 - c1) * t                        # Head of rarefaction

    # Compute the states at each points
    rho = np.zeros(npts, dtype=float)
    u = np.zeros(npts, dtype=float)
    p = np.zeros(npts, dtype=float)
    for i, x in enumerate(x_arr):
        if x < xhd:
            rho[i] = rho1
            u[i] = u1
            p[i] = p1
        elif x < xft:
            c = gm1/gp1 * (u1 - (x - xi) / t) + 2*c1/gp1
            u[i] = c + (x - xi) / t
            p[i] = p1 * (c/c1)**(g2/gm1)
            rho[i] = gamma * p[i] / c**2
        elif x < xcd:
            rho[i] = rho3
            u[i] = u3
            p[i] = p3
        elif x < xsh:
            rho[i] = rho4
            u[i] = u4
            p[i] = p4
        else:
            rho[i] = rho5
            u[i] = u5
            p[i] = p5
    
    return x_arr, rho, u, p

def save_checkpoint(model, device):
    device = next(model.parameters()).device
    torch.save(model.state_dict(), 'model_euler_multi.pt')

    model.eval()
    r_min = -2
    r_max = 10
    n_eval = 1000
    r_eval = np.linspace(r_min, r_max, n_eval)
    with torch.no_grad():
        preds = model(torch.Tensor(r_eval).view(-1,1).to(device))
    fig, ax = plt.subplots()
    ax.plot(r_eval, utils.minmod(r_eval), label="minmod")
    ax.plot(r_eval, utils.vanLeer(r_eval), label="van Leer")
    ax.plot(r_eval, utils.superbee(r_eval), label="superbee")
    ax.plot(r_eval, preds.cpu().numpy(), label="neural flux limiter")
    ax.set_xlabel('$r$')
    ax.set_ylabel('$\phi(r)$')
    ax.legend()
    fig.savefig('learned_limiter_euler', dpi=300)
    plt.close()

def calc_conserved_quantities(left_state, right_state, x_ini, x_fin, n_cells, gamma=1.4):
    # n_cells = 100
    # t = 0.1
    left_state = torch.tensor(left_state, dtype=torch.float32)
    right_state = torch.tensor(right_state, dtype=torch.float32)
                
    dx = (x_fin - x_ini) / n_cells            # Cell size
    x = (torch.arange(n_cells)+0.5)*dx + x_ini   # Mesh

    r0 = torch.where(x < 0.5*(x_ini + x_fin), left_state[0]*torch.ones(n_cells), right_state[0]*torch.ones(n_cells))        # Density
    u0 = torch.where(x < 0.5*(x_ini + x_fin), left_state[1]*torch.ones(n_cells), right_state[1]*torch.ones(n_cells))        # Velocity
    p0 = torch.where(x < 0.5*(x_ini + x_fin), left_state[2]*torch.ones(n_cells), right_state[2]*torch.ones(n_cells))        # Pressure

    E0 = p0/(gamma-1.) + 0.5*r0*u0**2      # Energy per unit volume
    initial_q  = torch.stack((r0, r0*u0, E0))        # Vector of conserved variables

    mesh = x

    return mesh, initial_q

@hydra.main(version_base="1.3", config_path="../configs", config_name="config_euler")
def train_neural_flux_limiter(cfg: DictConfig) -> None:
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    # Initiate wandb
    if cfg.wandb.log:
        wandb.init(
            project="1d-tvd-flux-limiter-euler-multi", 
            name=f"", 
            config={
            "learning_rate": cfg.opt.lr,
            "architecture": "MLP",
            # "dataset": cfg.data.filename,
            "epochs": cfg.opt.n_epochs,
            })    

    # Print out the config
    print(OmegaConf.to_yaml(cfg))

    # Model
    model = SymmetricFluxLimiter(
        cfg.net.n_input,
        cfg.net.n_output,
        cfg.net.n_hidden,
        cfg.net.n_layers,
        cfg.net.activation,
    )

    model = model.to(device)

    # if cfg.wandb.log:
    #     wandb.watch(model, log="all", log_freq=1)

    # Plot the flux limiter before training
    model.eval()
    with torch.no_grad():
        preds = model(torch.linspace(-2, 10, 1000).view(-1,1).to(device))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), utils.minmod(np.linspace(0, 10, 1000)))
    ax.plot(np.linspace(0, 10, 1000), utils.vanLeer(np.linspace(0, 10, 1000)))
    ax.plot(np.linspace(0, 10, 1000), utils.superbee(np.linspace(0, 10, 1000)))
    ax.plot(np.linspace(-2, 10, 1000), preds.cpu().numpy())
    fig.savefig('initial_model_euler', dpi=300)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.opt.lr,
        weight_decay=cfg.opt.weight_decay,
    )

    # Scheduler
    if cfg.opt.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.opt.factor,
            patience=cfg.opt.patience,
            mode="min",
        )
    elif cfg.opt.scheduler == "CosineAnnealingLR":
         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.opt.n_epochs,
        )
    elif cfg.opt.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5,
        )
    else:
        raise ValueError(f"No scheduler type: {cfg.opt.scheduler}")
    
    # Exact solution
    data_folder = os.path.expanduser('~/research/flux_limiter/data/shock_tube')
    initial_states = np.load(os.path.join(data_folder, 'initial_states.npy')) # (50, 6)
    solutions = np.load(os.path.join(data_folder, 'solutions.npy')) # (50, 3, 100)
    
    n_cells = solutions.shape[-1]

    mesh, initial_q_sod = calc_conserved_quantities(left_state=(1., 0., 1.), 
                                                    right_state=(0.125, 0., 0.1), 
                                                    x_ini=0., 
                                                    x_fin=1., 
                                                    n_cells=n_cells)

    initial_qs = torch.zeros((initial_states.shape[0], 3, n_cells))
    for i in range(initial_states.shape[0]):
        _, initial_q = calc_conserved_quantities(left_state=initial_states[i, 0:3], 
                                                    right_state=initial_states[i, 3:6], 
                                                    x_ini=0., 
                                                    x_fin=1., 
                                                    n_cells=n_cells)
        initial_qs[i] = initial_q
    
    # x_a, rho_a, u_a, p_a = analytic_sod(left_state=(1., 0., 1.), 
    #                                     right_state=(0.125, 0., 0.1), 
    #                                     npts=n_cells,
    #                                     t=t,
    #                                     )
    
    x_a_valid, rho_a_valid, u_a_valid, p_a_valid = analytic_sod(left_state=(1., 0., 1.), 
                                        right_state=(0.125, 0., 0.1), 
                                        npts=n_cells,
                                        t=0.2,
                                        )
    
    # Train and evaluate
    for epoch in range(cfg.opt.n_epochs):
        print('Epoch {}:'.format(epoch))

        # Begin training
        model.train(True)

        start_time = time.time()
        train_loss = 0.0
        for i in range(20):
            initial_q = initial_qs[i]
            x, rho, u, p = fvm_solver.solve_euler_1d_torch(mesh=mesh.to(device),
                                                        initial_q=initial_q.to(device),
                                                        flux_limiter=model, 
                                                        t_end=0.1, 
                                                        CFL=0.4,
                                                        gamma=1.4,
                                                        )
            loss = nn.MSELoss()(rho, torch.tensor(solutions[i, 0, :], dtype=torch.float32, device=device))
            print(f"loss {i}: {loss.item()}")
            train_loss += loss
        train_loss /= i + 1
        epoch_time = time.time() - start_time
        print(f"Loss train: {train_loss.item()}")
        print(f"epoch time: {epoch_time}")

        optimizer.zero_grad()
        train_loss.backward()
        # if cfg.opt.grad_clipping:
        #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, error_if_nonfinite=True)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x, rho, u, p = fvm_solver.solve_euler_1d_torch(mesh=mesh.to(device),
                                                       initial_q=initial_q_sod.to(device),
                                                       flux_limiter=model, 
                                                       t_end=0.2, 
                                                       CFL=0.4,
                                                       gamma=1.4,
                                                       )
            valid_loss = nn.MSELoss()(rho, torch.tensor(rho_a_valid, dtype=torch.float32, device=device))
            print(f"valid loss {epoch}: {valid_loss.item()}")

        scheduler.step(valid_loss.item())

        if cfg.wandb.log:
            wandb.log({"train loss": train_loss,
                        "valid loss": valid_loss,
        #                 "valid loss of lin fl": total_valid_loss_2,
                        "learning rate": scheduler.optimizer.param_groups[0]['lr'],
                        })

        if (epoch % cfg.opt.n_checkpoint == 0 or epoch == cfg.opt.n_epochs - 1):
            save_checkpoint(model, device)

    if cfg.wandb.log:
        wandb.finish()

if __name__ == "__main__":
    train_neural_flux_limiter()