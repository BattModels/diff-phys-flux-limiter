import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.model import FluxLimiter
from utils import fvm_solver, utils

from omegaconf import DictConfig, OmegaConf
import hydra

torch.manual_seed(3407)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train_neural_flux_limiter(cfg: DictConfig) -> None:
    # Setup device
    device = cfg.device

    # Print out the config
    print(OmegaConf.to_yaml(cfg))

    # Model
    model = FluxLimiter(
        cfg.net.n_input,
        cfg.net.n_output,
        cfg.net.n_hidden,
        cfg.net.n_layers,
    )
    model.load_state_dict(torch.load("pretrained_model.pt"))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(torch.linspace(-10, 100, 1000).view(-1,1))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(-10, 100, 1000), preds.cpu())
    # ax.plot(np.linspace(0, 100, 1000), utils.vanLeer(np.linspace(0, 100, 1000)))
    fig.savefig('model_before_finetune')

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
    else:
        raise ValueError(f"No scheduler type: {cfg.opt.scheduler}")

    # Initial data
    n_cells = cfg.solver.n_cells
    if cfg.solver.initial_data == "step":
        x0, u0 = utils.step(N=n_cells)
    elif cfg.solver.initial_data == "square_wave":
        x0, u0 = utils.square_wave(N=n_cells)
    elif cfg.solver.initial_data == "sin_wave":
        x0, u0 = utils.sin_wave(N=n_cells)
    else:
        raise ValueError(f"No initial data type: {cfg.solver.initial_data}")

    u0 = torch.Tensor(u0)
    dx = x0[1] - x0[0]
    u_true = u0
    
    # Begin training
    model.train(True)
    for epoch in range(cfg.opt.n_epochs):
        print('Epoch {}:'.format(epoch))
        
        # Solve for the state at the end of time T
        u = fvm_solver.solve_linear_advection_1D_torch(
            u0=u0,
            T=cfg.solver.t_end,
            a=cfg.solver.velocity,
            dx=dx,
            CFL=cfg.solver.CFL,
            model=model)
        
        # Only for sin initial data
        if cfg.solver.initial_data == "sin_wave":
            u_true = torch.sin(2*torch.pi*(torch.Tensor(x0)-cfg.solver.velocity*cfg.solver.t_end))

        loss = nn.MSELoss()(u, u_true)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step(loss.item())

        # wandb.log({"train loss": loss.item()})
        print(f"Loss train: {loss.item()}")

    torch.save(model.state_dict(), 'model.pt')

    # for param in model.parameters():
    #     print(param)
    
    #######################
    fig, ax = plt.subplots()
    ax.plot(x0, u0.numpy(),'b')
    ax.plot(x0, u_true.numpy(),'r')
    fig.savefig('aa')


    model.eval()
    with torch.no_grad():
        preds = model(torch.linspace(-10, 100, 1000).view(-1,1))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(-10, 100, 1000), preds.cpu(), '.')
    ax.plot(np.linspace(-10, 100, 1000), utils.vanLeer(np.linspace(-10, 100, 1000)))
    fig.savefig('learned_limiter_linear_case')

if __name__ == "__main__":
    train_neural_flux_limiter()



