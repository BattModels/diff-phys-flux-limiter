import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.model import FluxLimiter
from utils import fvm_solver, utils
from data.load_data import load_1d_adv_data, load_1d_adv_data_with_uniform_r

from omegaconf import DictConfig, OmegaConf
import hydra

import wandb

torch.manual_seed(3407)

def save_checkpoint(model):
    torch.save(model.state_dict(), 'model.pt')

    model.eval()
    r_min = -10
    r_max = 50
    n_eval = 1000
    r_eval = np.linspace(r_min, r_max, n_eval)
    with torch.no_grad():
        preds = model(torch.Tensor(r_eval).view(-1,1))
    fig, ax = plt.subplots()
    ax.plot(r_eval, preds.cpu(), '.', label="neural flux limiter")
    ax.plot(r_eval, utils.vanLeer(r_eval), label="van Leer")
    ax.legend()
    fig.savefig('learned_limiter_linear_case')

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train_neural_flux_limiter(cfg: DictConfig) -> None:
    # Setup device
    device = cfg.device

    # Print out the config
    print(OmegaConf.to_yaml(cfg))

    # Initiate wandb
    if cfg.wandb.log:
        wandb.init(
            project="1D-flux-limiter-linear-adv", 
            name=f"{cfg.training_type}-{cfg.solver.initial_data}-pretrain?{cfg.net.with_pretrain}-grad_clipping?{cfg.opt.grad_clipping}", 
            config={
            "learning_rate": cfg.opt.lr,
            "architecture": "MLP",
            "dataset": cfg.solver.initial_data,
            "epochs": cfg.opt.n_epochs,
            })    

    # Model
    model = FluxLimiter(
        cfg.net.n_input,
        cfg.net.n_output,
        cfg.net.n_hidden,
        cfg.net.n_layers,
        cfg.net.activation,
    )

    # Load pretrained model if with_pretrain == True
    if cfg.net.with_pretrain:
        model.load_state_dict(torch.load("pretrained_model.pt"))

    model = model.to(device)

    if cfg.wandb.log:
        wandb.watch(model, log="all", log_freq=1)

    # Plot the flux limiter before training
    model.eval()
    with torch.no_grad():
        preds = model(torch.linspace(-2, 10, 1000).view(-1,1))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(-2, 10, 1000), preds.cpu())
    ax.plot(np.linspace(0, 10, 1000), utils.vanLeer(np.linspace(0, 10, 1000)))
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
    elif cfg.opt.scheduler == "CosineAnnealingLR":
         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.opt.n_epochs,
        )       
    else:
        raise ValueError(f"No scheduler type: {cfg.opt.scheduler}")
    
    # Shape: [10000,2,1024], the second row is useless (the simulation data after one step)
    sim_data = torch.load("./data/sim_data.pt")

    # Coarse graining to make sure that the number of timesteps are not too big to make it hard for backpropagation 
    cg = 8
    dx = (cfg.data.xR - cfg.data.xL) / (cfg.data.spatial_length / cg)
    sim_data = sim_data[:,:,::cg]

    # Shift the initial conditions right to get the true solutions
    n_shift = 16
    sim_data[:,1,:] = torch.roll(sim_data[:,0,:], n_shift, 1)

    n_train = cfg.data.n_train
    n_test = cfg.data.n_test

    train_data = sim_data[:n_train,:,:].to(device)
    test_data = sim_data[n_train:n_train+n_test,:,:].to(device)

    for epoch in range(cfg.opt.n_epochs):
        print('Epoch {}:'.format(epoch))

        # Begin training
        model.train(True)

        # Shuffle
        perm = torch.randperm(train_data.shape[0])

        total_train_loss = 0.
        for ibatch in range(cfg.data.n_train//cfg.data.batch_size):
            # Solve for the state at the end of time T
            u0 = train_data[perm[(ibatch*cfg.data.batch_size):((ibatch+1)*cfg.data.batch_size)],0,:]

            u = fvm_solver.solve_linear_advection_1D_torch(
                u0=u0,
                # T=cfg.solver.t_end,
                T=n_shift/sim_data.shape[2],
                a=cfg.solver.velocity,
                dx=dx,
                CFL=cfg.solver.CFL,
                model=model)
            

            u_true = train_data[perm[(ibatch*cfg.data.batch_size):((ibatch+1)*cfg.data.batch_size)],1,:]

            loss = nn.MSELoss()(u, u_true)
            print(f"train batch loss {ibatch}: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            # if cfg.opt.grad_clipping:
            #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, error_if_nonfinite=True)
            optimizer.step()

            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param)
            #         print(name, param.grad)

            total_train_loss += loss.item()
        
        total_train_loss = total_train_loss/(ibatch+1)
        print(f"Loss train: {total_train_loss}")

        # Validation

        # Shuffle
        perm = torch.randperm(test_data.shape[0])

        total_valid_loss = 0.
        model.eval()

        with torch.no_grad():
            for ibatch in range(cfg.data.n_test//cfg.data.test_batch_size):
                # Solve for the state at the end of time T
                u0 = test_data[perm[(ibatch*cfg.data.test_batch_size):((ibatch+1)*cfg.data.test_batch_size)],0,:]

                u = fvm_solver.solve_linear_advection_1D_torch(
                    u0=u0,
                    # T=cfg.solver.t_end,
                    T=n_shift/sim_data.shape[2],
                    a=cfg.solver.velocity,
                    dx=dx,
                    CFL=cfg.solver.CFL,
                    model=model)
                
                u_true = test_data[perm[(ibatch*cfg.data.batch_size):((ibatch+1)*cfg.data.batch_size)],1,:]
                loss = nn.MSELoss()(u, u_true)
                print(f"valid batch loss {ibatch}: {loss.item()}")

                total_valid_loss += loss.item()

            total_valid_loss = total_valid_loss/(ibatch+1)
            print(f"Loss valid: {total_valid_loss}")
            
        scheduler.step(total_valid_loss)

        if cfg.wandb.log:
            wandb.log({"train loss": total_train_loss,
                        "valid loss": total_valid_loss,
                        "learning rate": scheduler.optimizer.param_groups[0]['lr']})

        if (epoch % cfg.opt.n_checkpoint == 0):
            save_checkpoint(model)

    if cfg.wandb.log:
        wandb.finish()

if __name__ == "__main__":
    train_neural_flux_limiter()