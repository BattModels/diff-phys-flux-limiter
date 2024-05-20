import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.model import FluxLimiter
from utils import fvm_solver, utils
from data.dataset import load_burgers_1d

from omegaconf import DictConfig, OmegaConf
import hydra

import wandb

torch.manual_seed(3407)

def save_checkpoint(model):
    torch.save(model.state_dict(), 'model_burgers.pt')

    model.eval()
    r_min = -2
    r_max = 10
    n_eval = 1000
    r_eval = np.linspace(r_min, r_max, n_eval)
    with torch.no_grad():
        preds = model(torch.Tensor(r_eval).view(-1,1))
    fig, ax = plt.subplots()
    ax.plot(r_eval, utils.minmod(r_eval), label="minmod")
    ax.plot(r_eval, utils.vanLeer(r_eval), label="van Leer")
    ax.plot(r_eval, utils.superbee(r_eval), label="superbee")
    ax.plot(r_eval, preds.cpu(), label="neural flux limiter")
    ax.set_xlabel('$r$')
    ax.set_ylabel('$\phi(r)$')
    ax.legend()
    fig.savefig('learned_limiter_burgers')

@hydra.main(version_base="1.3", config_path="../configs", config_name="config_burgers")
def train_neural_flux_limiter(cfg: DictConfig) -> None:
    # Setup device
    device = cfg.device

    # Print out the config
    print(OmegaConf.to_yaml(cfg))

    # Initiate wandb
    if cfg.wandb.log:
        wandb.init(
            project="1d-tvd-flux-limiter-burgers", 
            name=f"{cfg.training_type}-{cfg.data.CG}xCG", 
            config={
            "learning_rate": cfg.opt.lr,
            "architecture": "MLP",
            "dataset": cfg.data.filename,
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
    ax.plot(np.linspace(0, 10, 1000), utils.minmod(np.linspace(0, 10, 1000)))
    ax.plot(np.linspace(0, 10, 1000), utils.vanLeer(np.linspace(0, 10, 1000)))
    ax.plot(np.linspace(0, 10, 1000), utils.superbee(np.linspace(0, 10, 1000)))
    ax.plot(np.linspace(-2, 10, 1000), preds.cpu())
    fig.savefig('model_before_finetune_burgers')

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
    
    # Load data
    path = cfg.data.folder
    path = os.path.expanduser(path)
    flnm = cfg.data.filename
    path = os.path.join(path, flnm)
    train_loader, test_loader = load_burgers_1d(
        path=path,
        n_train=cfg.data.n_train,
        n_test=cfg.data.n_test,
        train_batch_size=cfg.data.batch_size,
        test_batch_size=cfg.data.test_batch_size,
        t_end=cfg.data.t_end,
        CG=8,
        )
    
    # model_lin = FluxLimiter(1,1,64,5,act="Tanh")
    # model_lin.load_state_dict(torch.load("model.pt"))
    # model_lin = model_lin.to(device)

    # Train and evaluate
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(cfg.opt.n_epochs):
        print('Epoch {}:'.format(epoch))

        # Begin training
        model.train(True)

        total_train_loss = 0.
        for ibatch, sample in enumerate(train_loader):
            # Solve for the state at the end of time T
            u = fvm_solver.solve_burgers_1D_torch(
                u0=sample['x'].to(device),
                T=cfg.data.t_end,
                dx=(cfg.data.xR - cfg.data.xL) / (cfg.data.spatial_length / cfg.data.CG),
                CFL=cfg.data.CFL,
                model=model)

            loss = nn.MSELoss()(u, sample['y'])
            print(f"train batch loss {ibatch}: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            # if cfg.opt.grad_clipping:
            #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, error_if_nonfinite=True)
            optimizer.step()

            total_train_loss += loss.item()
        
        total_train_loss = total_train_loss/(ibatch+1)
        print(f"Loss train: {total_train_loss}")

        # Validation
        total_valid_loss = 0.
        # total_valid_loss_2 = 0.
        model.eval()
        with torch.no_grad():
            for ibatch, sample in enumerate(test_loader):
                # Solve for the state at the end of time T
                u = fvm_solver.solve_burgers_1D_torch(
                    u0=sample['x'].to(device),
                    T=cfg.data.t_end,
                    dx=(cfg.data.xR - cfg.data.xL) / (cfg.data.spatial_length / cfg.data.CG),
                    CFL=cfg.data.CFL,
                    model=model)
                
                loss = nn.MSELoss()(u, sample['y'])
                print(f"valid batch loss {ibatch}: {loss.item()}")

                total_valid_loss += loss.item()

            total_valid_loss = total_valid_loss/(ibatch+1)
            print(f"Loss valid: {total_valid_loss}")

            # The second validation of flux limiter trained by linear advection
            # Check whether superbee is superior than linear neural flux limiter
            # for ibatch, sample in enumerate(test_loader):
            #     # Solve for the state at the end of time T
            #     u = fvm_solver.solve_burgers_1D_torch(
            #         u0=sample['x'].to(device),
            #         T=cfg.data.t_end,
            #         dx=(cfg.data.xR - cfg.data.xL) / (cfg.data.spatial_length / cfg.data.CG),
            #         CFL=cfg.data.CFL,
            #         model=model_lin)
                
            #     loss = nn.MSELoss()(u, sample['y'])
            #     print(f"Second valid batch loss {ibatch}: {loss.item()}")

            #     total_valid_loss_2 += loss.item()

            # total_valid_loss_2 = total_valid_loss_2/(ibatch+1)
            # print(f"Second Loss valid: {total_valid_loss_2}")
            
        scheduler.step(total_valid_loss)

        if cfg.wandb.log:
            wandb.log({"train loss": total_train_loss,
                        "valid loss": total_valid_loss,
                        # "valid loss of lin fl": total_valid_loss_2,
                        "learning rate": scheduler.optimizer.param_groups[0]['lr']})

        if (epoch % cfg.opt.n_checkpoint == 0 or epoch == cfg.opt.n_epochs):
            save_checkpoint(model)

    if cfg.wandb.log:
        wandb.finish()

if __name__ == "__main__":
    train_neural_flux_limiter()