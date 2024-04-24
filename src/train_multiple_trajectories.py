import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.model import FluxLimiter
from utils import fvm_solver, utils
from data.linear_adv_dataset import load_linear_adv_1d

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
            name=f"{cfg.training_type}-pretrain?{cfg.net.with_pretrain}-{cfg.data.CG}xCG-grad_clipping?{cfg.opt.grad_clipping}", 
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
    
    # Load data
    path = cfg.data.folder
    path = os.path.expanduser(path)
    flnm = cfg.data.filename
    path = os.path.join(path, flnm)
    train_loader, test_loader = load_linear_adv_1d(
        path=path,
        n_train=cfg.data.n_train,
        n_test=cfg.data.n_test,
        train_batch_size=cfg.data.batch_size,
        test_batch_size=cfg.data.test_batch_size,
        )

    # Train and evaluate
    for epoch in range(cfg.opt.n_epochs):
        print('Epoch {}:'.format(epoch))

        # Begin training
        model.train(True)

        total_train_loss = 0.
        for ibatch, sample in enumerate(train_loader):
            # Solve for the state at the end of time T
            u = fvm_solver.solve_linear_advection_1D_torch(
                u0=sample['x'].to(device),
                T=cfg.data.t_end,
                a=cfg.data.velocity,
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

            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param)
            #         print(name, param.grad)

            total_train_loss += loss.item()
        
        total_train_loss = total_train_loss/(ibatch+1)
        print(f"Loss train: {total_train_loss}")

        # Validation
        total_valid_loss = 0.
        model.eval()
        with torch.no_grad():
            for ibatch, sample in enumerate(test_loader):
                # Solve for the state at the end of time T
                u = fvm_solver.solve_linear_advection_1D_torch(
                    u0=sample['x'].to(device),
                    T=cfg.data.t_end,
                    a=cfg.data.velocity,
                    dx=(cfg.data.xR - cfg.data.xL) / (cfg.data.spatial_length / cfg.data.CG),
                    CFL=cfg.data.CFL,
                    model=model)
                
                loss = nn.MSELoss()(u, sample['y'])
                print(f"valid batch loss {ibatch}: {loss.item()}")

                total_valid_loss += loss.item()

            total_valid_loss = total_valid_loss/(ibatch+1)
            print(f"Loss valid: {total_valid_loss}")
            
        scheduler.step(total_valid_loss)

        if cfg.wandb.log:
            wandb.log({"train loss": total_train_loss,
                        "valid loss": total_valid_loss,
                        "learning rate": scheduler.optimizer.param_groups[0]['lr']})

        if (epoch % cfg.opt.n_checkpoint == 0 or epoch == cfg.opt.n_epochs):
            save_checkpoint(model)

    if cfg.wandb.log:
        wandb.finish()

if __name__ == "__main__":
    train_neural_flux_limiter()