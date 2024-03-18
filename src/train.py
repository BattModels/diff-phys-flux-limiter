import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.model import FluxLimiter
from utils import fvm_solver, utils
from data.load_data import load_1d_adv_data

from omegaconf import DictConfig, OmegaConf
import hydra

import wandb

torch.manual_seed(3407)

def train_one_epoch(model, optimizer, train_db, train_data_loader, device):
    loss = 0.
    running_loss = 0.

    for i, data in enumerate(train_data_loader):
        r, labels, LF, HF, cur_state = data

        r = r.to(device)
        labels = labels.to(device)
        LF = LF.to(device)
        HF = HF.to(device)
        cur_state = cur_state.to(device)

        phi = model(r.view(-1,1))
        phi = phi.view(labels.shape[0], labels.shape[1])
        F = (1 - phi) * LF + phi * HF
        preds = cur_state - train_db.dt/train_db.dx * (F - torch.roll(F, 1, 1))

        loss = nn.MSELoss()(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # return element-wise error
    return running_loss / (i+1)

def validate_one_epoch(model, validation_db, validation_data_loader, device):
    running_vloss = 0.

    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_data_loader):
            r, labels, LF, HF, cur_state = vdata

            r = r.to(device)
            labels = labels.to(device)
            LF = LF.to(device)
            HF = HF.to(device)
            cur_state = cur_state.to(device)

            phi = model(r.view(-1,1))
            phi = phi.view(labels.shape[0], labels.shape[1])
            F = (1 - phi) * LF + phi * HF
            preds = cur_state - validation_db.dt/validation_db.dx * (F - torch.roll(F, 1, 1))

            vloss = nn.MSELoss()(preds, labels)
            running_vloss += vloss.item()

    return running_vloss / (i+1)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train_neural_flux_limiter(cfg: DictConfig) -> None:
    # Setup device
    device = cfg.device

    # Print out the config
    print(OmegaConf.to_yaml(cfg))

    # Initiate wandb
    if cfg.wandb.log:
        if cfg.training_type == "solver in the loop":
            wandb.init(
                project="1D-flux-limiter-linear-adv", 
                name=f"{cfg.training_type}-{cfg.solver.initial_data}-pretrain?{cfg.net.with_pretrain}-grad_clipping?{cfg.opt.grad_clipping}", 
                config={
                "learning_rate": cfg.opt.lr,
                "architecture": "MLP",
                "dataset": cfg.solver.initial_data,
                "epochs": cfg.opt.n_epochs,
                })
        else: # training_type == single_step
            wandb.init(
                project="1D-flux-limiter-linear-adv", 
                name=f"{cfg.training_type}-pretrain?{cfg.net.with_pretrain}-grad_clipping?{cfg.opt.grad_clipping}", 
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
        preds = model(torch.linspace(-10, 100, 1000).view(-1,1))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(-10, 100, 1000), preds.cpu())
    ax.plot(np.linspace(0, 100, 1000), utils.vanLeer(np.linspace(0, 100, 1000)))
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

    if cfg.training_type == "solve in the loop":
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
            if cfg.opt.grad_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step(loss.item())

            if cfg.wandb.log:
                wandb.log({"train loss": loss.item()})
            print(f"Loss train: {loss.item()}")

            # if (epoch<5):
            #     print(epoch)
            #     for name, param in model.named_parameters():
            #         if param.requires_grad:
            #             print(name, param)
            #             print(name, param.grad)
    
    else: # training_type == "single_step"
        train_db, train_loader, validation_db, validation_loader = load_1d_adv_data(cfg)

        # Begin training
        model.train(True)
        train_loss_his = []
        val_loss_his = []

        for epoch in range(cfg.opt.n_epochs):
            print('EPOCH {}:'.format(epoch))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            train_loss = train_one_epoch(
                model=model,
                optimizer=optimizer,
                train_db=train_db,
                train_data_loader=train_loader,
                device=device,
                )

            vloss = validate_one_epoch(
                model=model,
                validation_db=validation_db,
                validation_data_loader=validation_loader,
                device=device,
            )

            # Use the scheduler
            scheduler.step(vloss)

            print('LOSS train {} valid {}'.format(train_loss, vloss))

            if cfg.wandb.log:
                wandb.log({"train loss": train_loss, "val loss": vloss})

    torch.save(model.state_dict(), 'model.pt')

    ############### EVALUATION ###############
    # fig, ax = plt.subplots()
    # ax.plot(x0, u0.numpy(),'b')
    # ax.plot(x0, u_true.numpy(),'r')
    # fig.savefig('doublecheck_u_true')

    model.eval()
    r_min = -10
    r_max = 50
    n_eval = 1000
    r_eval = np.linspace(r_min, r_max, n_eval)
    with torch.no_grad():
        preds = model(torch.Tensor(r_eval).view(-1,1))
    fig, ax = plt.subplots()
    ax.plot(r_eval, preds.cpu(), '.')
    ax.plot(r_eval, utils.vanLeer(r_eval))
    fig.savefig('learned_limiter_linear_case')

    if cfg.wandb.log:
        wandb.finish()

if __name__ == "__main__":
    train_neural_flux_limiter()



