import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

from models.model import FluxLimiter
from utils import fvm_solver, utils

from omegaconf import DictConfig, OmegaConf
import hydra

torch.manual_seed(3407)

class Dataset(data.Dataset):
    def __init__(self, f, start, end, size):
        """
        Inputs:
            f - Prescribed function
            start - Start point on x-axis
            end - End point on x-axis
            size - Number of data points we want to generate
        """
        super().__init__()
        self.f = f
        self.start = start
        self.end = end
        self.size = size
        self.generate_data()

    def generate_data(self):
        data = self.start + (self.end - self.start) * torch.rand(self.size)
        data = data.view(-1, 1)
        label = self.f(data)

        self.data = data
        self.label = label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

def train_one_epoch(model, optimizer, train_data_loader, device):
    loss = 0.
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_data_loader):
        #print(i, np.array(data).shape)
        #exit()
        # Every data instance is an input + label pair
        inputs, labels = data

        # Move input data to device (only strictly necessary if we use GPU)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = nn.MSELoss()(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i+1)

def validate_one_epoch(model, validation_data_loader, device):
    running_vloss = 0.
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_data_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs.to(device))
            vloss = nn.MSELoss()(voutputs, vlabels.to(device))
            running_vloss += vloss

    avg_vloss = running_vloss / (i+1)

    return avg_vloss.cpu()

def eval_model(model, dataset, device):
    model.eval()
    with torch.no_grad():
        preds = model(dataset.data.to(device))
    return preds

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def pretrain_flux_limiter(cfg: DictConfig) -> None:
    # Setup device
    device = cfg.device

    # Model
    model = FluxLimiter(
        cfg.net.n_input,
        cfg.net.n_output,
        cfg.net.n_hidden,
        cfg.net.n_layers,
        cfg.net.activation,
    )
    model = model.to(device)

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

    # Dataset
    train_dataset = Dataset(utils.vanLeer, start=0, end=100, size=4096)
    validation_dataset = Dataset(utils.vanLeer, start=0, end=100, size=256)
    train_data_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_data_loader = data.DataLoader(validation_dataset, batch_size=16, shuffle=True)

    EPOCHS = 1500
    train_loss_his = []
    val_loss_his = []

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            device=device,
            )

        avg_vloss = validate_one_epoch(
            model=model,
            validation_data_loader=validation_data_loader,
            device=device,
        )

        # Use the scheduler
        scheduler.step(avg_vloss)

        print('LOSS train {} valid {}'.format(train_loss, avg_vloss))
        train_loss_his.append(train_loss)
        val_loss_his.append(avg_vloss)

    torch.save(model.state_dict(), 'pretrained_model.pt')

    fig, ax = plt.subplots()
    ax.plot(train_loss_his, label='training loss')
    ax.plot(val_loss_his, label='validation loss')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig("loss_his")

    train_dataset_preds = eval_model(model, train_dataset, device)
    fig, ax = plt.subplots()
    ax.set_title('Training result')
    ax.plot(train_dataset.data.numpy(), train_dataset.label.numpy(), 'r.')
    ax.plot(train_dataset.data.numpy(), train_dataset_preds.cpu().numpy(), 'b.')
    fig.savefig("train_res")

    validation_dataset_preds = eval_model(model, validation_dataset, device)
    fig, ax = plt.subplots()
    ax.set_title('Validation result')
    ax.plot(validation_dataset.data.numpy(), validation_dataset.label.numpy(), 'r.')
    ax.plot(validation_dataset.data.numpy(), validation_dataset_preds.cpu().numpy(), 'b.')
    fig.savefig("val_res")

if __name__ == "__main__":
    pretrain_flux_limiter()