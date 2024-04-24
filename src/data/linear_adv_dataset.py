import torch
import numpy as np
torch.manual_seed(3407)

from omegaconf import DictConfig, OmegaConf
import hydra

def load_linear_adv_1d(path, n_train, n_test, train_batch_size, test_batch_size):
    data = torch.load(path).float()

    # Shuffle the whole dataset
    perm = torch.randperm(data.shape[0])

    x_train = data[perm[0:n_train],0,:]
    y_train = data[perm[0:n_train],1:,:]

    x_test = data[perm[n_train:(n_train + n_test)],0,:]
    y_test = data[perm[n_train:(n_train + n_test)],1:,:]

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=train_batch_size,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        assert (x.size(0) == y.size(0)), "Size mismatch between tensors."
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return {'x': x, 'y': y}

    def __len__(self):
        return self.x.size(0)
    
def load_dataset_with_CG(path, CG):
    # path = "./data/1D_Advection_Sols_beta1.0.npy"

    # The shape of sim_data: [1,10000,321,1024]
    data = np.load(path)

    sim_data = np.zeros((data.shape[1], (data.shape[2]-1)//CG+1, data.shape[3]//CG))
    sim_data[:,:,:] = data[0,:,::CG,::CG]
    del data
    torch.save(torch.from_numpy(sim_data),"1D_linear_adv_beta1.0_" + str(CG) + "xCG.pt")