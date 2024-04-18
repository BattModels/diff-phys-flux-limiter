import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(3407)

from omegaconf import DictConfig, OmegaConf
import hydra

from .gen_data import gen_1d_adv_data_with_uniform_r

class LinearAdvDataset1D(torch.utils.data.Dataset):
    def __init__(self, sim_data, a, dx, dt, istart, Ns, Nt):
        """
        Parameters:
            sim_data: simulation data, which has shape = (n, t, x), where n is the number of simulation
            a: advection velocity
            dx: spacial grid size
            dt: temporal grid size
            istart: the index where Ns simulation start from in sim_data 
            Ns: number of simulation used to construct training or validation data
            Nt: timesteps used to construct training data
            Nx: number of spacial grid
        """
        super().__init__()
        self.sim_data = sim_data
        self.a = a
        self.dx = dx
        self.dt = dt
        self.CFL = self.a * self.dt / self.dx
        self.istart = istart
        self.Ns = Ns
        self.Nt = Nt
        self.Nx = sim_data.shape[2]
        self.data = None
        self.label = None
        self.r = None
        self.LF = None
        self.HF = None

        self.generate_training_data() # or validation data for validation dataset
        self.generate_label()
        self.compute_r()
        self.compute_low_order_flux()
        self.compute_high_order_flux()

    def generate_training_data(self):
        data = torch.zeros((self.Ns*self.Nt, self.Nx))
        for i in range(self.Ns):
            data[i*self.Nt:(i+1)*self.Nt] = self.sim_data[self.istart+i,:self.Nt,:]
        self.data = data
    
    def generate_label(self):
        label = torch.zeros((self.Ns*self.Nt, self.Nx))
        for i in range(self.Ns):
            label[i*self.Nt:(i+1)*self.Nt] = self.sim_data[self.istart+i,1:self.Nt+1,:]
        self.label = label

    def compute_r(self):
        u = self.data
        ul = torch.roll(u, 1, 1)
        ur = torch.roll(u, -1, 1)

        eps = torch.rand(1).item()*1e-16 # In case that ur - u accidentally equals to -eps so we use rand
        self.r = (u - ul) / (ur - u + eps)
        
    def compute_low_order_flux(self):
        # First order upwind
        if self.a > 0:
            self.LF = self.a * self.data
        else:
            self.LF = self.a * torch.roll(self.data, -1, 1)

    def compute_high_order_flux(self):
        # Lax-Wendroff
        f = self.a * self.data
        self.HF = f + 0.5 * (1 - self.CFL) * (torch.roll(f, -1, 1) - f)

    def __len__(self):
        # Number of data point we have
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Note that the actual input to the neural network is r instead of data
        r = self.r[idx]
        label = self.label[idx]
        LF = self.LF[idx]
        HF = self.HF[idx]
        cur_state = self.data[idx]
        return r, label, LF, HF, cur_state
    
    def plot_r_dist(self, bins, range=None):
        fig, ax = plt.subplots()
        r_np = self.r.numpy().reshape((-1,1))
        ax.hist(r_np,bins=bins,range=range)
        return fig, ax

def load_1d_adv_data_from_h5_file(cfg):
    # For 1d linear advection data, dt = 0.01, dx = 1/1024, grid points are
    # placed at the centers of 1024 cells which divide the interval [0,1] equally
    print("LOADING DATA...")

    path = cfg.data.folder
    path = os.path.expanduser(path)
    flnm = cfg.data.filename
    assert os.path.isfile(path + flnm), 'no such file! ' + path + flnm

    # Read the h5 file and store the data
    # data.shape: (batch, t, x) = (10000, 201, 1024)
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        xcrd = np.array(h5_file["x-coordinate"], dtype=np.float32)
        tcrd = np.array(h5_file["t-coordinate"], dtype=np.float32)
        data = np.array(h5_file["tensor"], dtype=np.float32)

    # print(f"xcrd.shape = {xcrd.shape}")
    # print(f"tcrd = {tcrd}")
    # print(f"tcrd.shape = {tcrd.shape}")
    # print(f"xcrd = {xcrd}")
    # print(f"data.shape = {data.shape}")
    print("FINISH LOADING DATA.")

    # Process the training data
    # 1: We need four states, u(j-2), u(j-1), u(j), u(j+1) at t(n) to calculate u(j) at the next time instant
    # t(n+1) assuming that we are using the low order flux as the first order upwind and the high order flux 
    # as the Lax-Wendroff scheme. Thus, each training data point and its label essentially include five states,
    # i.e., u(j-2), u(j-1), u(j), u(j+1) at t(n) and u(j) at t(n+1) (this is label).
    # 2: Here we adopt another approach: we use the whole states at a time instant as the training data point, 
    # which essentially saves the memory (otherwise the memory required is four times of the original data).
    
    # Because the dt of the training data from PDEBench is too large (does not satisfy the CFL requirement), we
    # need to generate the states after one time step (assume CFL=1 so we do not need to do interpolation) by shift
    # the current states to the right by 1 cell.
    dx = xcrd[1] - xcrd[0]
    dt = tcrd[1] - tcrd[0]
    sim_data = torch.zeros((data.shape[0], 2, data.shape[2]))
    sim_data[:,0,:] = torch.tensor(data[:,0,:], dtype=torch.float)
    sim_data[:,1,:] = torch.roll(sim_data[:,0,:], 1, 1)
    del data
    torch.save(sim_data,"sim_data.pt")

    CFL = 1.
    train_db = LinearAdvDataset1D(sim_data, cfg.solver.velocity, dx, CFL*dx/cfg.solver.velocity, istart=0, Ns=cfg.data.n_train, Nt=1)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=cfg.data.batch_size, shuffle=True)
    validation_db = LinearAdvDataset1D(sim_data, cfg.solver.velocity, dx, CFL*dx/cfg.solver.velocity, istart=cfg.data.n_train, Ns=cfg.data.n_test, Nt=1)
    validation_loader = torch.utils.data.DataLoader(validation_db, batch_size=cfg.data.test_batch_size, shuffle=True)

    return train_db, train_loader, validation_db, validation_loader

def load_1d_adv_data(cfg):
    path = cfg.data.folder
    path = os.path.expanduser(path)
    flnm = cfg.data.filename
    if flnm[-2:] == "pt":
        sim_data = torch.load(os.path.join(path, flnm))
    else:
        sim_data = np.load(os.path.join(path, flnm)).squeeze()
        sim_data = torch.from_numpy(sim_data)

    dx = (cfg.data.xR - cfg.data.xL) / cfg.data.spatial_length
    CFL = cfg.solver.CFL

    train_db = LinearAdvDataset1D(sim_data, cfg.solver.velocity, dx, CFL*dx/cfg.solver.velocity, istart=0, Ns=cfg.data.n_train, Nt=1)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=cfg.data.batch_size, shuffle=True)
    validation_db = LinearAdvDataset1D(sim_data, cfg.solver.velocity, dx, CFL*dx/cfg.solver.velocity, istart=cfg.data.n_train, Ns=cfg.data.n_test, Nt=1)
    validation_loader = torch.utils.data.DataLoader(validation_db, batch_size=cfg.data.test_batch_size, shuffle=True)

    # Plot the histogram of r
    # fig, ax = train_db.plot_r_dist(bins=list(range(0,11)))
    # ax.set_xlabel("r")
    # ax.set_ylabel("counts")
    # fig.savefig("r_histogram")

    return train_db, train_loader, validation_db, validation_loader

def load_1d_rand_adv_data(cfg):
    path = cfg.data.folder
    path = os.path.expanduser(path)
    flnm = 'sim_data_random.pt'
    sim_data = torch.load(os.path.join(path, flnm))

    dx = 1./1024
    CFL = 1.
    train_db = LinearAdvDataset1D(sim_data, cfg.solver.velocity, dx, CFL*dx/cfg.solver.velocity, istart=0, Ns=cfg.data.n_train, Nt=1)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=cfg.data.batch_size, shuffle=True)
    validation_db = LinearAdvDataset1D(sim_data, cfg.solver.velocity, dx, CFL*dx/cfg.solver.velocity, istart=cfg.data.n_train, Ns=cfg.data.n_test, Nt=1)
    validation_loader = torch.utils.data.DataLoader(validation_db, batch_size=cfg.data.test_batch_size, shuffle=True)

    # Plot the histogram of r
    # fig, ax = train_db.plot_r_dist(bins=np.arange(0,10,0.1))
    # ax.set_xlabel("r")
    # ax.set_ylabel("counts")
    # fig.savefig("r_histogram_random")

    return train_db, train_loader, validation_db, validation_loader

def load_1d_adv_data_with_uniform_r(cfg):
    sim_data = gen_1d_adv_data_with_uniform_r(r_end=10., a=cfg.solver.velocity, dx=1./1024, CFL=cfg.solver.CFL, Ns=cfg.data.n_train+cfg.data.n_test)

    dx = 1./1024
    CFL = cfg.solver.CFL
    train_db = LinearAdvDataset1D(sim_data, cfg.solver.velocity, dx, CFL*dx/cfg.solver.velocity, istart=0, Ns=cfg.data.n_train, Nt=1)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=cfg.data.batch_size, shuffle=True)
    validation_db = LinearAdvDataset1D(sim_data, cfg.solver.velocity, dx, CFL*dx/cfg.solver.velocity, istart=cfg.data.n_train, Ns=cfg.data.n_test, Nt=1)
    validation_loader = torch.utils.data.DataLoader(validation_db, batch_size=cfg.data.test_batch_size, shuffle=True)

    # Plot the histogram of r
    fig, ax = train_db.plot_r_dist(bins=np.arange(0,10,0.1))
    ax.set_xlabel("r")
    ax.set_ylabel("counts")
    fig.savefig("r_histogram_uniform")

    return train_db, train_loader, validation_db, validation_loader

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # load_1d_adv_data(cfg)
    load_1d_adv_data_with_uniform_r(cfg)

if __name__ == "__main__":
    main()
    