import torch
torch.manual_seed(3407)

# Here we generate Ns random initial conditions in [0,L] with Nx cells each
# We also shift the data to the right by 1 cells to get the advected states
# after one time steps
def gen_1d_rand_adv_data(a, L, Nx, Ns):
    CFL = 1.
    a = 1.
    dx = L/Nx
    dt = CFL*dx/a
    
    data = torch.zeros((Ns, 2, Nx))
    data[:,0,:] = torch.rand(Ns, Nx)
    data[:,1,:] = torch.roll(data[:,0,:], 1, 1)

    torch.save(data, "./data/sim_data_random.pt")

    return data

if __name__ == "__main__":
    gen_1d_rand_adv_data(a=1., L=1., Nx=1024, Ns=10000)




