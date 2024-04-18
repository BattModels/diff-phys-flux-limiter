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

def gen_1d_adv_data_with_uniform_r(r_end, a, dx, CFL, Ns):
    # r = torch.linspace(0,r_end,Ns+1)
    r = r_end * torch.rand(Ns+1)

    # The four states are: [0, 1, 1+1/r, 0]
    data = torch.zeros((Ns,2,4))
    data[:,0,-1] = 0 - 1/r[1:]
    data[:,0,1] = 1.
    data[:,0,2] = 1 + 1/r[1:]

    # Advect the ICs by method of characteristics
    # The four states located at the cells centers [dx/2, 3dx/2, 5dx/2, 7dx/2]
    xc = [dx/2, 3*dx/2, 5*dx/2, 7*dx/2]
    dt = CFL * dx/a

    ul = torch.roll(data[:,0,:], 1, 1)
    u = data[:,0,:]
    data[:,1,:] = CFL*ul + (1-CFL)*u

    return data

if __name__ == "__main__":
    # gen_1d_rand_adv_data(a=1., L=1., Nx=1024, Ns=10000)

    data = gen_1d_adv_data_with_uniform_r(r_end=10., a=1., dx=1./1024, CFL=0.4, Ns=10000)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(data[0,0,:])
    ax.plot(data[0,1,:])
    fig.savefig("check")

