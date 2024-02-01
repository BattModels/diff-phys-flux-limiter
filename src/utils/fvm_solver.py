import torch
import numpy as np

from utils import utils

# L = 2*np.pi
# N = 101
# dx = L / N
# x0 = np.zeros(N)
# for i in range(N):
#     x0[i] = (i+0.5)*dx
# u0 = np.sin(x0)

def analytic_flux_burgers(u):
    return u**2/2

# def solve_burgers(u0, T, dt, dx, nu=0.01):
#     n_timesteps = int(T/dt)

#     u = u0.copy()
#     a = np.zeros_like(u)
#     f = analytic_flux(u)

#     for i in range(n_timesteps):
#         a = (np.roll(f, -1) - f) / (np.roll(u, -1) - u)
#         u = f + 0.5 * (1 - a*dt/dx) * (np.roll(f, -1) - f) - (np.roll(f, 1) + 0.5 * (1 - np.roll(a, 1)) * (f - np.roll(f, 1)))

#     return u

# u = solve_burgers(u0, 0.1, 1e-3, dx)

# fig, ax = plt.subplots()
# ax.plot(x0, u0)
# ax.plot(x0, u)
# plt.show()

def solve_burgers_1D(u0, T, dt, dx, flux_limiter, nu=0.01):
    n_timesteps = int(T/dt)
    u_all = np.zeros((n_timesteps+1, u0.shape[0]))
    u_all[0] = u0.copy()

    u = u0.copy()
    a = np.zeros_like(u)
    f = np.zeros_like(u)

    for i in range(n_timesteps):
        f = analytic_flux_burgers(u)
        a = (np.roll(f, -1) - f) / (np.roll(u, -1) - u) # a_{i+1/2}

        # If nan exists in array a, it means that the states of left and right cells are the same,
        # then we should use the derivate of u**2/2, i.e., u itself
        a[np.isnan(a)] = u[np.isnan(a)]
        a[np.isinf(a)] = u[np.isinf(a)]
        
        r = compute_r(u, np.roll(u, 1), np.roll(u, -1)) # r_{i+1/2}
        # r_plus = compute_r(u, np.roll(u, 1), np.roll(u, -1))
        # r_minus = compute_r(np.roll(u, 1), np.roll(u, 2), u) # equivalent to np.roll(r_plus, 1)

        F_low = 0.5 * (f + np.roll(f, -1)) - 0.5 * np.abs(a) * (np.roll(u, -1) - u)
        F_high = f + 0.5 * (1 - a*dt/dx) * (np.roll(f, -1) - f)
        # F_high = 0.5 * (f + np.roll(f, -1)) - 0.5 * a**2 * dt/dx * (np.roll(u, -1) - u)

        phi = flux_limiter(r)
        F = (1 - phi) * F_low + phi * F_high 

        u -= dt/dx * (F - np.roll(F, 1))

        u_all[i+1] = u.copy() 

        CFL = np.max(np.abs(a))*dt/dx
        print(f"Iter {i+1}, Max CFL = {CFL:.4f}")

    return u, u_all

def solve_linear_advection_1D(u0, T, a, dx, CFL, flux_limiter):
    """ Solve the linear advection equation
        \partial{u}/\partial{t} + a \partial{u}/\partial{x} = 0
    """

    dt = CFL * dx / a
    n_timesteps = int(T/dt)

    u_all = np.zeros((n_timesteps+1, u0.shape[0]))
    u_all[0] = u0.copy()

    u = u0.copy()

    for i in range(n_timesteps):
        f = a * u
        
        r = utils.compute_r(u, np.roll(u, 1), np.roll(u, -1)) # r_{i+1/2}

        F_low = 0.5 * (f + np.roll(f, -1)) - 0.5 * np.abs(a) * (np.roll(u, -1) - u)
        F_high = f + 0.5 * (1 - CFL) * (np.roll(f, -1) - f)

        phi = flux_limiter(r)
        F = (1 - phi) * F_low + phi * F_high 

        u -= dt/dx * (F - np.roll(F, 1))

        u_all[i+1] = u.copy() 

    return u, u_all

def solve_linear_advection_1D_torch(u0: torch.Tensor, T, a, dx, CFL, model):
    """ Solve the linear advection equation using torch instead of np for the purpose of back propagation
        \partial{u}/\partial{t} + a \partial{u}/\partial{x} = 0

        parameters:
        u0: 1 by N tensor, N is the number of cells
    """

    dt = CFL * dx / a
    n_timesteps = int(T/dt)

    # u_all = np.zeros((n_timesteps+1, u0.shape[0]))
    # u_all[0] = u0.copy()

    u = torch.clone(u0)

    for i in range(n_timesteps):
        f = a * u
        
        # r: 1d tensor
        r = utils.compute_r_torch(u, torch.roll(u, 1), torch.roll(u, -1)) # r_{i+1/2}

        F_low = 0.5 * (f + torch.roll(f, -1)) - 0.5 * abs(a) * (torch.roll(u, -1) - u)
        F_high = f + 0.5 * (1 - CFL) * (torch.roll(f, -1) - f)

        # phi: 2d tensor
        phi = model(r.view(-1, 1))
        # phi: 1d tensor
        phi = phi.squeeze()

        F = (1 - phi) * F_low + phi * F_high 

        u -= dt/dx * (F - torch.roll(F, 1))

        # u_all[i+1] = u.copy() 

    # return u, u_all
    return u