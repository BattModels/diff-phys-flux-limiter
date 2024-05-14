import torch
import numpy as np
from scipy.optimize import bisect

from utils import utils

def analytic_flux_burgers(u):
    return u**2/2

def solve_burgers_1D(u0, T, dx, CFL, flux_limiter, nu=0.01):
    u = u0.copy()

    dt = CFL * np.max(np.abs(u)) * dx
    dt = np.minimum(dt, T-t)
    
    eps = np.random.rand(1)*1e-16

    t = 0
    while t < T:

        # Define sonic point
        u_bar = 0

        u_plus = np.maximum(u,u_bar)
        u_minus = np.minimum(u,u_bar)
        f_plus = analytic_flux_burgers(u_plus)
        f_minus = analytic_flux_burgers(u_minus)

        f_eo = f_plus + np.roll(f_minus, -1)
        delta_f_plus = np.roll(f_plus, -1) - f_plus
        delta_f_minus = np.roll(f_minus, -1) - f_minus

        delta_u = np.roll(u, -1) - u
        div_zero_idx = (delta_u == 0)
        delta_f_plus[div_zero_idx] = u_plus[div_zero_idx]
        delta_f_minus[div_zero_idx] = u_minus[div_zero_idx]
        delta_u[div_zero_idx] = 1
        CFL_plus = dt/dx * delta_f_plus / delta_u
        CFL_minus = dt/dx * delta_f_minus / delta_u

        alpha_plus = 0.5 * (1 - CFL_plus)
        alpha_minus = 0.5 * (1 + CFL_minus)
        r_plus = (np.roll(alpha_plus, 1) * np.roll(delta_f_plus, 1)) / (alpha_plus * delta_f_plus + eps)
        r_minus = (alpha_minus * delta_f_minus) / (np.roll(alpha_minus, 1) * np.roll(delta_f_minus, 1) + eps)
        phi_plus = flux_limiter(r_plus)
        phi_minus = flux_limiter(r_minus)

        u -= dt/dx * ((f_eo - np.roll(f_eo, 1)) \
                    + (phi_plus * alpha_plus * delta_f_plus \
                    - np.roll(phi_plus, 1) * np.roll(alpha_plus, 1) * np.roll(delta_f_plus, 1)) \
                    + (phi_minus * np.roll(alpha_minus, 1) * np.roll(delta_f_minus, 1) \
                    - np.roll(phi_minus, -1) * alpha_minus * delta_f_minus))
        
        t += dt

    return u

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

        # FOU
        F_low = 0.5 * (f + np.roll(f, -1)) - 0.5 * np.abs(a) * (np.roll(u, -1) - u)

        # Lax-Friedrichs
        # F_low = 0.5 * (f + np.roll(f, -1)) - 0.5 * dx/dt * (np.roll(u, -1) - u)

        # Lax-Wendroff
        F_high = f + 0.5 * (1 - CFL) * (np.roll(f, -1) - f)

        phi = flux_limiter(r)
        F = (1 - phi) * F_low + phi * F_high 

        u -= dt/dx * (F - np.roll(F, 1))

        u_all[i+1] = u.copy() 

    return u, u_all

def solve_linear_advection_1D_torch(u0: torch.Tensor, T, a, dx, CFL, model):
    """ Solve the linear advection equation using torch instead of numpy for the purpose of back propagation
        \partial{u}/\partial{t} + a \partial{u}/\partial{x} = 0

        parameters:
        u0: 1 by N tensor, N is the number of cells
    """

    dt = CFL * dx / a
    n_timesteps = int(T/dt)

    # Need to fix when u0 is 1d tensor
    u_all = torch.zeros((u0.shape[0], n_timesteps, u0.shape[1]))
    # r_all = torch.zeros(u_all.shape)

    u = torch.clone(u0)

    for i in range(n_timesteps):
        f = a * u
        
        # Roll along the last dimension of the tensor
        dim = u.dim() - 1

        r = utils.compute_r_torch(u, torch.roll(u, 1, dim), torch.roll(u, -1, dim)) # r_{i+1/2}

        F_low = 0.5 * (f + torch.roll(f, -1, dim)) - 0.5 * abs(a) * (torch.roll(u, -1, dim) - u)
        F_high = f + 0.5 * (1 - CFL) * (torch.roll(f, -1, dim) - f)

        phi = model(r.view(-1, 1))
        phi = phi.view(u.shape)

        F = (1 - phi) * F_low + phi * F_high 

        u -= dt/dx * (F - torch.roll(F, 1, dim))

        u_all[:,i,:] = torch.clone(u)
        # r_all[:,i,:] = torch.clone(r).detach()

    # return u, u_all
    # return u_all, r_all
    return u_all

def construct_char_eqn(x, t):
    def char_eqn(x0):
        return x0 + np.sin(2*np.pi*x0)*t - x
    return char_eqn

def solve_burgers_1D_exactly(Nx=1000, T_end=1., dt=1e-3):
    # Only compute the solution of the first half of the interval [0, 1] because of symmetry
    L_half = 1./2
    Nx_half = int(Nx/2)
    dx = L_half/Nx_half

    # Define time vector T
    Nt = int(T_end/dt)
    # Index 0 in T is used for storing the initial condition
    T = np.arange(Nt+1)*dt

    # Define cell centers
    X = np.linspace(dx/2, L_half-dx/2, Nx_half)

    # Initial condition, which located at the centers of the cells
    u0 = np.sin(2*np.pi*X)

    # Initialize the solution matrix to store the solutions for interval [0, L/2] at different time instances
    U = np.zeros((Nt+1, Nx_half))
    U[0] = u0

    # Main loop
    for it in range(Nt):
        t = T[it+1]
        for ix in range(Nx_half):
            x = X[ix]
            char_curve_eqn = construct_char_eqn(x, t)
            # Because of the upwind nature of the advection (u is positive in the first half of the interval),
            # the value would come from the left of x, thus we can set the upper bound of [a, b] to b = x.
            root = bisect(char_curve_eqn, a=0, b=x)
            U[it+1, ix] = np.sin(2*np.pi*root)

    # Complete the other half of the finite volume mesh vector and the solution matrix
    if Nx % 2 == 0:
        X = np.concatenate((X, X+0.5))
        U = np.concatenate((U, -np.fliplr(U)), axis=1)
    else:
        X = np.concatenate((X, np.array([0.5]), X+0.5))
        U = np.concatenate((U, np.zeros((Nt+1, 1)), -np.fliplr(U)), axis=1)

    return T, X, U
