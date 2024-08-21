import torch
import numpy as np
from scipy.optimize import bisect

from utils import utils

def analytic_flux_burgers(u):
    return u**2/2

def solve_burgers_1D(u0, T, dx, CFL, flux_limiter, nu=0.01):
    u = u0.copy()

    eps = np.random.rand(1)*1e-16

    t = 0
    while t < T:
        dt = CFL * dx / np.max(np.abs(u))
        dt = np.minimum(dt, T-t)
        
        # Define sonic point
        u_bar = 0

        u_plus = np.maximum(u,u_bar)
        u_minus = np.minimum(u,u_bar)
        f_plus = analytic_flux_burgers(u_plus)
        f_minus = analytic_flux_burgers(u_minus)

        F_eo = f_plus + np.roll(f_minus, -1)
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

        u -= dt/dx * ((F_eo - np.roll(F_eo, 1)) \
                    + (phi_plus * alpha_plus * delta_f_plus \
                    - np.roll(phi_plus, 1) * np.roll(alpha_plus, 1) * np.roll(delta_f_plus, 1)) \
                    + (phi_minus * np.roll(alpha_minus, 1) * np.roll(delta_f_minus, 1) \
                    - np.roll(phi_minus, -1) * alpha_minus * delta_f_minus))
        
        t += dt

    return u

def solve_burgers_1D_torch(u0, T, dx, CFL, model, nu=0.01):
    u = torch.clone(u0)

    # Roll along the last dimension of the tensor
    dim = u.dim() - 1

    # Define eps to avoid division by 0
    eps = torch.rand(1).item()*1e-16

    t = 0
    while t < T:
        dt = CFL * dx / torch.max(torch.abs(u))
        dt = torch.minimum(dt, torch.Tensor([T-t]))

        # Define sonic point
        u_bar = 0.

        # u_plus = torch.maximum(torch.clone(u),torch.Tensor([u_bar]))
        # u_minus = torch.minimum(torch.clone(u),torch.Tensor([u_bar]))
        u_plus = torch.maximum(u,torch.Tensor([u_bar]))
        u_minus = torch.minimum(u,torch.Tensor([u_bar]))
        f_plus = analytic_flux_burgers(u_plus)
        f_minus = analytic_flux_burgers(u_minus)

        F_eo = f_plus + torch.roll(f_minus, -1, dim)            # f_{i+1/2}

        # ==========================================================================================================
        # Flux-difference splitting
        # ==========================================================================================================
        delta_f_plus = torch.roll(f_plus, -1, dim) - f_plus
        delta_f_minus = torch.roll(f_minus, -1, dim) - f_minus
        delta_u = torch.roll(u, -1, dim) - u
        
        # (1) Inplace operations
        # div_zero_idx = (delta_u == 0)
        # delta_f_plus[div_zero_idx] = u_plus[div_zero_idx]
        # delta_f_minus[div_zero_idx] = u_minus[div_zero_idx]
        # delta_u[div_zero_idx] = 1
        # CFL_plus = dt/dx * delta_f_plus / delta_u
        # CFL_minus = dt/dx * delta_f_minus / delta_u

        # (2) Without inplace operations
        # Without 1e-8, there would be a division by 0 issue in backward pass
        CFL_plus = dt/dx * torch.where(delta_u == 0, u_plus, delta_f_plus / (delta_u + 1e-8)) 
        CFL_minus = dt/dx * torch.where(delta_u == 0, u_minus, delta_f_minus / (delta_u + 1e-8))

        alpha_plus = 0.5 * (1 - CFL_plus)
        alpha_minus = 0.5 * (1 + CFL_minus)
        r_plus = (torch.roll(alpha_plus, 1, dim) * torch.roll(delta_f_plus, 1, dim)) / (alpha_plus * delta_f_plus + eps)
        r_minus = (alpha_minus * delta_f_minus) / (torch.roll(alpha_minus, 1, dim) * torch.roll(delta_f_minus, 1, dim) + eps)
        phi_plus = model(r_plus.view(-1,1))
        phi_minus = model(r_minus.view(-1,1))
        phi_plus = phi_plus.view(u.shape)
        phi_minus = phi_minus.view(u.shape)

        # It is better not to use '-=' because it is likely to cause inplace modification error during backward pass
        u = u - dt/dx * ((F_eo - torch.roll(F_eo, 1, dim)) \
                    + (phi_plus * alpha_plus * delta_f_plus \
                    - torch.roll(phi_plus, 1, dim) * torch.roll(alpha_plus, 1, dim) * torch.roll(delta_f_plus, 1, dim)) \
                    + (phi_minus * torch.roll(alpha_minus, 1, dim) * torch.roll(delta_f_minus, 1, dim) \
                    - torch.roll(phi_minus, -1, dim) * alpha_minus * delta_f_minus))
        
        # ==========================================================================================================
        # Another implementation without using flux splitting (something is going wrong)
        # ==========================================================================================================
        # f = u**2/2
        # delta_f = torch.roll(f, -1, dim) - f
        # delta_u = torch.roll(u, -1, dim) - u
        # a = torch.where(delta_u == 0, u, delta_f / (delta_u))

        # r_pos = (u - torch.roll(u, 1, dim)) / (torch.roll(u, -1, dim) - u + eps)
        # r_neg = (torch.roll(u, -2, dim) - torch.roll(u, -1, dim)) / (torch.roll(u, -1, dim) - u + eps)

        # r = torch.where(a > 0, r_pos, r_neg)
        # phi = model(r.view(-1, 1))
        # phi = phi.view(u.shape)

        # F_correction = 0.5*torch.abs(a)*(1-dt/dx*torch.abs(a))*phi*delta_u
        # F = F_eo + F_correction
        # u = u - dt/dx * (torch.roll(F, -1, dim) - F)

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

@torch.compile
def solve_linear_advection_1D_torch(u0: torch.Tensor, T, a, dx, CFL, model):
    """ Solve the linear advection equation using torch instead of numpy for the purpose of back propagation
        \partial{u}/\partial{t} + a \partial{u}/\partial{x} = 0

        parameters:
        u0: 1 by N tensor, N is the number of cells
    """

    device = u0.device

    dt = CFL * dx / a
    n_timesteps = int(T/dt)

    # Need to fix when u0 is 1d tensor
    u_all = torch.zeros((u0.shape[0], n_timesteps, u0.shape[1])).to(device)

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

    return u_all

# @torch.compile
# def _solve_linear_advection_1D_torch(u0: torch.Tensor, T, a, dx, CFL, model):
#     """ Solve the linear advection equation using torch instead of numpy for the purpose of back propagation
#         \partial{u}/\partial{t} + a \partial{u}/\partial{x} = 0

#         parameters:
#         u0: 1 by N tensor, N is the number of cells
#     """

#     dt = CFL * dx / a
#     n_timesteps = int(T/dt)

#     u_all = []
#     u = torch.clone(u0)

#     for i in range(n_timesteps):
#         f = a * u

#         r = utils.compute_r_torch(u, torch.roll(u, 1), torch.roll(u, -1)) # r_{i+1/2}

#         F_low = 0.5 * (f + torch.roll(f, -1)) - 0.5 * abs(a) * (torch.roll(u, -1) - u)
#         F_high = f + 0.5 * (1 - CFL) * (torch.roll(f, -1) - f)

#         phi = model(r.view(-1, 1))
#         # phi = phi.view(u.shape)
#         phi = phi.squeeze()

#         F = (1 - phi) * F_low + phi * F_high 

#         u -= dt/dx * (F - torch.roll(F, 1))

#         u_all.append(torch.clone(u))

#     u_all = torch.stack(u_all)
    
#     return u_all

# solve_linear_advection_1D_torch = torch.vmap(_solve_linear_advection_1D_torch, in_dims=(0, None, None, None, None, None))

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

def flux_func_torch(q, gamma=1.4):
    # Primitive variables
    r = q[0]
    u = q[1]/r
    E = q[2]
    p = (gamma - 1.)*(E - 0.5*r*u**2)
    
    # Flux vector
    f0 = r*u
    f1 = r*u**2+p
    f2 = u*(E+p)
    flux = torch.tensor([f0, f1, f2], dtype=torch.float32)
    
    return flux

def roe_flux_torch(ql, qr, gamma=1.4):
    # Primitive variables
    rl = ql[0]
    ul = ql[1]/rl
    El = ql[2]
    pl = (gamma - 1.) * (El - 0.5*rl*ul**2)
    hl = (El + pl) / rl

    rr = qr[0]
    ur = qr[1]/rr
    Er = qr[2]
    pr = (gamma - 1.) * (Er - 0.5*rr*ur**2)
    hr = (Er + pr) / rr

    # Roe-averages
    R = torch.sqrt(rr/rl)
    uhat = (ul + R*ur) / (1+R)
    hhat = (hl + R*hr) / (1+R)
    chat = torch.sqrt((gamma - 1.)*(hhat - 0.5*uhat**2))

    # Right Eigenvectors
    r1 = torch.tensor([1, uhat - chat, hhat - uhat*chat], dtype=torch.float32)
    r2 = torch.tensor([1, uhat,        0.5*uhat**2],      dtype=torch.float32)
    r3 = torch.tensor([1, uhat + chat, hhat + uhat*chat], dtype=torch.float32)
    r = torch.stack((r1, r2, r3))
    
    # Auxiliary variables to compute left eigenvectors l_i (no relation with alpha_i below)
    alpha = (gamma - 1.) * uhat**2 / (2*chat**2)
    beta = (gamma - 1.) / (chat**2)

    # Left eigenvectors
    l1 = torch.tensor([0.5 * (alpha + uhat/chat), -0.5 * (beta*uhat + 1/chat), 0.5 * beta], dtype=torch.float32)
    l2 = torch.tensor([1-alpha,                   beta*uhat,                   -beta],      dtype=torch.float32)
    l3 = torch.tensor([0.5 * (alpha - uhat/chat), -0.5 * (beta*uhat - 1/chat), 0.5 * beta], dtype=torch.float32)

    # Compute wave coefficients
    dq = qr - ql
    alpha1 = torch.dot(dq, l1)
    alpha2 = torch.dot(dq, l2)
    alpha3 = torch.dot(dq, l3)
    wave_coefs = torch.tensor([alpha1, alpha2, alpha3], dtype=torch.float32)

    # Wave speeds (eigenvalues)
    s1 = uhat - chat
    s2 = uhat
    s3 = uhat + chat
    s = torch.tensor([s1, s2, s3], dtype=torch.float32)

    fl = flux_func_torch(ql)
    fr = flux_func_torch(qr)
    
    # Roe flux
    F = 0.5*(fl + fr) - 0.5*(torch.abs(s1)*alpha1*r1 + torch.abs(s2)*alpha2*r2 + torch.abs(s3)*alpha3*r3)
    
    return F, s, wave_coefs, r

def solve_euler_1d_torch(mesh, initial_q, flux_limiter, t_end=0.2, CFL=0.4, gamma=1.4):
    # # Parameters
    dx = mesh[1] - mesh[0]                 # Cell size
    n_edges = torch.numel(mesh)+1               # Number of edges (including two boundaries)

    q = torch.clone(initial_q)
    # Solver loop
    t = 0
    while t < t_end:
        # Initialize flux residuals and wave speeds
        residuals = torch.zeros_like(q)
        wave_speeds = torch.zeros((3, n_edges))

        # Variables stored for flux limiter (tall matrices instead of fat matrices!!!)
        alpha_all = torch.zeros((n_edges, 3))     # Coefficients of wave families
        r_all = torch.zeros((n_edges, 3, 3))      # Wave families

        # Loop over interior cell interfaces
        for iEdge in range(1, n_edges-1):
            # Convert the edge index to cell indices
            iL = iEdge-1
            iR = iEdge
            F, s, alpha, r = roe_flux_torch(q[:, iL].clone(), q[:, iR].clone())
            residuals[:, iL] += F
            residuals[:, iR] -= F
            wave_speeds[:,iEdge] = s

            alpha_all[iEdge] = alpha
            r_all[iEdge] = r
        
        dt = torch.min(torch.tensor([CFL*dx/(torch.max(torch.abs(wave_speeds))), t_end-t], dtype=torch.float32))

        # Apply flux limiter
        # First populate the left and right boundaries to enforce Dirichlet BCs
        alpha_all[0] = alpha_all[1]
        alpha_all[-1] = alpha_all[-2]
        r_all[0] = r_all[1]
        r_all[-1] = r_all[-2]
        for iEdge in range(1, n_edges-1):
            iL = iEdge-1
            iR = iEdge
            for iWave in range(3):
                wave_speed = wave_speeds[iWave, iEdge]
                alpha = alpha_all[iEdge][iWave]
                r = r_all[iEdge][iWave]
                if wave_speed > 0:
                    alpha_l = alpha_all[iEdge-1][iWave]
                    rl = r_all[iEdge-1][iWave]
                    theta = torch.dot(alpha_l*rl, r)/torch.dot(r,r) / (alpha + 1e-8)
                else:
                    alpha_r = alpha_all[iEdge+1][iWave]
                    rr = r_all[iEdge+1][iWave]
                    theta = torch.dot(alpha_r*rr, r)/torch.dot(r,r) / (alpha + 1e-8)
                F_correction = 0.5*torch.abs(wave_speed)*(1-dt/dx*torch.abs(wave_speed))*alpha*flux_limiter(torch.tensor([theta], dtype=torch.float32))*r
                residuals[:, iL] += F_correction
                residuals[:, iR] -= F_correction

        # Update states without the left and right boundary cells
        # to enforce Dirichlet BCs
        q[:, 1:-1] -= dt/dx*residuals[:, 1:-1]
        t += dt

        # Compute primary variables
        rho = q[0]
        u = q[1]/rho
        E = q[2]
        p = (gamma-1.)*(E-0.5*rho*u**2)
        if torch.min(p)<0: print ('negative pressure found!')

    return mesh, rho, u, p