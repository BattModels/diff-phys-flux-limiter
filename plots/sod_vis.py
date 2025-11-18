import torch
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scienceplots

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter
from utils import utils

plt.style.use('science')
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":10,
    })

def flux_func(q, gamma=1.4):
    # Primitive variables
    r = q[0]
    u = q[1]/r
    E = q[2]
    p = (gamma - 1.)*(E - 0.5*r*u**2)
    
    # Flux vector
    f0 = r*u
    f1 = r*u**2+p
    f2 = u*(E+p)
    flux = np.array([f0, f1, f2])
    
    return flux

def roe_flux(ql, qr, gamma=1.4):
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
    R = np.sqrt(rr/rl)
    uhat = (ul + R*ur) / (1+R)
    hhat = (hl + R*hr) / (1+R)
    chat = np.sqrt((gamma - 1.)*(hhat - 0.5*uhat**2))

    # Right Eigenvectors
    r1 = np.array([1, uhat - chat, hhat - uhat*chat])
    r2 = np.array([1, uhat, 0.5*uhat**2])
    r3 = np.array([1, uhat + chat, hhat + uhat*chat])
    r = np.array([r1, r2, r3])
    
    # Auxiliary variables to compute left eigenvectors l_i (no relation with alpha_i below)
    alpha = (gamma - 1.) * uhat**2 / (2*chat**2)
    beta = (gamma - 1.) / (chat**2)

    # Left eigenvectors
    l1 = np.array([0.5 * (alpha + uhat/chat), -0.5 * (beta*uhat + 1/chat), 0.5 * beta])
    l2 = np.array([1-alpha,                   beta*uhat,                   -beta])
    l3 = np.array([0.5 * (alpha - uhat/chat), -0.5 * (beta*uhat - 1/chat), 0.5 * beta])

    # Compute wave coefficients
    dq = qr - ql
    alpha1 = np.dot(dq, l1)
    alpha2 = np.dot(dq, l2)
    alpha3 = np.dot(dq, l3)
    wave_coefs = np.array([alpha1, alpha2, alpha3])

    # Wave speeds (eigenvalues)
    s1 = uhat - chat
    s2 = uhat
    s3 = uhat + chat
    s = np.array([s1, s2, s3])

    fl = flux_func(ql)
    fr = flux_func(qr)
    
    # Roe flux
    F = 0.5*(fl + fr) - 0.5*(abs(s1)*alpha1*r1 + abs(s2)*alpha2*r2 + abs(s3)*alpha3*r3)
    
    return F, s, wave_coefs, r

def create_initial_states_arr(config_name, n_cells, gamma=1.4):
    if config_name == 'Sod' or config_name == 'Lax':
        if config_name == 'Sod':
            left_state = (1., 0., 1.)
            right_state = (0.125, 0., 0.1)
        else: # Lax problem
            left_state = (0.445, 0.698, 3.528)
            right_state = (0.5, 0., 0.571)

        x_ini = 0.
        x_fin = 1.
        dx = (x_fin - x_ini) / n_cells            # Cell size
        x = (np.arange(n_cells)+0.5)*dx + x_ini   # Mesh

        r0 = np.where(x < 0.5*(x_ini + x_fin), left_state[0]*np.ones(n_cells), right_state[0]*np.ones(n_cells))        # Density
        u0 = np.where(x < 0.5*(x_ini + x_fin), left_state[1]*np.ones(n_cells), right_state[1]*np.ones(n_cells))        # Velocity
        p0 = np.where(x < 0.5*(x_ini + x_fin), left_state[2]*np.ones(n_cells), right_state[2]*np.ones(n_cells))        # Pressure

    elif config_name == 'Shu-Osher':
        x_ini = -5.
        x_fin = 5.
        dx = (x_fin - x_ini) / n_cells            # Cell size
        x = (np.arange(n_cells)+0.5)*dx + x_ini   # Mesh
    
        eps = 0.2
        r0 = np.where(x < -4., 3.857143*np.ones(n_cells), 1 + eps*np.sin(5*x))        # Density
        u0 = np.where(x < -4., 2.629369*np.ones(n_cells), np.zeros(n_cells))          # Velocity
        p0 = np.where(x < -4., 10.33333*np.ones(n_cells), np.ones(n_cells))           # Pressure

    else:
        raise ValueError(f"{config_name} not implemented.")

    E0 = p0/(gamma-1.) + 0.5*r0*u0**2      # Energy per unit volume
    q  = np.array([r0, r0*u0, E0])         # Vector of conserved variables

    return x, q


def solve_euler_1d(mesh, initial_q, flux_limiter, t_end=0.2, CFL=0.4, gamma=1.4):
    # Parameters
    dx = mesh[1] - mesh[0]                 # Cell size
    n_edges = mesh.size+1               # Number of edges (including two boundaries)

    q = initial_q.copy()
    # Solver loop
    t = 0
    while t < t_end:
        # Initialize flux residuals and wave speeds
        residuals = np.zeros_like(q)
        wave_speeds = np.zeros((3, n_edges))

        # Variables stored for flux limiter (tall matrices instead of fat matrices!!!)
        alpha_all = np.zeros((n_edges, 3))     # Coefficients of wave families
        r_all = np.zeros((n_edges, 3, 3))      # Wave families

        # Loop over interior cell interfaces
        for iEdge in range(1, n_edges-1):
            # Convert the edge index to cell indices
            iL = iEdge-1
            iR = iEdge
            F, s, alpha, r = roe_flux(q[:, iL], q[:, iR])
            residuals[:, iL] += F
            residuals[:, iR] -= F
            wave_speeds[:,iEdge] = s

            alpha_all[iEdge] = alpha
            r_all[iEdge] = r
        
        dt = np.min(np.array([CFL*dx/(np.max(np.abs(wave_speeds))), t_end-t]))

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
                    theta = np.dot(alpha_l*rl, r)/np.dot(r,r) / (alpha + 1e-8)
                else:
                    alpha_r = alpha_all[iEdge+1][iWave]
                    rr = r_all[iEdge+1][iWave]
                    theta = np.dot(alpha_r*rr, r)/np.dot(r,r) / (alpha + 1e-8)
                F_correction = 0.5*np.abs(wave_speed)*(1-dt/dx*np.abs(wave_speed))*alpha*flux_limiter(theta)*r
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
        if min(p)<0: print ('negative pressure found!')

    return mesh, rho, u, p

def shock_tube_func(p4, p1, p5, rho1, rho5, u1, u5, gamma):
    # Only solve the case where the solution is composed by:
    # left: expansion fan
    # mid: contact discontinuity
    # right: shock
    c1 = np.sqrt(gamma*p1/rho1)
    c5 = np.sqrt(gamma*p5/rho5)

    gm1 = gamma - 1
    gp1 = gamma + 1
    g2 = 2. * gamma

    f_expansion_fan = 2*c1/gm1 * ((p4/p1)**(gm1/g2)-1)
    f_shock = (p4 - p5) / (rho5*c5 * np.sqrt(gp1/g2 * p4/p5 + gm1/g2))

    u3 = u1 - f_expansion_fan
    u4 = u5 + f_shock

    return u4 - u3

def analytic_sod(left_state, right_state, npts=1000, t=0.2, L=1., gamma=1.4):
    # Define some auxiliary variables
    gm1 = gamma - 1
    gp1 = gamma + 1
    g2 = 2. * gamma

    # Define positions where we need to solve for the states
    # x_arr = np.linspace(0, L, npts)

    # This is another way to define the same mesh as that 
    # used in numerical solution so that we can compare the MSE loss
    dx = L / npts                      # Cell size
    x_arr = (np.arange(npts)+0.5)*dx   # Mesh

    rho1, u1, p1 = left_state
    rho5, u5, p5 = right_state

    p4 = fsolve(shock_tube_func, 0.5, (p1, p5, rho1, rho5, u1, u5, gamma))[0]

    # Compute post-shock density and velocity
    c5 = np.sqrt(gamma*p5/rho5)
    
    f_shock = (p4 - p5) / (rho5*c5 * np.sqrt(gp1/g2 * p4/p5 + gm1/g2))
    u4 = u5 + f_shock
    s = u5 - (p4 - p5) / (rho5*(u5 - u4))           # Shock speed
    rho4 = rho5 * (u5 - s) / (u4 - s)
    
    # Compute values at foot of the rarefaction
    p3 = p4
    u3 = u4
    rho3 = rho1 * (p3 / p1) ** (1. / gamma)         # Use the isentropic relation

    # Compute the position of each structure
    c1 = np.sqrt(gamma*p1/rho1)
    c3 = np.sqrt(gamma*p3/rho3)
    xi = L / 2                                      # Initial position of the barrier
    xsh = xi + s * t                                # Shock
    xcd = xi + u3 * t                               # Contact discontinuity
    xft = xi + (u3 - c3) * t                        # Foot of rarefaction
    xhd = xi + (u1 - c1) * t                        # Head of rarefaction

    # Compute the states at each points
    rho = np.zeros(npts, dtype=float)
    u = np.zeros(npts, dtype=float)
    p = np.zeros(npts, dtype=float)
    for i, x in enumerate(x_arr):
        if x < xhd:
            rho[i] = rho1
            u[i] = u1
            p[i] = p1
        elif x < xft:
            c = gm1/gp1 * (u1 - (x - xi) / t) + 2*c1/gp1
            u[i] = c + (x - xi) / t
            p[i] = p1 * (c/c1)**(g2/gm1)
            rho[i] = gamma * p[i] / c**2
        elif x < xcd:
            rho[i] = rho3
            u[i] = u3
            p[i] = p3
        elif x < xsh:
            rho[i] = rho4
            u[i] = u4
            p[i] = p4
        else:
            rho[i] = rho5
            u[i] = u5
            p[i] = p5
    
    return x_arr, rho, u, p

def main(config_name, n_cells):
    device = 'cpu'

    # Load Model
    model = FluxLimiter(1,1,64,5,act="relu") #
    model.load_state_dict(torch.load("model_linear_relu.pt", map_location=device))
    model = model.to(device)

    def neural_flux_limiter_linear(r):
        model.eval()
        with torch.no_grad():
            phi = model(torch.Tensor([r]).view(-1, 1).to(device))
        return phi.numpy().squeeze()
    
    model_euler = FluxLimiter(1,1,64,5,act="tanh") #
    model_euler.load_state_dict(torch.load("model_euler.pt"))
    model_euler = model_euler.to(device)
    
    def neural_flux_limiter_euler(r):
        model_euler.eval()
        with torch.no_grad():
            phi = model_euler(torch.Tensor([r]).view(-1, 1).to(device))
        return phi.numpy().squeeze()

    flux_limiters = {
        # "Upwind": utils.FOU,
        # "Lax-Wendroff": utils.LaxWendroff,
        # "Minmod": utils.minmod,
        # "Superbee": utils.superbee,
        # "Van Leer": utils.vanLeer,
        # "Koren": utils.koren,
        # "MC": utils.MC,
        "DPFL": neural_flux_limiter_linear,
        # "Neural network (euler)": neural_flux_limiter_euler,
        # "Piecewise linear": utils.piecewise_linear_limiter,
    }
    # markers = ['s', 'x', '+', 'o', '^', '.', '*', 'p']
    # colors = ['pink', 'brown', 'red', 'purple', 'green', 'blue', 'orange', 'cyan']

    if config_name == 'Sod':
        t = 0.2
    elif config_name == 'Lax':
        t = 0.13
    elif config_name == 'Shu-Osher':
        t = 1.8

    mesh, initial_q = create_initial_states_arr(config_name=config_name, n_cells=n_cells)

    # Analytic solution (or numerical solution using very fine mesh)
    print('Computing the reference solution...')
    if config_name == 'Sod': # Exact Riemann solver
        x_a, rho_a, u_a, p_a = analytic_sod(left_state=(1., 0., 1.), 
                                            right_state=(0.125, 0., 0.1), 
                                            npts=n_cells,
                                            t=t,
                                            )
    else:
        mesh_a, initial_q_a = create_initial_states_arr(config_name=config_name, n_cells=909)
        x_a, rho_a, u_a, p_a = solve_euler_1d(mesh=mesh_a,
                                              initial_q=initial_q_a,
                                              flux_limiter=utils.vanLeer, 
                                              t_end=t, 
                                              CFL=0.4,
                                              gamma=1.4,
                                              )
    
    # Plot to compare solutions
    fig,axes = plt.subplots(nrows=1, ncols=3)
    axes[0].plot(x_a, rho_a, 'k-', label='Exact', clip_on=False)
    axes[1].plot(x_a, u_a,   'k-', label='Exact', clip_on=False)
    axes[2].plot(x_a, p_a,   'k-', label='Exact', clip_on=False)
    axes[0].set_ylabel(r'$\rho$')
    axes[1].set_ylabel('$u$')
    axes[2].set_ylabel('$p$')
    for i, (name, flux_limiter) in enumerate(flux_limiters.items()):
        # Numerical solutions
        print(f'Computing numerical solution using {name}...')
        x, rho, u, p = solve_euler_1d(mesh=mesh,
                                      initial_q=initial_q,
                                      flux_limiter=flux_limiter, 
                                      t_end=t, 
                                      CFL=0.4,
                                      gamma=1.4,
                                      )
        axes[0].plot(x, rho, label=name, clip_on=False, linestyle='--')
        axes[1].plot(x, u, label=name, clip_on=False, linestyle='--')
        axes[2].plot(x, p, label=name, clip_on=False, linestyle='--')
        if config_name == 'Sod':
            print(f"MSE of density:  {np.sum((rho - rho_a)**2)/rho.size}")
            print(f"MSE of velocity: {np.sum((u - u_a)**2)/u.size}")
            print(f"MSE of pressure: {np.sum((p - p_a)**2)/p.size}")
        else:
            print(f"MSE of density:  {np.sum((rho - rho_a[4::9])**2)/rho.size}")
            print(f"MSE of velocity: {np.sum((u - u_a[4::9])**2)/u.size}")
            print(f"MSE of pressure: {np.sum((p - p_a[4::9])**2)/p.size}")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    
    fig.set_size_inches(10,3.3)
    fig.savefig('figures/paper/sod.pdf', dpi=300)

if __name__ == "__main__":
    # main(config_name='Shu-Osher', n_cells=101)
    main(config_name='Sod', n_cells=100)

