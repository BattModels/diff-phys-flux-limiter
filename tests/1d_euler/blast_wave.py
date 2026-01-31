import torch
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter
from utils import utils

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

def create_initial_states_with_ghost_cells(n_cells, gamma=1.4):
    """
    Woodward–Colella blast wave.
    Domain: x in [0, 1], gamma=1.4
      (rho,u,p) = (1,0,1000) for x < 0.1
      (rho,u,p) = (1,0,0.01) for 0.1 <= x < 0.9
      (rho,u,p) = (1,0,100)  for x >= 0.9

    Returns
    -------
    x : (n_cells,) cell-centered mesh
    q : (3, n_cells) conserved variables [rho, rho*u, E]
    """
    x_ini, x_fin = 0.0, 1.0
    dx = (x_fin - x_ini) / n_cells
    x = (np.arange(n_cells) + 0.5) * dx + x_ini
    x = np.concatenate(([x[0]-dx], x, [x[-1]+dx]))  # add ghost cells

    # primitives
    rho0 = np.ones(n_cells+2)
    u0   = np.zeros(n_cells+2)

    p0 = np.empty(n_cells+2)
    p0[x < 0.1] = 1000.0
    p0[(x >= 0.1) & (x < 0.9)] = 0.01
    p0[x >= 0.9] = 100.0

    # conserved
    E0 = p0 / (gamma - 1.0) + 0.5 * rho0 * u0**2
    q  = np.array([rho0, rho0 * u0, E0])

    return x, q

def solve_euler_1d(mesh, initial_q, flux_limiter, t_end=0.038, CFL=0.4, gamma=1.4):
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
        # First populate the left and right ghost boundaries, which are likely to be used as upwind values by neighboring interfaces
        F, s, alpha, r = roe_flux(np.array([q[0,2], -q[1, 2], q[2,2]]), np.array([q[0,0], q[1, 0], q[2,0]]))
        alpha_all[0] = alpha
        r_all[0] = r
        F, s, alpha, r = roe_flux(np.array([q[0,-1], q[1, -1], q[2,-1]]), np.array([q[0,-3], -q[1, -3], q[2,-3]]))
        alpha_all[-1] = alpha
        r_all[-1] = r
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

        # # Enforce reflective BCs
        q[0,0] = q[0,1]
        q[1,0] = -q[1,1]
        q[2,0] = q[2,1]
        q[0,-1] = q[0,-2]
        q[1,-1] = -q[1,-2]
        q[2,-1] = q[2,-2]

        # Compute primary variables
        rho = q[0]
        u = q[1]/rho
        E = q[2]
        p = (gamma-1.)*(E-0.5*rho*u**2)
        if min(p)<0: print ('negative pressure found!')

    return mesh[1:-1], rho[1:-1], u[1:-1], p[1:-1] # remove ghost cells

def solve_blast_wave(n_cells, flux_limiter, t_end=0.038, CFL=0.4, gamma=1.4):
    # Create initial states
    mesh, initial_q = create_initial_states_blast_with_ghost_cells(n_cells=n_cells, gamma=gamma)

    # Solve using the FVM solver
    x, rho, u, p = solve_euler_1d(mesh=mesh,
                                  initial_q=initial_q,
                                  flux_limiter=flux_limiter, 
                                  t_end=t_end, 
                                  CFL=CFL,
                                  gamma=gamma,
                                  )
    return x, rho, u, p

def main(config_name, n_cells, t=0.038):
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

    flux_limiters = {
        "Upwind": utils.FOU,
        "Minmod": utils.minmod,
        "Superbee": utils.superbee,
        "Van Leer": utils.vanLeer,
        "Koren": utils.koren,
        "MC": utils.MC,
        "DPFL": neural_flux_limiter_linear,
    }
    markers = ['s', 'x', '+', 'o', '^', '.', '*']
    colors = ['pink', 'brown', 'red', 'purple', 'green', 'blue', 'orange']

    mesh, initial_q = create_initial_states_with_ghost_cells(n_cells=n_cells)

    # Reference solution: numerical solution using very fine mesh
    print('Computing the reference solution...')
    mesh_a, initial_q_a = create_initial_states_with_ghost_cells(n_cells=1000)
    x_a, rho_a, u_a, p_a = solve_euler_1d(mesh=mesh_a,
                                            initial_q=initial_q_a,
                                            flux_limiter=utils.vanLeer, 
                                            t_end=t, 
                                            CFL=0.4,
                                            gamma=1.4,
                                            )

    np.save(f'{config_name}_x.npy', x_a)
    np.save(f'{config_name}_rho.npy', rho_a)
    np.save(f'{config_name}_u.npy', u_a)
    np.save(f'{config_name}_p.npy', p_a)

    # Plot to compare solutions
    fig,axes = plt.subplots(nrows=1, ncols=3)
    axes[0].plot(x_a, rho_a, 'k-', label='Exact', clip_on=False)
    axes[1].plot(x_a, u_a,   'k-', label='Exact', clip_on=False)
    axes[2].plot(x_a, p_a,   'k-', label='Exact', clip_on=False)
    axes[0].set_ylabel('$rho$',fontsize=16)
    axes[1].set_ylabel('$U$',  fontsize=16)
    axes[2].set_ylabel('$P$',  fontsize=16)
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
        # axes[0].scatter(x, rho, s=8, marker=markers[i], c=colors[i], label=name, clip_on=False)
        # axes[1].scatter(x, u,   s=8, marker=markers[i], c=colors[i], label=name, clip_on=False)
        # axes[2].scatter(x, p,   s=8, marker=markers[i], c=colors[i], label=name, clip_on=False)
        axes[0].plot(x, rho, color=colors[i], label=name, clip_on=False)
        axes[1].plot(x, u,   color=colors[i], label=name, clip_on=False)
        axes[2].plot(x, p,   color=colors[i], label=name, clip_on=False)
        print(f"MSE of density:  {np.sum((rho - rho_a[2::5])**2)/rho.size}")
        print(f"MSE of velocity: {np.sum((u - u_a[2::5])**2)/u.size}")
        print(f"MSE of pressure: {np.sum((p - p_a[2::5])**2)/p.size}")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    
    fig.set_size_inches(30,10)
    fig.savefig(config_name, dpi=300)

if __name__ == "__main__":
    main(config_name='Blast', n_cells=200)

