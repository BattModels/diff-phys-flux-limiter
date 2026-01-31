import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter
from utils import utils

def rp_euler(q, efix, gamma=1.4):
    gm1 = gamma - 1.

    # q: ndarray([num_eqn, num_cells]) (including ghost cells)
    # Conserved quantities:
    # 0 density             rho
    # 1 momentum          rho u
    # 2 energy              E  

    n_cells = q.shape[1]
    n_edges = n_cells - 1

    wave = np.zeros((3, 3, n_edges))
    s = np.zeros((3, n_edges))
    amdq = np.zeros((3, n_edges))
    apdq = np.zeros((3, n_edges))

    for i in range(n_edges):
        ql = q[:, i]
        qr = q[:, i+1]

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

        wave[:, 0, i] = alpha1*np.array([1., uhat - chat, hhat - uhat*chat])
        wave[:, 1, i] = alpha2*np.array([1., uhat, 0.5*uhat**2])
        wave[:, 2, i] = alpha3*np.array([1., uhat + chat, hhat + uhat*chat])

        # Wave speeds (eigenvalues)
        s1 = uhat - chat
        s2 = uhat
        s3 = uhat + chat
        s[:, i] = np.array([s1, s2, s3])

    if efix:
        for i in range(n_edges):
            ql = q[:, i]
            qr = q[:, i+1]

            # Check 1-wave
            # ------------
            rhoim1 = ql[0]
            pim1 = gm1 * (ql[2] - 0.5*ql[1]**2 / rhoim1)
            cim1 = np.sqrt(gamma*pim1/rhoim1)
            s0 = ql[1]/rhoim1 - cim1     # u-c in left state (cell i)

            # Check for fully supersonic case:
            if (s0 >= 0 and s[0, i] > 0):
                # everything is right-going
                amdq[:, i] = 0
                continue
            
            rho1 = ql[0] + wave[0, 0, i]
            rhou1 = ql[1] + wave[1, 0, i]
            en1 = ql[2] + wave[2, 0, i]
            p1 = gm1*(en1 - 0.5*rhou1**2/rho1)
            c1 = np.sqrt(gamma*p1/rho1)
            s1 = rhou1/rho1 - c1       # u-c to right of 1-wave
            if (s0 < 0 and s1 > 0):
                # transonic rarefaction in the 1-wave
                sfract = s0 * (s1-s[0, i]) / (s1-s0)
            elif (s[0, i] < 0):
                # 1-wave is leftgoing
                sfract = s[0, i]
            else:
                # 1-wave is rightgoing
                sfract = 0         # this shouldn't happen since s0 < 0
            amdq[:, i] = sfract*wave[:, 0, i]

            # check contact discontinuity:
            #------------------------------
            if (s[1, i] >= 0):
                # 2-waves are rightgoing
                continue
            amdq[:, i] += s[1, i]*wave[:, 1, i]

            # Check 3-wave
            # ------------
            rhoi = qr[0]
            pi = gm1 * (qr[2] - 0.5*qr[1]**2 / rhoi)
            ci = np.sqrt(gamma*pi/rhoi)
            s3 = qr[1]/rhoi + ci      # u+c in right state  (cell i+1)
            rho2 = qr[0] - wave[0, 2, i]
            rhou2 = qr[1] - wave[1, 2, i]
            en2 = qr[2] - wave[2, 2, i]
            p2 = gm1*(en2 - 0.5*rhou2**2/rho2)
            c2 = np.sqrt(gamma*p2/rho2)
            s2 = rhou2/rho2 + c2       # u+c to left of 3-wave
            if (s2 < 0 and s3 > 0):
                # transonic rarefaction in the 3-wave
                sfract = s2 * (s3-s[2, i]) / (s3-s2)
            elif (s[2, i] < 0):
                # 3-wave is leftgoing
                sfract = s[2, i]
            else:
                # 3-wave is rightgoing
                continue
            amdq[:, i] += sfract*wave[:, 2, i]

        # compute the rightgoing flux differences:
        # df = SUM s*wave   is the total flux difference and apdq = df - amdq
        df = np.zeros((3, n_edges))
        for i in range(n_edges):
            for mw in range(3):
                df[:, i] += s[mw, i]*wave[:, mw, i]
        
        apdq = df - amdq

    else: # No entropy fix
        for i in range(n_edges):
            for j in range(3):
                if s[j, i] < 0:
                    amdq[:, i] += s[j, i] * wave[:, j, i]
                else:
                    apdq[:, i] += s[j, i] * wave[:, j, i]

    return wave, s, amdq, apdq

def limit(num_eqn, wave, s, limit_func):
    # Adapted from 'limit' function in pyclaw/src/pyclaw/limiters/tvd.py
    r"""
    Apply a limiter to the waves

    Function that limits the given waves using the methods contained
    in limiter.  This is the vectorized version of the function acting on a 
    row of waves at a time.
    
    :Input:
     - *wave* - (ndarray(num_eqn,num_waves,:)) The waves at each interface
     - *s* - (ndarray(num_waves,:)) Speeds for each wave
     - *limit_func* - flux limiter function
        
    :Output:
     - (ndarray(num_eqn,num_waves, :)) - Returns the limited waves

    """
    
    # wave_norm2 is the sum of the squares along the num_eqn axis,
    # so the norm of the interface (edge) i for wave number j is addressed 
    # as wave_norm2[j, i]
    wave_norm2 = np.sum(np.square(wave),axis=0)
    wave_zero_mask = np.array((wave_norm2 == 0), dtype=float)
    wave_nonzero_mask = (1.0-wave_zero_mask)

    # dotls contains the products of adjacent cell values summed
    # along the num_eqn axis.  For reference, dotls[:, 0] is the dot
    # product of the 0 edge and the 1 edge.
    dotls = np.sum(wave[:,:,1:]*wave[:,:,:-1], axis=0)

    # array containing ones where s > 0, zeros elsewhere
    spos = np.array(s > 0.0, dtype=float)[:,1:-1]

    # Here we construct a masked array, then fill the empty values with 0,
    # this is done in case wave_norm2 is 0 or close to it
    # Take upwind dot product
    r = np.ma.array((spos*dotls[:,:-1] + (1-spos)*dotls[:,1:]))
    # Divide it by the norm**2
    r /= np.ma.array(wave_norm2[:,1:-1])
    # Fill the rest of the array
    r.fill_value = 0
    r = r.filled()
    
    for mw in range(wave.shape[1]):
        wlimitr = limit_func(r[mw,:])
        for m in range(num_eqn):
            wave[m,mw,1:-1] = wave[m,mw,1:-1]*wave_zero_mask[mw,1:-1] \
                + wlimitr * wave[m,mw,1:-1] * wave_nonzero_mask[mw,1:-1]

    return wave

def flux(q, dtdx, limit_func, efix):
    nx_cells = q.shape[1]
    nx_edges = nx_cells - 1

    # Initialization
    qadd = np.zeros((3, nx_cells))
    fadd = np.zeros((3, nx_edges))

    # Solve Riemann problem at each interface and compute Godunov updates
    wave, s, amdq, apdq = rp_euler(q=q, efix=efix)

    # Check CFL
    CFL = dtdx*np.max(np.abs(s))
    if CFL > 1:
        raise ValueError(f"CFL={CFL} > 1.")

    # Set qadd for the donor-cell upwind method (Godunov)
    qadd[:, :-1] -= dtdx*amdq
    qadd[:, 1:] -= dtdx*apdq

    # modify F fluxes for second order q_{xx} correction terms
    # Limit waves
    wave = limit(num_eqn=3, wave=wave, s=s, limit_func=limit_func)

    # Compute second-order corrections
    cqxx = np.zeros((3, nx_edges))
    for i in range(nx_edges):
        for mw in range(3):
            cqxx[:, i] += np.abs(s[mw, i]) * (1-dtdx*np.abs(s[mw, i])) * wave[:, mw, i]
        fadd[:, i] += 0.5*cqxx[:, i]

    return qadd, fadd, CFL

def create_initial_states(n_cells, gamma=1.4):
    """
    Expansion (rarefaction) from a left state into vacuum.
    Standard setup (Munz, 1994):
      Domain: x in [-0.3, 0.7], gamma = 1.4
      Left  (rho,u,p) = (1.0,  0.0, 2.5)
      Right (rho,u,p) = (1e-8, 0.0, 1e-8)

    Returns
    -------
    x : (n_cells,) cell-centered mesh
    q : (3, n_cells) conserved variables [rho, rho*u, E]
    """   
    x_ini, x_fin = -0.3, 0.7
    dx = (x_fin - x_ini) / n_cells
    x = (np.arange(n_cells) + 0.5) * dx + x_ini

    rho0 = np.empty(n_cells)
    u0   = np.empty(n_cells)
    p0   = np.empty(n_cells)

    rho0[x < 0] = 1.0
    u0[x < 0]   = 0.0
    p0[x < 0]   = 2.5
    rho0[x >= 0] = 1e-10
    u0[x >= 0]   = 0.0
    p0[x >= 0]   = 1e-10

    # conserved
    E0 = p0 / (gamma - 1.0) + 0.5 * rho0 * u0**2
    q  = np.array([rho0, rho0 * u0, E0])

    return x, q

def solve_rarefaction(mesh, q, dt_initial, dx, T, CFL, limit_func, efix, gamma=1.4):
    n_cells = q.shape[1]   
    n_edges = n_cells - 1

    qold = q
    
    t = 0.0
    dt = dt_initial
    while t < T:

        dtdx = dt/dx

        qnew = qold.copy()
        maxCFL = 0.

        qadd, fadd, CFL1d = flux(qnew, dtdx, limit_func=limit_func, efix=efix)
        qnew[:, 2:-2] += qadd[:, 2:-2] - dtdx * (fadd[:, 2:-1] - fadd[:, 1:-2])
        maxCFL = np.max(np.array([maxCFL, CFL1d]))


        # Apply zero-order extrapolation BCs
        # Left
        qnew[:, 0] = qnew[:, 2]
        qnew[:, 1] = qnew[:, 2]
        # Right
        qnew[:, -1] = qnew[:, -3]
        qnew[:, -2] = qnew[:, -3]
        del qold
        qold = qnew

        # Update t and dt
        t += dt
        dt = CFL/maxCFL*dt
        dt = np.min(np.array([dt, T-t]))
        # print(f"t={t: .4f}, max CFL={maxCFL}")

    rho = qnew[0, :]
    u = qnew[1, :] / rho
    E = qnew[2, :]
    p = (1.4 - 1.0) * (E - 0.5 * rho * u**2)

    return mesh, rho, u, p

import numpy as np

def euler_rarefaction_into_vacuum(
    x, t,
    rho_L=1.0, u_L=0.0, p_L=1.0,
    gamma=1.4,
    return_primitives=False
):
    """
    Analytical solution: expansion (rarefaction) from a left state into vacuum.
    Ref: A tracking method for gas flow into vacuum based on the vacuum Riemann problem (Munz, 1994)

    Piecewise in xi = x/t:
      (rho, m, E) = (rho_L, rho_L*u_L, E_L)                       for xi < u_L - c_L
                   (rho0(xi), m0(xi), E0(xi))                     for u_L - c_L < xi < u_L + 2 c_L/(gamma-1)
                   (0, 0, 0)                                      otherwise

    where:
      c_L = sqrt(gamma p_L / rho_L)
      u0(xi) = [ (gamma-1) u_L + 2 (xi + c_L) ] / (gamma+1)
      c0(xi) = u0 - xi
      K = p_L / rho_L^gamma
      rho0 = ( c0^2 / (gamma K) )^(1/(gamma-1))
      p0 = K rho0^gamma
      m0 = rho0 u0
      E0 = p0/(gamma-1) + 0.5 rho0 u0^2
    """
    x = np.asarray(x, dtype=float)

    if t <= 0.0:
        raise ValueError("Analytical self-similar solution requires t > 0.")

    # Left state sound speed
    cL = np.sqrt(gamma * p_L / rho_L)

    # Similarity variable
    xi = x / t

    # Region bounds
    xi_head = u_L - cL
    xi_tail = u_L + 2.0 * cL / (gamma - 1.0)

    # Allocate
    rho = np.zeros_like(xi)
    u   = np.zeros_like(xi)
    p   = np.zeros_like(xi)

    # 1) Left constant state
    mask_L = (xi < xi_head)
    rho[mask_L] = rho_L
    u[mask_L]   = u_L
    p[mask_L]   = p_L

    # 2) Rarefaction fan
    mask_fan = (xi >= xi_head) & (xi <= xi_tail)
    if np.any(mask_fan):
        xi_f = xi[mask_fan]

        # u0(xi) from your screenshot
        u0 = ((gamma - 1.0) * u_L + 2.0 * (xi_f + cL)) / (gamma + 1.0)

        # From characteristic u - c = xi  =>  c0 = u0 - xi
        c0 = u0 - xi_f

        # Isentropic relation p = K rho^gamma, with K = p_L / rho_L^gamma
        K = p_L / (rho_L**gamma)

        # rho0 = (c0^2/(gamma K))^(1/(gamma-1))
        # (equivalently: ((c0^2 * rho_L^gamma)/(gamma p_L))^(1/(gamma-1)))
        rho0 = (c0**2 / (gamma * K))**(1.0 / (gamma - 1.0))
        p0   = K * rho0**gamma

        rho[mask_fan] = rho0
        u[mask_fan]   = u0
        p[mask_fan]   = p0

    # 3) Vacuum region is already zeros (xi > xi_tail)

    # Conserved variables
    m = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    q = np.vstack([rho, m, E])

    if return_primitives:
        return q, rho, u, p
    return q


if __name__ == "__main__":
    n_cells = 200
    dt_initial = 0.0001
    T = 0.1
    CFL = 0.4
    efix = True

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
    markers = ['s', 'x', '+', 'o', '^', '.', '*', 'p']
    colors = ['pink', 'brown', 'red', 'purple', 'green', 'blue', 'orange', 'cyan']

    # Create initial states
    mesh, initial_q = create_initial_states(n_cells=n_cells, gamma=1.4)

    # Analytical solution
    _, rho_exact, u_exact, p_exact = euler_rarefaction_into_vacuum(
        x=mesh,
        t=T,
        rho_L=1.0, u_L=0.0, p_L=2.5,
        gamma=1.4,
        return_primitives=True
    )
    # Remove ghost cells
    x = mesh[2:-2]
    rho_exact = rho_exact[2:-2]
    u_exact = u_exact[2:-2]
    p_exact = p_exact[2:-2]

    # Plot to compare solutions
    fig,axes = plt.subplots(nrows=1, ncols=3)
    axes[0].plot(x, rho_exact, 'k-', label='Exact', clip_on=False)
    axes[1].plot(x, u_exact,   'k-', label='Exact', clip_on=False)
    axes[2].plot(x, p_exact,   'k-', label='Exact', clip_on=False)
    axes[0].set_ylabel('$rho$',fontsize=16)
    axes[1].set_ylabel('$U$',  fontsize=16)
    axes[2].set_ylabel('$P$',  fontsize=16)
    for i, (name, flux_limiter) in enumerate(flux_limiters.items()):
        # Numerical solutions
        print(f'Computing numerical solution using {name}...')
        mesh, rho, u, p = solve_rarefaction(mesh=mesh, q=initial_q, dt_initial=dt_initial, dx=mesh[1]-mesh[0], T=T, CFL=CFL, limit_func=flux_limiter, efix=efix)
        # Remove ghost cells
        rho = rho[2:-2]
        u = u[2:-2]
        p = p[2:-2]
        # axes[0].scatter(x, rho, s=8, marker=markers[i], c=colors[i], label=name, clip_on=False)
        # axes[1].scatter(x, u,   s=8, marker=markers[i], c=colors[i], label=name, clip_on=False)
        # axes[2].scatter(x, p,   s=8, marker=markers[i], c=colors[i], label=name, clip_on=False)
        axes[0].plot(x, rho, marker=markers[i], markersize=4,  color=colors[i], label=name, clip_on=False)
        axes[1].plot(x, u,   marker=markers[i], markersize=4,  color=colors[i], label=name, clip_on=False)
        axes[2].plot(x, p,   marker=markers[i], markersize=4,  color=colors[i], label=name, clip_on=False)
        print(f"MSE of density:  {np.mean((rho - rho_exact)**2)}")
        print(f"MSE of velocity: {np.mean((u - u_exact)**2)}")
        print(f"MSE of pressure: {np.mean((p - p_exact)**2)}")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    
    fig.set_size_inches(30,10)
    fig.savefig('aaaaaaRarefaction_solution.png', dpi=300)