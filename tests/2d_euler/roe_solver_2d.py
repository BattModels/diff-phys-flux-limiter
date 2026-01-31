import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter
from utils import utils

def rpn2_euler(ixy, q, efix, gamma=1.4):
    # The data is along a slice in the x-direction if ixy=1 or the y-direction if ixy=2
    if ixy == 1:
          mu = 1
          mv = 2
    else:
          mu = 2
          mv = 1

    gm1 = gamma - 1.

    # q: ndarray([num_eqn, num_cells]) (including ghost cells)
    # Conserved quantities:
    # 0 density             rho
    # 1 x_momentum          rho u
    # 2 y_momentum          rho v
    # 3 energy              E  

    # The Roe averages can be stored in arrays since they are later used 
    # in routine rpt2_euler to do the transverse wave splitting.

    n_cells = q.shape[1]
    n_edges = n_cells - 1

    u = np.zeros(n_edges)
    v = np.zeros(n_edges)
    h = np.zeros(n_edges)
    u2v2 = np.zeros(n_edges)
    c = np.zeros(n_edges)
    gm1c2 = np.zeros(n_edges)
    huv = np.zeros(n_edges)

    ind = [0, mu, mv, 3]
    wave = np.zeros((4, 4, n_edges))
    s = np.zeros((4, n_edges))
    amdq = np.zeros((4, n_edges))
    apdq = np.zeros((4, n_edges))

    for i in range(n_edges):
        ql = q[:, i]
        qr = q[:, i+1]
        rl_sqrt = np.sqrt(ql[0])
        rr_sqrt = np.sqrt(qr[0])
        rsq2 = rl_sqrt + rr_sqrt

        pl = gm1 * (ql[3] - 0.5*(ql[1]**2+ql[2]**2)/ql[0])
        pr = gm1 * (qr[3] - 0.5*(qr[1]**2+qr[2]**2)/qr[0])

        # Roe-averages
        u[i] = (ql[mu]/rl_sqrt + qr[mu]/rr_sqrt) / rsq2
        v[i] = (ql[mv]/rl_sqrt + qr[mv]/rr_sqrt) / rsq2
        h[i] = ((ql[3] + pl)/rl_sqrt + (qr[3] + pr)/rr_sqrt) / rsq2
        u2v2[i] = u[i]**2+v[i]**2
        c2 = gm1*(h[i] - 0.5*u2v2[i])
        c[i] = np.sqrt(c2)
        gm1c2[i] = gm1/c2
        huv[i] = h[i] - u2v2[i]

        dq1 = qr[0] - ql[0]
        dq2 = qr[mu] - ql[mu]
        dq3 = qr[mv] - ql[mv]
        dq4 = qr[3] - ql[3]
        alpha3 = gm1c2[i] * (huv[i]*dq1 + u[i]*dq2 + v[i]*dq3 - dq4)
        alpha2 = dq3 - v[i]*dq1
        alpha4 = (dq2 + (c[i] - u[i])*dq1 - c[i]*alpha3) / (2*c[i])
        alpha1 = dq1 - alpha3 - alpha4
        
        wave[ind, 0, i] = alpha1*np.array([1., u[i]-c[i], v[i], h[i]-u[i]*c[i]])
        wave[ind, 1, i] = alpha2*np.array([0., 0., 1., v[i]])
        wave[ind, 2, i] = alpha3*np.array([1., u[i], v[i], 0.5*u2v2[i]])
        wave[ind, 3, i] = alpha4*np.array([1., u[i]+c[i], v[i], h[i]+u[i]*c[i]])

        s[:, i] = np.array([u[i]-c[i], u[i], u[i], u[i]+c[i]])

    if efix:
        for i in range(n_edges):
            ql = q[:, i]
            qr = q[:, i+1]

            # Check 1-wave
            # ------------
            rhoim1 = ql[0]
            pim1 = gm1 * (ql[3] - 0.5*(ql[mu]**2 + ql[mv]**2) / rhoim1)
            cim1 = np.sqrt(gamma*pim1/rhoim1)
            s0 = ql[mu]/rhoim1 - cim1     # u-c in left state (cell i)

            # Check for fully supersonic case:
            if (s0 >= 0 and s[0, i] > 0):
                # everything is right-going
                amdq[:, i] = 0
                continue
            
            rho1 = ql[0] + wave[0, 0, i]
            rhou1 = ql[mu] + wave[mu, 0, i]
            rhov1 = ql[mv] + wave[mv, 0, i]
            en1 = ql[3] + wave[3, 0, i]
            p1 = gm1*(en1 - 0.5*(rhou1**2 + rhov1**2)/rho1)
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
                # 2- and 3-waves are rightgoing
                continue
            amdq[:, i] += s[1, i]*wave[:, 1, i]
            amdq[:, i] += s[2, i]*wave[:, 2, i]

            # Check 4-wave
            # ------------
            rhoi = qr[0]
            pi = gm1 * (qr[3] - 0.5*(qr[mu]**2 + qr[mv]**2) / rhoi)
            ci = np.sqrt(gamma*pi/rhoi)
            s3 = qr[mu]/rhoi + ci      # u+c in right state  (cell i+1)
            rho2 = qr[0] - wave[0, 3, i]
            rhou2 = qr[mu] - wave[mu, 3, i]
            rhov2 = qr[mv] - wave[mv, 3, i]
            en2 = qr[3] - wave[3, 3, i]
            p2 = gm1*(en2 - 0.5*(rhou2**2 + rhov2**2)/rho2)
            c2 = np.sqrt(gamma*p2/rho2)
            s2 = rhou2/rho2 + c2       # u+c to left of 4-wave
            if (s2 < 0 and s3 > 0):
                # transonic rarefaction in the 4-wave
                sfract = s2 * (s3-s[3, i]) / (s3-s2)
            elif (s[3, i] < 0):
                # 4-wave is leftgoing
                sfract = s[3, i]
            else:
                # 4-wave is rightgoing
                continue
            amdq[:, i] += sfract*wave[:, 3, i]

        # compute the rightgoing flux differences:
        # df = SUM s*wave   is the total flux difference and apdq = df - amdq
        df = np.zeros((4, n_edges))
        for i in range(n_edges):
            for mw in range(4):
                df[:, i] += s[mw, i]*wave[:, mw, i]
        
        apdq = df - amdq

    else: # No entropy fix
        for i in range(n_edges):
            for j in range(4):
                if s[j, i] < 0:
                    amdq[:, i] += s[j, i] * wave[:, j, i]
                else:
                    apdq[:, i] += s[j, i] * wave[:, j, i]

    return wave, s, amdq, apdq, u, v, h, u2v2, c, gm1c2, huv
    
def rpt2_euler(ixy, q, asdq, u, v, h, u2v2, c, gm1c2, huv, gamma=1.4):
    # The data is along a slice in the x-direction if ixy=1 or the y-direction if ixy=2
    if ixy == 1:
          mu = 1
          mv = 2
    else:
          mu = 2
          mv = 1

    gm1 = gamma - 1.

    # Conserved quantities:
    # 0 density             rho
    # 1 x_momentum          rho u
    # 2 y_momentum          rho v
    # 3 energy              E  

    n_cells = q.shape[1]
    n_edges = n_cells - 1

    ind = [0, mu, mv, 3]
    waveb = np.zeros((4, 4))
    bmasdq = np.zeros((4, n_edges))
    bpasdq = np.zeros((4, n_edges))

    for i in range(n_edges):

        alpha3 = gm1c2[i] * (huv[i]*asdq[0,i] + u[i]*asdq[mu,i] + v[i]*asdq[mv,i] - asdq[3,i])
        alpha2 = asdq[mu,i] - u[i]*asdq[0,i]
        alpha4 = (asdq[mv,i] + (c[i]-v[i])*asdq[0,i] - c[i]*alpha3) / (2*c[i])
        alpha1 = asdq[0,i] - alpha3 - alpha4

        # Note that the 2-wave and 3-wave travel at the same speed and 
        # are lumped together in wave[:, 1]. The 4-wave is then stored in
        # wave[:, 2].
        waveb[ind, 0] = alpha1*np.array([1., u[i], v[i]-c[i], h[i]-v[i]*c[i]])
        waveb[ind, 1] = alpha2*np.array([0., 1., 0., u[i]]) \
                      + alpha3*np.array([1., u[i], v[i], 0.5*u2v2[i]])
        waveb[ind, 2] = alpha4*np.array([1. ,u[i], v[i]+c[i], h[i]+v[i]*c[i]])

        sb = np.array([v[i] - c[i], v[i], v[i] + c[i]])

        for j in range(3):
            if sb[j] < 0:
                bmasdq[:, i] += sb[j] * waveb[:, j]
            else:
                bpasdq[:, i] += sb[j] * waveb[:, j]

    return bmasdq, bpasdq

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

def flux2(ixy, q1d, dtdx, limit_func, efix):
    nx_cells = q1d.shape[1]
    nx_edges = nx_cells - 1

    # Initialization
    qadd = np.zeros((4, nx_cells))
    fadd = np.zeros((4, nx_edges))
    gadd = np.zeros((4, 2, nx_cells))

    # Solve Riemann problem at each interface and compute Godunov updates
    # -------------------------------------------------------------------

    wave, s, amdq, apdq, u, v, h, u2v2, c, gm1c2, huv = rpn2_euler(ixy=ixy, q=q1d, efix=efix)

    # Check CFL
    CFL1d = dtdx*np.max(np.abs(s))
    if CFL1d > 1:
        raise ValueError(f"CFL={CFL1d} > 1.")

    # Set qadd for the donor-cell upwind method (Godunov)
    qadd[:, :-1] -= dtdx*amdq
    qadd[:, 1:] -= dtdx*apdq

    # modify F fluxes for second order q_{xx} correction terms
    # --------------------------------------------------------

    # Limit waves
    wave = limit(num_eqn=4, wave=wave, s=s, limit_func=limit_func)

    # Compute second-order corrections
    cqxx = np.zeros((4, nx_edges))
    for i in range(nx_edges):
        for mw in range(4):
            cqxx[:, i] += np.abs(s[mw, i]) * (1-dtdx*np.abs(s[mw, i])) * wave[:, mw, i]
        fadd[:, i] += 0.5*cqxx[:, i]

    # Incorporate cqxx into amdq and apdq so that it is split also
    amdq += cqxx
    apdq -= cqxx

    # Modify G flux for transverse propagation
    # ----------------------------------------

    # Split the left-going flux difference into down-going and up-going:
    bmamdq, bpamdq = rpt2_euler(ixy, q1d, amdq, u, v, h, u2v2, c, gm1c2, huv)

    # Modify flux below and above by B^- A^- Delta q and  B^+ A^- Delta q:
    # Modify G below cell i
    gadd[:, 0, :-1] -= 0.5*dtdx*bmamdq
    # Modify G above cell i
    gadd[:, 1, :-1] -= 0.5*dtdx*bpamdq

    # Split the left-going flux difference into down-going and up-going:
    bmapdq, bpapdq = rpt2_euler(ixy, q1d, apdq, u, v, h, u2v2, c, gm1c2, huv)

    # Modify flux below and above by B^- A^+ Delta q and  B^+ A^+ Delta q:
    # Modify G below cell i
    gadd[:, 0, 1:] -= 0.5*dtdx*bmapdq
    # Modify G above cell i
    gadd[:, 1, 1:] -= 0.5*dtdx*bpapdq

    return qadd, fadd, gadd, CFL1d


def solve_riemann(nx_cells, ny_cells, dt_initial, T, CFL, limit_func, efix, gamma=1.4):
    # There are two ghost cells at the boundary of each direction
    nx_edges = nx_cells - 1
    ny_edges = ny_cells - 1

    # Geometry and mesh
    x_ini = 0.
    x_fin = 1.
    y_ini = 0.
    y_fin = 1.
    dx = (x_fin - x_ini) / nx_cells
    dy = (y_fin - y_ini) / ny_cells
    x = x_ini + (np.arange(nx_cells) + 0.5)*dx
    y = y_ini + (np.arange(ny_cells) + 0.5)*dy
    
    xx, yy = np.meshgrid(x, y)

    # Initial data
    q = np.zeros((4, ny_cells, nx_cells))
    l = xx < 0.8
    r = xx >= 0.8
    b = yy < 0.8
    t = yy >= 0.8
    q[0,...] = 1.5 * r * t + 0.532258064516129 * l * t          \
                           + 0.137992831541219 * l * b          \
                           + 0.532258064516129 * r * b
    u = 0.0 * r * t + 1.206045378311055 * l * t                                \
                    + 1.206045378311055 * l * b                                \
                    + 0.0 * r * b
    v = 0.0 * r * t + 0.0 * l * t                                              \
                    + 1.206045378311055 * l * b                                \
                    + 1.206045378311055 * r * b
    p = 1.5 * r * t + 0.3 * l * t + 0.029032258064516 * l * b + 0.3 * r * b
    q[1,...] = q[0, ...] * u
    q[2,...] = q[0, ...] * v
    q[3,...] = 0.5 * q[0,...]*(u**2 + v**2) + p / (gamma - 1.0)

    # q[0,...] = 1.0 * r * t + 0.5313 * l * t          \
    #                                       + 0.8 * l * b          \
    #                                       + 0.5313 * r * b
    # u = 0.1 * r * t + 0.827 * l * t                                \
    #                 + 0.1 * l * b                                \
    #                 + 0.1 * r * b
    # v = 1.0 * r * t + 0.0 * l * t                                              \
    #                 + 0.0 * l * b                                \
    #                 + 0.7276 * r * b
    # p = 0.1 * r * t + 0.4 * l * t + 0.4 * l * b + 0.4 * r * b
    # q[1,...] = q[0, ...] * u
    # q[2,...] = q[0, ...] * v
    # q[3,...] = 0.5 * q[0,...]*(u**2 + v**2) + p / (gamma - 1.0)

    qold = q
    
    t = 0.0
    dt = dt_initial
    while t < T:

        dtdx = dt/dx
        dtdy = dt/dy

        qnew = qold.copy()
        maxCFL = 0.

        # Perform x-sweeps
        # ================
        for ny in range(1, ny_cells-1):
            q1d = qold[:, ny, :]
            qadd, fadd, gadd, CFL1d = flux2(1, q1d, dtdx, limit_func=limit_func, efix=efix)
            # qnew[:, ny, 1:-1] += qadd[:, 1:-1] - dtdx * (fadd[:, 1:] - fadd[:, :-1]) - dtdy * (gadd[:, 1, 1:-1] - gadd[:, 0, 1:-1])
            # qnew[:, ny-1, :] -= gadd[:, 0, :]
            # qnew[:, ny+1, :] += gadd[:, 1, :]
            qnew[:, ny, 2:-2] += qadd[:, 2:-2] - dtdx * (fadd[:, 2:-1] - fadd[:, 1:-2]) - dtdy * (gadd[:, 1, 2:-2] - gadd[:, 0, 2:-2])
            qnew[:, ny-1, 2:-2] -= dtdy * gadd[:, 0, 2:-2]
            qnew[:, ny+1, 2:-2] += dtdy * gadd[:, 1, 2:-2]
            maxCFL = np.max(np.array([maxCFL, CFL1d]))

            
        # Perform y-sweeps
        # ================
        for nx in range(1, nx_cells-1):
            q1d = qold[:, :, nx]
            qadd, gadd, fadd, CFL1d = flux2(2, q1d, dtdy, limit_func=limit_func, efix=efix)
            # qnew[:, 1:-1, nx] += qadd[:, 1:-1] - dtdy * (gadd[:, 1:] - gadd[:, :-1]) - dtdx * (fadd[:, 1, 1:-1] - fadd[:, 0, 1:-1])
            # qnew[:, :, nx-1] -= fadd[:, 0, :]
            # qnew[:, :, nx+1] += fadd[:, 1, :]
            qnew[:, 2:-2, nx] += qadd[:, 2:-2] - dtdy * (gadd[:, 2:-1] - gadd[:, 1:-2]) - dtdx * (fadd[:, 1, 2:-2] - fadd[:, 0, 2:-2])
            qnew[:, 2:-2, nx-1] -= dtdx * fadd[:, 0, 2:-2]
            qnew[:, 2:-2, nx+1] += dtdx * fadd[:, 1, 2:-2]
            maxCFL = np.max(np.array([maxCFL, CFL1d]))

        # Apply zero-order extrapolation BCs
        # Left
        qnew[:, :, 0] = qnew[:, :, 2]
        qnew[:, :, 1] = qnew[:, :, 2]
        # Right
        qnew[:, :, -1] = qnew[:, :, -3]
        qnew[:, :, -2] = qnew[:, :, -3]
        # Bottom
        qnew[:, 0, :] = qnew[:, 2, :]
        qnew[:, 1, :] = qnew[:, 2, :]
        # Top
        qnew[:, -1, :] = qnew[:, -3, :]
        qnew[:, -2, :] = qnew[:, -3, :]

        del qold
        qold = qnew

        # Update t and dt
        t += dt
        dt = CFL/maxCFL*dt
        dt = np.min(np.array([dt, T-t]))
        print(f"t={t: .4f}, max CFL={maxCFL}")
        
    fig, ax = plt.subplots()
    contour = ax.contourf(xx, yy, qnew[0,...], levels=400)
    fig.colorbar(contour, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig("riemann_2d", dpi=300)

    return qnew

def solve_sedov(nx_cells, ny_cells, dt_initial, T, CFL, limit_func, efix, gamma=1.4):
    # There are two ghost cells at the boundary of each direction
    n_ghost = 2

    # Geometry and mesh
    x_ini = 0.
    x_fin = 0.5
    y_ini = 0.
    y_fin = 0.5
    dx = (x_fin - x_ini) / nx_cells
    dy = (y_fin - y_ini) / ny_cells
    # x = x_ini - dx/2 + (np.arange(-n_ghost, nx_cells+n_ghost) + 0.5)*dx
    # y = y_ini - dy/2 + (np.arange(-n_ghost, ny_cells+n_ghost) + 0.5)*dy
    x = x_ini + (np.arange(-n_ghost, nx_cells+n_ghost) + 0.5)*dx
    y = y_ini + (np.arange(-n_ghost, ny_cells+n_ghost) + 0.5)*dy
    
    xx, yy = np.meshgrid(x, y)

    # Initial data
    q = np.zeros((4, ny_cells+2*n_ghost, nx_cells+2*n_ghost))
    # Setup density
    q[0,...] = 1.0
    
    # Setup energy
    r_init = 0.01
    nsub = 10
    E_sedov = 1.0
    dist = np.sqrt(xx*xx + yy*yy)
    p = 1e-5
    q[3,...] = p / (gamma - 1.0)
    for j, i in np.transpose(np.nonzero(dist < 2.0*r_init)):

        xsub = x[i] - 0.5*dx + (dx/nsub)*(np.arange(nsub) + 0.5)
        ysub = y[j] - 0.5*dy + (dy/nsub)*(np.arange(nsub) + 0.5)

        xx_sub, yy_sub = np.meshgrid(xsub, ysub)

        dist = np.sqrt(xx_sub**2 + yy_sub**2)

        n_in_pert = np.count_nonzero(dist <= r_init)

        # Weighted average of pressure
        p = n_in_pert*(gamma - 1.0)*E_sedov/(np.pi*r_init*r_init) + \
            (nsub*nsub - n_in_pert)*1e-5
        p = p/(nsub*nsub)

        q[3, j, i] = p/(gamma - 1.0)
    # q[3, ...] = 0.244816/dx/dy

    qold = q
    
    t = 0.0
    dt = dt_initial
    while t < T:

        dtdx = dt/dx
        dtdy = dt/dy

        qnew = qold.copy()
        maxCFL = 0.

        # Perform x-sweeps
        # ================
        for ny in range(1, ny_cells-1):
            q1d = qold[:, ny, :]
            qadd, fadd, gadd, CFL1d = flux2(1, q1d, dtdx, limit_func=limit_func, efix=efix)
            qnew[:, ny, 2:-2] += qadd[:, 2:-2] - dtdx * (fadd[:, 2:-1] - fadd[:, 1:-2]) - dtdy * (gadd[:, 1, 2:-2] - gadd[:, 0, 2:-2])
            qnew[:, ny-1, 2:-2] -= dtdy * gadd[:, 0, 2:-2]
            qnew[:, ny+1, 2:-2] += dtdy * gadd[:, 1, 2:-2]
            maxCFL = np.max(np.array([maxCFL, CFL1d]))

            
        # Perform y-sweeps
        # ================
        for nx in range(1, nx_cells-1):
            q1d = qold[:, :, nx]
            qadd, gadd, fadd, CFL1d = flux2(2, q1d, dtdy, limit_func=limit_func, efix=efix)
            qnew[:, 2:-2, nx] += qadd[:, 2:-2] - dtdy * (gadd[:, 2:-1] - gadd[:, 1:-2]) - dtdx * (fadd[:, 1, 2:-2] - fadd[:, 0, 2:-2])
            qnew[:, 2:-2, nx-1] -= dtdx * fadd[:, 0, 2:-2]
            qnew[:, 2:-2, nx+1] += dtdx * fadd[:, 1, 2:-2]
            maxCFL = np.max(np.array([maxCFL, CFL1d]))

        # Apply BCs
        # Left (not include corner): reflecting BC
        qnew[np.array([0,2,3]), 2:-2, 0] = qnew[np.array([0,2,3]), 2:-2, 3]
        qnew[np.array([0,2,3]), 2:-2, 1] = qnew[np.array([0,2,3]), 2:-2, 2]
        # reflecting u
        qnew[1, 2:-2, 0] = -qnew[1, 2:-2, 3]
        qnew[1, 2:-2, 1] = -qnew[1, 2:-2, 2]
        # Right (not include corner): zero-order extrapolation
        qnew[:, 2:-2, -1] = qnew[:, 2:-2, -3]
        qnew[:, 2:-2, -2] = qnew[:, 2:-2, -3]
        # Bottom: reflecting BC
        qnew[np.array([0,1,3]), 0, :] = qnew[np.array([0,1,3]), 3, :]
        qnew[np.array([0,1,3]), 1, :] = qnew[np.array([0,1,3]), 2, :]
        # reflecting v
        qnew[2, 0, :] = -qnew[2, 3, :]
        qnew[2, 1, :] = -qnew[2, 2, :]
        # Top: zero-order extrapolation
        qnew[:, -1, :] = qnew[:, -3, :]
        qnew[:, -2, :] = qnew[:, -3, :]

        del qold
        qold = qnew

        # Update t and dt
        t += dt
        dt = CFL/maxCFL*dt
        dt = np.min(np.array([dt, T-t]))
        print(f"t={t: .4f}, max CFL={maxCFL}")
        
    q = qold

    fig, ax = plt.subplots()
    contour = ax.contourf(xx[n_ghost:-n_ghost, n_ghost:-n_ghost], 
                          yy[n_ghost:-n_ghost, n_ghost:-n_ghost], 
                          q[0,n_ghost:-n_ghost, n_ghost:-n_ghost], 
                          levels=400)
    fig.colorbar(contour, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig("sedov_2d", dpi=300)

    return q[:,n_ghost:-n_ghost, n_ghost:-n_ghost]


if __name__ == "__main__":
    device = 'cpu'

    # Load Model
    model = FluxLimiter(1,1,64,5,act="tanh") #
    # model.load_state_dict(torch.load("model_euler_best.pt"))
    model.load_state_dict(torch.load("model_linear_tanh.pt", map_location=torch.device(device)))
    model = model.to(device)

    def neural_flux_limiter(r):
        model.eval()
        with torch.no_grad():
            phi = model(torch.Tensor(r).view(-1, 1).to(device))
        return phi.numpy().squeeze()
    
    # q = solve_sedov(nx_cells=50, ny_cells=50, dt_initial=1e-5, T=0.1, CFL=0.005, limit_func=utils.vanLeer, efix=False)
    # q = solve_sedov(nx_cells=100, ny_cells=100, dt_initial=1e-5, T=0.1, CFL=0.4, limit_func=utils.minmod, efix=False)
    # np.save("solution_sedov_2d.npy", q)
    # exit()
    
    q = solve_riemann(nx_cells=200, ny_cells=200, dt_initial=1e-3, T=0.8, CFL=0.4, limit_func=neural_flux_limiter, efix=False)
    np.save("solution_riemann_2d.npy", q)















