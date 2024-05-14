import numpy as np
import torch

def minmod(r):
    return np.maximum(0, np.minimum(1, r))

def minmod_torch(r):
    return torch.maximum(torch.tensor([0.]), torch.minimum(torch.tensor([1.]), r))

def superbee(r):
    return np.maximum(np.maximum(0, np.minimum(2 * r, 1)), np.minimum(r, 2))

def superbee_torch(r):
    return torch.maximum(torch.maximum(torch.tensor([0.]), torch.minimum(2 * r, torch.tensor([1.]))), torch.minimum(r, torch.tensor([2.])))

def vanLeer(r):
    return (r + np.abs(r)) / (1 + np.abs(r))

def koren(r):
    return np.maximum(0, np.minimum(2*r, np.minimum((1+2*r)/3, 2)))

def FOU(r):
    return np.zeros_like(r)

def LaxWendroff(r):
    return np.ones_like(r)

def compute_r(u, ul, ur):
    # r = (u - ul) / (ur - u)
    # r = np.where(np.isnan(r), 1e6, r)
    # r = np.where(np.isinf(r), 1e6, r)

    # In case that ur - u accidentally equals to -eps so we use rand
    eps = np.random.rand(1)*1e-16 
    r = (u - ul) / (ur - u + eps)
    return r

def compute_r_torch(u, ul, ur):
    # r = (u - ul) / (ur - u)
    # r = torch.where(torch.isnan(r), 1e6, r)
    # r = torch.where(torch.isinf(r), 1e6, r)

    # In case that ur - u accidentally equals to -eps so we use rand
    eps = torch.rand(1).item()*1e-16 
    r = (u - ul) / (ur - u + eps)
    return r

def sin_wave(N=1000, L=1):
    dx = L / N
    x0 = (np.arange(N)+0.5)*dx
    u0 = np.sin(2*np.pi*x0/L)
    return x0, u0

def square_wave(N=1000, L=1):
    dx = L / N
    x0 = (np.arange(N)+0.5)*dx
    u0 = np.where(np.logical_and(x0 >= 0.25, x0 <= 0.75), 1., 0.)
    return x0, u0

def ramp(slope_ratio, N=100, L=1):
    dx = L / N
    x0 = (np.arange(N)+0.5)*dx
    start = 0.25*L
    mid = x0[int(np.floor((N+1)/2)-1)]
    end = 0.75*L
    mid_state = 1.
    left_slope = mid_state / (0.25*L)
    right_slope = left_slope / slope_ratio
    left_ramp = mid_state + left_slope * (x0 - mid)
    right_ramp = mid_state + right_slope * (x0 - mid)
    ramp = np.where(x0 <= mid, left_ramp, right_ramp)
    u0 = np.where(np.logical_and(x0 >= start, x0 <= end), ramp, 0.)
    return x0, u0

def linear_spike(N=1000, L=1):
    dx = L / N
    x0 = (np.arange(N)+0.5)*dx
    start = 0.25*L
    mid = 0.5*L
    end = 0.75*L
    peak = 1.
    slope = peak / (0.25*L)
    spike = peak - slope * np.abs(x0 - mid)
    u0 = np.where(np.logical_and(x0 >= start, x0 <= end), spike, 0.)
    return x0, u0

def step(N=1000, L=1):
    dx = L / N
    x0 = (np.arange(N)+0.5)*dx
    u0 = np.where(x0 < 0.5, 1., -1.)
    return x0, u0

def wave_combination(N=200, L=2):
    a = 0.5
    z = -0.7
    delta = 0.005
    alpha = 10
    beta = np.log(2)/(36*delta**2)

    def F(x, alpha, a):
        return np.sqrt(np.maximum(1-alpha**2*(x-a)**2,0))
    
    def G(x, beta, z):
        return np.exp(-beta*(x-z)**2)
    
    dx = L / N
    x0 = (np.arange(N)+0.5)*dx - 1 - dx/2
    u0 = 1/6 * (G(x0,beta,z-delta) + G(x0,beta,z+delta) + 4*G(x0,beta,z)) * np.logical_and(x0 >= -0.8, x0 <= -0.6) \
         + 1 * np.logical_and(x0 >= -0.4, x0 <= -0.2) \
         + (1 - np.abs(10*(x0 - 0.1))) * np.logical_and(x0 >= 0, x0 <= 0.2) \
         + 1/6 * (F(x0,alpha,a-delta) + F(x0,alpha,a+delta) + 4*F(x0,alpha,a)) * np.logical_and(x0 >= 0.4, x0 <= 0.6)
    
    return x0, u0