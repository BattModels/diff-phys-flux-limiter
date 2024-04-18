import numpy as np
import torch

def minmod(r):
    return np.maximum(0, np.minimum(1, r))

def superbee(r):
    return np.maximum(np.maximum(0, np.minimum(2 * r, 1)), np.minimum(r, 2))

def vanLeer(r):
    return (r + np.abs(r)) / (1 + np.abs(r))

def FOU(r):
    return np.zeros_like(r)

def LaxWendroff(r):
    return np.ones_like(r)

def compute_r(u, ul, ur):
    r = (u - ul) / (ur - u)
    # r = np.where(np.isnan(r), 1e6, r)
    # r = np.where(np.isinf(r), 1e6, r)

    # In case that ur - u accidentally equals to -eps so we use rand
    eps = np.random.rand(1)*1e-16 
    r = (u - ul) / (ur - u + eps)
    return r

def compute_r_torch(u, ul, ur):
    r = (u - ul) / (ur - u)
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