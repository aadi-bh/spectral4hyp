# Operators

import numpy as np
from numpy.fft import *
from numpy import pi
from scipy import sparse
from common import *

# Sort of roughness indicator
def to_filter(uh, tol=1.0e-2):
    uh = fftshift(uh)
    l2 = np.sum(np.abs(uh)**2)
    N = len(uh)
    M = N//6
    ends = np.r_[uh[:M], uh[-M:]]
    el2 = np.sum(np.abs(ends)**2)
    return el2/l2

# Linear advection differential operator
# u_t + a u_x = 0
def linadv(t, u_hat, N, M, filter, a= 2 * pi): 
    NN = (2 * M//2) + N
    u_hat = pad(u_hat, M//2)
    kk = freqs(NN)
    nonlinear = -1j * kk * u_hat
    nonlinear *= a
    return unpad(mask(nonlinear), M//2)

# Burgers equation
# u_t + a(u**2/2)_x = 0
def burgers(t, u_hat, N, M, filter, a=1):
    u_hat = pad(u_hat, M//2)
    # Check the roughness of the function.
    res = to_filter(u_hat)
    if res > 0.5:
        u_hat *= filter
        print("Applied filter.")
    NN = len(u_hat)
    u = ifft(u_hat)
    kk = freqs(NN)
    nonlinear = -1 * fft(u**2/2) * 1j * kk
    nonlinear *= a
    return unpad(mask(nonlinear), M//2)

# Heat equation solution semigroup
def semigroup_heat(dt, k, eps):
    S_half = sparse.diags(np.exp(-1 * eps * k**2 * dt / 2))
    S = sparse.diags(np.exp(-1 * eps * k**2 * dt))
    return S_half, S

# For purely nonlinear, can use this.
def semigroup_none(dt, k, eps):
    S = sparse.diags(np.ones(len(k)))
    S_half = S
    return S_half, S

# u_t = 0, just in case.
def zero(t, u_hat, N, M, eps, filter, a=1):
    return np.zeros(len(u_hat));

