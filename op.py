# Operators

import numpy as np
from numpy.fft import *
from scipy import sparse
from numpy import pi
from common import *


def linadv(t, u_hat, N, M, filter, a= 2 * pi): 
    NN = (2 * M//2) + N
    u_hat = pad(u_hat, M//2)
    kk = freqs(NN)
    nonlinear = -1j * kk * u_hat
    nonlinear *= a
    nonlinear *= filter
    return unpad(mask(nonlinear), M//2)

def burgers(t, u_hat, N, M, filter, a=1):
    NN = (2 * M//2) + N
    u_hat = pad(u_hat, M//2)
    u = ifft(u_hat)
    kk = freqs(NN)
    nonlinear = -1 * fft(u**2/2) * 1j * kk
    nonlinear *= a
    nonlinear *= filter
    return unpad(mask(nonlinear), M//2)

def semigroup_heat(dt, k, eps):
    S_half = sparse.diags(np.exp(-1 * eps * k**2 * dt / 2))
    S = sparse.diags(np.exp(-1 * eps * k**2 * dt))
    return S_half, S

def semigroup_none(dt, k, eps):
    S = sparse.diags(np.ones(len(k)))
    S_half = S
    return S_half, S

def zero(t, u_hat, N, M, eps, filter, a=1):
    return np.zeros(len(u_hat));
