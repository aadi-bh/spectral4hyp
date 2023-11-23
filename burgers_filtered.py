#!/usr/bin/env python
# coding: utf-8

# # Burgers' with spectral method

import numpy as np
from numpy.fft import *
from scipy.integrate import solve_ivp
from scipy import sparse
from numpy import pi
import matplotlib.pyplot as plt
from past.builtins import execfile

execfile("common.py")

xmin = 0
xmax = 2 * pi

def saw_tooth(x):
    return np.where(x < pi, x / pi, 2 - x / pi)
def sin(x):
    return np.sin(x)
def step(x):
    y = np.where(x < np.pi, 1, 0) * np.where(x > np.pi/2, 1, 0)
    return y

def plot_resolution(c, ax, kwargs):
    k = fftshift(freqs(len(c)))
    ax.semilogy(k, np.abs(fftshift(c)), **kwargs)
    return

def pad(c, m):
    # ASSUME c is fft(u) / len(u)
    N = len(c)
    newN = 2*m + N
    r = fftshift(c)
    r = np.r_[np.zeros(m), r, np.zeros(m)]
    r *= 1 # N / newN
    r = fftshift(r)
    return r
def unpad(c, m):
    N = len(c)
    newN = N - 2 * m
    r = fftshift(c)
    r = r[m:m + newN]
    r *= 1 # N / newN
    r = fftshift(r)
    return r
def freqs(n, x1=xmin, x2=xmax):
    return fftfreq(n, 1./ (n))

def mask(c, tol=1e-14):
    return c * np.where(np.abs(c)<tol, 0, 1)

def sigma(eta, p=1):
    return np.exp(-15* np.power(eta, 2*p))

def create_filter(k, sigma, args):
    K = np.max(np.abs(k))
    filter = sigma(k / K, **args)
    return filter
def no_filter(k, sigma, args):
    return np.ones(len(kk))
def zero(u_hat, N, M, eps, filter, a=1):
    return np.zeros(len(u_hat));
def linadv(u_hat, N, M, filter, a=1): 
    u_hat = pad(u_hat, M//2) * NN / N # because fft scales these, and we want the ifft to work as usual
    kk = freqs(NN)
    nonlinear = -1j * kk * u_hat
    nonlinear *= a
    nonlinear *= filter
    return unpad(mask(nonlinear), M//2)
def burgers(u_hat, N, M, filter, a=1):
    # because fft scales these, and we want the ifft to work as usual
    u_hat = pad(u_hat, M//2) * NN / N
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

a = 1
initial_condition = sin
rhs = burgers
visc = semigroup_heat
USE_FILTER = True
tf = 0.9
dt = 0.1
# fig, axs = plt.subplots(nrows = 9, ncols = 2, figsize = (10, 2 * 9), width_ratios=[3,1])
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 10), width_ratios=[3,1])
fig.tight_layout()
for i in range(9):
    N = np.power(2, i + 4);
    M = 3 * N // 2;
    NN = (2 * M//2) + N
    m = M // 2;
    dx = (xmax-xmin) / N
    x = np.arange(xmin, xmax, dx)
    k = freqs(N)
    kk = freqs(NN)
    filter = np.ones(len(kk))
    p = i
    if USE_FILTER == True:
        filter = create_filter(kk, sigma, {'p':p})
    args = (N, M, filter)
    u_hat_init = fft(initial_condition(x))
#    output = solve_ivp(semigroup_heat, rhs, [0, tf], u_hat_init, t_eval=np.linspace(0, tf, 1+int(tf/dt)), args=args)
    S_half, S = visc(dt, k, eps = 1e-1)
    times, u = elrk4([S_half, S], rhs, u_hat_init, (0, tf), dt, args)
#    print(output['message'])
#    u = output['y'].transpose()
#    times = output['t']
#    axs[i][0].plot(x, ifft(u[-1]).real, label=str(N))
#    axs[i][0].legend()
    ax[0].plot(x, ifft(u[-1]).real, label=str(N)+f", t={np.round(times[-1], 3)}")
    ax[0].legend()
#    plot_resolution(u[-1], axs[i][1])
    plot_resolution(u[-1], ax[1], {'linewidth':0.1, 'markersize':0.1})

    f, a = plt.subplots(ncols=2,nrows=len(times),figsize=(10, 2 * len(times)),width_ratios=[3,1])
    f.tight_layout()
    for j in range(len(times)):
        a[j][0].plot(x, ifft(u[j]).real, label=str(np.round(times[j], 3)))
        a[j][0].legend()
        plot_resolution(u[j], a[j][1], {'color':'k', 'linewidth':0.1, 'markersize':0.1})
    np.savetxt(f"{N}.txt", u[-1].real)
    f.savefig(f"{N}-f{USE_FILTER}-f{str(p)}.png")
fig.savefig(f"out-f{str(USE_FILTER)}-f{str(p)}.png")
