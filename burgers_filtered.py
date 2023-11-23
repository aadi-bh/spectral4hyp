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
import argparse

execfile("common.py")

xmin = 0
xmax = 2 * pi

def saw_tooth(x):
    left = np.where(x <= pi, 1, 0);
    return x * left + (x - 2 * pi) * (1-left)
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

def zero(t, u_hat, N, M, eps, filter, a=1):
    return np.zeros(len(u_hat));

def linadv(t, u_hat, N, M, filter, a=1): 
    # because fft scales these, and we want the ifft to work as usual
    NN = (2 * M//2) + N
    u_hat = pad(u_hat, M//2) * NN / N 
    kk = freqs(NN)
    nonlinear = -1j * kk * u_hat
    nonlinear *= a
    nonlinear *= filter
    return unpad(mask(nonlinear), M//2)
def burgers(t, u_hat, N, M, filter, a=1):
    # because fft scales these, and we want the ifft to work as usual
    NN = (2 * M//2) + N
    u_hat = pad(u_hat, M//2) * NN / N
    u = ifft(u_hat)
    kk = freqs(NN)
    nonlinear = -1 * fft(0.5 * u * u) * 1j * kk
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

parser = argparse.ArgumentParser()
# parser.add_argument('-N', type=int, help='Number of cells', default=100)
parser.add_argument('--cfl', type=float, help='CFL number', default=0.1)
# parser.add_argument('-scheme',
#                     choices=('C','LF','GLF','LLF','LW','ROE','EROE','GOD'),
#                     help='Scheme', default='LF')
parser.add_argument('--ic',
                    choices=('saw_tooth','sin','step'),
                    help='Initial condition', default='sin')
parser.add_argument('--Tf', type=float, help='Final time', default=1.0)
parser.add_argument('--pde', choices=('linadv', 'burgers'), default='burgers')
parser.add_argument('--add_visc', type=bool, default=False)
parser.add_argument('--use_filter', type=bool, default=False)
parser.add_argument('--max_lgN', type=int, default=7)
parser.add_argument('--integrator', choices=('solve_ivp', 'elrk4'), default='elrk4')
args = parser.parse_args()

a = 2 * pi
rhs = linadv
initial_condition = sin
visc = semigroup_none
USE_FILTER = False
tf = 2
cfl = 0.5
if args.pde == 'burgers':
    a = 1
    rhs = burgers
if args.ic == 'saw_tooth':
    initial_condition = saw_tooth
elif args.ic == 'sin':
    initial_condition = sin
elif args.ic == 'step':
    initial_condition = step
if args.add_visc == True:
    visc = semigroup_heat
tf = max(0, args.Tf)
cfl = min(1, args.cfl)
USE_FILTER = args.use_filter

dt = 0.02
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 10), width_ratios=[3,1])
fig.tight_layout()
x = np.linspace(xmin, xmax, 1000)
ax[0].plot(x, initial_condition(x), linewidth=0.1, color='k', label="init")

for i in range(0, args.max_lgN - 4 + 1):
    N = np.power(2, i + 4);
    print(f"N={N}")
    M = 10 * N // 2;
    m = M // 2;
    NN = (2 * m) + N
    dx = (xmax-xmin) / N
    dt = min(dt, cfl * dx)
    x = np.arange(xmin, xmax, dx)
    k = freqs(N)
    kk = freqs(NN)
    filter = np.ones(len(kk))
    p = i 
    if USE_FILTER == True:
        filter = create_filter(kk, sigma, {'p':p})
    args = (N, M, filter, a)
    u_hat_init = fft(initial_condition(x))
    S_half, S = visc(dt, k, eps = 1e-2)
    '''
    output = solve_ivp(fun = rhs,
                         t_span = [0, tf],
                         t_eval = [0, tf],
                         y0 = u_hat_init,
                         args = args)
    times, u = output.t, output.y.transpose()
    '''
    times, u = elrk4([S_half, S], rhs, u_hat_init, (0, tf), dt, args)
    ax[0].plot(x, ifft(u[-1]).real, label=str(N)+f", t={np.round(times[-1], 3)}")
    plot_resolution(u[-1], ax[1], {'linewidth':0.1, 'markersize':0.1})
    '''
    nr = 11
    f, a = plt.subplots(ncols=2, nrows = nr,
                        figsize=(10, 2 * nr),width_ratios=[3,1])
    f.tight_layout()
    for k in range(nr):
        j = int(np.linspace(0, len(times), nr, endpoint=True)[k])
        print(j)
        a[k][0].plot(x, ifft(u[j]).real, label=str(np.round(times[j], 3)))
        a[k][0].legend()
        plot_resolution(u[j], a[k][1], {'color':'k', 'linewidth':0.1, 'markersize':0.1})
    f.savefig(f"{N}-f{USE_FILTER}-f{str(p)}.png")
    '''
    np.savetxt(f"{N}.txt", np.vstack((k, u[-1].real)))
    print("Saved file.")

if rhs == burgers:
    if tf == 1:
        god = np.loadtxt("burg3_GOD_5.txt").transpose()
    if tf == 0.6:
        god = np.loadtxt("burg3_GOD_3.txt").transpose()
    if tf == 0.2:
        god = np.loadtxt("burg3_GOD_1.txt").transpose()
    ax[0].plot(god[0] * 2 * np.pi, god[1], 'ko', markersize=0.1, label="Godunov flux")
ax[0].legend()
fig.savefig(f"f{rhs}-f{str(USE_FILTER)}.png")

plt.close()
print("Done.")
