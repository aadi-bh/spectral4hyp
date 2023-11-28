# Make plots

import matplotlib.pyplot as plt
from argparse import Namespace
import numpy as np
import os
from common import *
from filters import *

# plt.style.use('fivethirtyeight')
plt.style.use('./presentation.mplstyle')
def filterplots():
    x = np.linspace(-2, 2, 100)
    fig, ax = plt.subplots(2,2, figsize=(14,8))
    ax[0][0].plot(x, cesaro(x), label='Cesaro')
    ax[0][1].plot(x, raisedcos(x), label="Raised cosine")
    ax[1][0].plot(x, lanczos(x), label="Lanczos")
    ax[1][1].plot(x, exponential(x, 2), label="Exponential2")
    for i in range(2):
        for j in range(2):
            ax[i][j].legend()
            ax[i][j].grid()
    fig.savefig("filters.svg")

# Plot all the ggb polys
def ggbplots():
    import ggb
    x = np.linspace(-1, 1, 2048)
    fig, ax = plt.subplots(figsize=(14,8))
    l = 3
    ax.set_ylim((-20, 20))
    for n in range(5):
        ax.plot(x, ggb.C(x, n, l), label=f"$k={n}$")
    ax.legend()
    fig.savefig('ggbs.svg')

# Plot the resolution of the given FFT
def plot_resolution(c, ax, **kwargs):
    k = fftshift(freqs(len(c)))
    ax.semilogy(k, np.abs(fftshift(c)), **kwargs)
    return

def plot_error_fft(uh, exact, ax, **kwargs):
    x = exact[0]
    if len(uh) < len(x):
        u = ifft_at(x, uh)
        e = np.abs(u - exact[1])
        ax[1].plot(x, e, **kwargs)
    else:
        print("WARNING: Interpolating exact not supported yet.\nSupply exact solution on finer grid.")
        return

def plot_error(c, exact, ax, **kwargs):
    print("print_error not implemented yet.")

# Fourier interpolate the IFFT
def smoothplot(v, ax,nn=2048, **plotargs):
    n = len(v)
    w = pad(v, (nn - n)//2)
    dy = (xmax - xmin) / n
    dz = (xmax - xmin) / nn
    y = cgrid(n)
    z = np.linspace(y[0], y[-1]+dy-dz, nn, endpoint=True)
    ax.plot(z, ifft(w).real, **plotargs)
    return z, ifft(w)

def smooth_and_error(solax, errax, v, exact, nn=2048, **plotargs):
    if np.any(exact == None):
        print("Exact not found.")
        return smoothplot(v, solax, **plotargs)
    plot_and_error(solax, errax, exact[0], ifft_at(exact[0], v).real, exact, **plotargs)

def plot_and_error(solax, errax, x, u, exact, **plotargs):
    solax.plot(x, u, **plotargs)
    if np.any(exact == None):
        print("Exact not found.")
        return
    if len(x) > len(exact[0]):
        ei = np.interp(x, exact[0], exact[1])
        errax.semilogy(x, np.abs(u-ei), **plotargs)
    elif (len(x) < len(exact[0])):
        ui = np.interp(exact[0], x, u)
        errax.semilogy(exact[0], np.abs(ui-exact[1]), **plotargs)
    else:
        errax.semilogy(x, np.abs(u-exact[1]), **plotargs)
    errax.set_ylim((1e-6, 1e1))


def convergence_plot(exactfile, filenames, saveas, **kwargs):
    for fn in filenames:
        if not os.path.isfile(fn):
            print(fn + " not found or is invalid.")
            exit(1)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 8), width_ratios=[3,1])
    fig.tight_layout()

    ex = np.loadtxt(exact).transpose()
    ax[0].plot(ex[0] * 2 * np.pi, ex[1], 'ko', markersize=0.1, label="Exact")

    x = np.arange(xmin, xmax, len(ex[0]))
    for fn in filenames:
        ku = np.loadtxt(fn)
        k = ku[0]
        u = ku[1]
        N = len(k)
        ax[0].plot(x, ifft(u[-1]).real, label=str(N))
        plot_resolution(u[-1], ax[1], {'linewidth':0.5, 'markersize':0.5})
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 8), width_ratios=[3,1])
    x = np.linspace(xmin, xmax, 1000)
    ax[0].plot(x, initial_condition(x), linewidth=0.1, color='k', label="init")
    ax[0].legend()
    fig.savefig(saveas)

def solplot(sols, args, plotname):
    return
    nn = 2048
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 8), width_ratios=[3,1])
    # Initial
    x = np.linspace(xmin, xmax, 1000)
    ax[0].plot(x, initial_condition(x), linewidth=1, color='k', label="Init")
    # All the modes
    for (t, uf, v) in sols:
        n = len(uf)
        label = str(n) + f", t={np.round(t, 3)}"
        plots.smoothplot(uf, ax[0], label=label, linewidth=1)
        plots.plot_resolution(uf, ax[1], linewidth=0.5, markersize=0.5)
        if args.ggb:
            label += ",ggb"
            if sigma != filters.no_filter:
                label += "," + args.filter
            ax[0].plot(cgrid(n), v, label=label) 
        elif sigma != filters.no_filter:
            label += "," + args.filter
            plots.smoothplot(uf, ax[0])
        if args.show_markers:
            ax[0].plot(cgrid(n), ifft(uf), "+", color= "red", markersize=5)
    # Exact
    if args.exact != None:
        exact = np.loadtxt(args.exact).transpose()
        ax[0].plot(exact[0] * 2 * np.pi, exact[1], 'ko', markersize=0.8, label="Exact")

    ax[0].grid(visible=True)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(plotname)
    plt.close()
    print("Saved plot to "+plotname)
# convergence_plot('a', ['16.txt', '32.txt', '64.txt'])
if __name__ == "__main__":
    ggbplots()
    filterplots()
    '''
    initial_condition = ic.sin
    args = Namespace(L=3, N=[16, 64], Tf=1.0, cfl=0.1, exact=None, filter='no_filter', ggb=False, ic='sin', pde='burgers', show_markers=False)
    sols, prefix, _ = main.run(args)
#    prefix = 'burgers-1.0-sin-no_filter-gFalse'
#    sols = [np.loadtxt(file) for file in [prefix+'-16.txt', prefix+'-64']]
    solplot(sol, args, initial_condition, prefix+'.svg')
    '''

