# Make plots

import matplotlib.pyplot as plt
import numpy as np
import os
from common import *
from filters import *

xmin, xmax = 0, 2 * np.pi

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
    fig.savefig("filters.png")

def plot_resolution(c, ax, **kwargs):
    k = fftshift(freqs(len(c)))
    ax.semilogy(k, np.abs(fftshift(c)), **kwargs)
    return

def smoothplot(v, ax,nn=2048, **plotargs):
    n = len(v)
    w = pad(v, (nn - n)//2)
    dy = (xmax - xmin) / n
    y = cgrid(n)
    z = np.linspace(y[0], y[-1]+dy, nn, endpoint=True)
    ax.plot(z, ifft(w).real, **plotargs)

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

# convergence_plot('a', ['16.txt', '32.txt', '64.txt'])
if __name__ == "__main__":
    filterplots()

