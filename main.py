#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.fft import *
from scipy.integrate import solve_ivp
from scipy import sparse
from numpy import pi
import matplotlib.pyplot as plt
import argparse

from common import *
import ic
import op
import filters
import plots
import ggb


xmin, xmax = 0, 2 * pi

parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, help='Number of cells', action='append')
parser.add_argument('--cfl', type=float, help='CFL number', default=0.1)
# parser.add_argument('-scheme',
#                     choices=('C','LF','GLF','LLF','LW','ROE','EROE','GOD'),
#                     help='Scheme', default='LF')
parser.add_argument('--ic',
                    choices=('saw_tooth','sin','step', 'bump'),
                    default='sin', help = "Initial condition")
parser.add_argument('--Tf', type=float, default=1.0, help = "Final time")
parser.add_argument('--pde', choices=('linadv', 'burgers'), default='burgers', help = "PDE to solve")
# parser.add_argument('--add_visc', type=bool, default=False)
parser.add_argument('--filter', choices=('no_filter', 'exponential', 'cesaro', 'raisedcos', 'lanczos'), default='no_filter', help = "Which filter, if any")
# parser.add_argument('--filterp', type=int, default=1, "p value for the exponential")
parser.add_argument('--ggb', type=bool, default=False, help = "Whether to reconstruct the analytic part of the solution")
parser.add_argument('-L', type=int, default=3, help = "Number of elements in the GGB basis")
# parser.add_argument('--max_lgN', type=int, default=7, help = "Largest power of 2 to calculate until")
#parser.add_argument('--integrator', choices=('solve_ivp', 'elrk4'), default='elrk4')
parser.add_argument('--show_markers', type=bool, default=False, help = "Show the original sample values in red crosses")
parser.add_argument('--exact', type=str, default=None, help = "Exact solution file")
args = parser.parse_args()

if __name__ == '__main__':
    rhs = op.linadv
    initial_condition = ic.sin
#    visc = op.semigroup_none
    sigma = filters.no_filter
    tf = 2
    cfl = 0.5
    if args.pde == 'burgers':
        rhs = op.burgers
    if args.ic == 'saw_tooth':
        initial_condition = ic.saw_tooth
    elif args.ic == 'sin':
        initial_condition = ic.sin
    elif args.ic == 'step':
        initial_condition = ic.step
    elif args.ic == 'bump':
        initial_condition = ic.bump
    if args.filter == 'no_filter':
        sigma = filters.no_filter;
    elif args.filter == 'exponential':
        sigma = filters.exponential
    elif args.filter == 'cesaro':
        sigma = filters.cesaro
    elif args.filter == 'raisedcos':
        sigma = filters.raisedcos
    elif args.filter == 'lanczos':
        sigma = filters.lanczos
#    if args.add_visc == True:
#        visc = op.semigroup_heat
    tf  = max(0, args.Tf)
    cfl = min(1, args.cfl)

    plotname = f"{args.pde}-{tf}-{args.ic}-{args.filter}-g{args.ggb}.png"
    print("PDE   :", args.pde)
    print("TF    :", tf)
    print("CFL   :", cfl)
    print("IC    :", args.ic)
    print("FILTER:", args.filter)
    print("FILE  :", plotname)
#    print("VISC  :", args.add_visc)
#    plotname = f"{args.pde}-visc{str(args.add_visc)}-{tf}-{args.ic}-{args.filter}.png"
    sols = []
    for N in args.N:
        M = 3 * N // 2;
        m = M // 2;
        NN = (2 * m) + N
        dx = (xmax-xmin) / N
        dt = (cfl * dx)
        x = cgrid(N)
        k, kk = freqs(N), freqs(NN)
        p = 2
        print(f"N={N}")
        arguments = (N, M, create_filter(kk, sigma, p=p))
        u_hat_init = fft(initial_condition(x)) 

        times = [0]
        us = [u_hat_init]
        if (tf > 0):
            output = solve_ivp(fun = rhs,
                             t_span = [0, tf],
                             t_eval = [0, tf],
                             y0 = u_hat_init,
                             args = arguments)
            print(output.message)
            times, us = output.t, output.y.transpose()

        # Postprocessing!
        # Apply whatever filter was chosen.
        uh = us[-1]
        uf = uh * create_filter(k, sigma, p = p)
        v = ifft(uf).real
        # Gegenbauer reconstruction
        if args.ggb:
            # Conveniently for us, the shock stays put at pi
            left = np.where(x < pi, True, False)
            right = np.where(x > pi, True, False)
            ggbleft = ggb.recon_fft(x[left], uf, args.L)
            ggbright = ggb.recon_fft(x[right], uf, args.L)
            v[left] = ggbleft
            v[right] = ggbright
        sols.append((times[-1], uf, v))
        filename = f"{N}.txt"
        np.savetxt(filename, np.vstack((x, us[-1], v)))
        print("Saved solution to " + filename)

    # PLOT
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
    print("Done.")

    '''
    ## Gegenbauer test
    u = u[-1]
    x = cgrid(len(u))
    plots.smoothplot(u, plt)
#    plt.plot(x, ifft(u).real)
    ai = np.where(x < 3, True, False)
    y = x[ai]
    v = ifft(u)
    v[ai] = ggb.expand_fft(y, u, 3)
    plt.plot(x, v.real)
#    plots.smoothplot(fft(v), plt)
    plt.show()
    '''
