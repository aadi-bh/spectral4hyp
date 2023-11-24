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


xmin, xmax = 0, 2 * pi

parser = argparse.ArgumentParser()
# parser.add_argument('-N', type=int, help='Number of cells', default=100)
parser.add_argument('--cfl', type=float, help='CFL number', default=0.1)
# parser.add_argument('-scheme',
#                     choices=('C','LF','GLF','LLF','LW','ROE','EROE','GOD'),
#                     help='Scheme', default='LF')
parser.add_argument('--ic',
                    choices=('saw_tooth','sin','step', 'bump'),
                    help='Initial condition', default='sin')
parser.add_argument('--Tf', type=float, help='Final time', default=1.0)
parser.add_argument('--pde', choices=('linadv', 'burgers'), default='burgers')
parser.add_argument('--add_visc', type=bool, default=False)
parser.add_argument('--filter', choices=('no_filter', 'exponential', 'cesaro', 'raisedcos', 'lanczos'), default='no_filter')
parser.add_argument('--filterp', type=int, default=1)
parser.add_argument('--max_lgN', type=int, default=7)
parser.add_argument('--integrator', choices=('solve_ivp', 'elrk4'), default='elrk4')
args = parser.parse_args()

if __name__ == '__main__':
    rhs = op.linadv
    initial_condition = ic.sin
    visc = op.semigroup_none
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
    if args.add_visc == True:
        visc = op.semigroup_heat
    tf = max(0, args.Tf)
    cfl = min(1, args.cfl)

    print("PDE   :", args.pde)
    print("TF    :", tf)
    print("CFL   :", cfl)
    print("IC    :", args.ic)
    print("FILTER:", args.filter)
    print("VISC  :", args.add_visc)
    plotname = f"{args.pde}-visc{str(args.add_visc)}-{tf}-{args.ic}-{args.filter}.png"
    sols = []
    for i in range(0, args.max_lgN - 4 + 1):
        N = np.power(2, i + 4);
        print(f"N={N}")
        M = 3 * N // 2;
        m = M // 2;
        NN = (2 * m) + N
        dx = (xmax-xmin) / N
        dt = (cfl * dx)
        x = cgrid(N)
        k = freqs(N)
        kk = freqs(NN)
        p = 0
        args = (N, M)
        u_hat_init = fft(initial_condition(x)) 
        S_half, S = visc(dt, k, eps = 1e-2)

        times = [0]
        u = [u_hat_init]
        if (tf > 0):
            output = solve_ivp(fun = rhs,
                             t_span = [0, tf],
                             t_eval = [0, tf],
                             y0 = u_hat_init,
                             args = args)
            print(output.message)
            times, u = output.t, output.y.transpose()
        # times, u = elrk4([S_half, S], rhs, u_hat_init, (0, tf), dt, args)

        ## Apply filter
        u *= create_filter(k, sigma, p=p)

        sols.append((times, u))

        filename = f"{N}.txt"
        np.savetxt(filename, np.vstack((k, u[-1].real)))
        print("Saved solution to " + filename)

    # PLOT
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14, 8), width_ratios=[3,1])
    if rhs == op.burgers and initial_condition == ic.sin:
        if tf >= 1 or True:
            god = np.loadtxt("burg3_GOD_5.txt").transpose()
        elif tf == 0.6:
            god = np.loadtxt("burg3_GOD_3.txt").transpose()
        elif tf == 0.2:
            god = np.loadtxt("burg3_GOD_1.txt").transpose()
        ax[0].plot(god[0] * 2 * np.pi, god[1], 'ko', markersize=0.8, label="Godunov flux")

    fig.tight_layout()
    ax[0].grid(visible=True)
    nn = 2048
#    x = np.linspace(xmin, xmax, 2048, endpoint=False)
    for (times, u) in sols:
        v = u[-1]
        t = times[-1]
        n = len(v)
        # ax[0].plot(cgrid(n), ifft(v).real, "+", color= "red", markersize=4)
        plots.smoothplot(v, ax[0], label=str(n)+f", t={np.round(t, 3)}", linewidth=1)
        plots.plot_resolution(v, ax[1], linewidth=0.5, markersize=0.5)

    x = np.linspace(xmin, xmax, 1000)
    ax[0].plot(x, initial_condition(x), linewidth=1, color='k', label="init")

    ax[0].legend()
    fig.savefig(plotname)

    plt.close()
    print("Done.")
