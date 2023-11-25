# Gegenbauer polynomials -- the Gibbs complimentary basis to Fourier

import numpy as np
from scipy.special import gegenbauer as ggb
from scipy.special import gamma, factorial
from common import *

'''
Maps interval [a,b] to [-1,1]
'''
def xi(y, a, b):
    return -1 + 2 * (y-a)/(b-a)

'''
Maps interval [-1, 1] to [a, b]
'''
def xi_inv(z, a, b):
    return (b+a)/2 + z * (b-a)/2

'''
Weight function
'''
def w(x, lam):
    return np.power(1-np.power(x,2), lam-0.5)

'''
Ggb polynomial of order n evaluated at x
'''
def C(x, n, lam):
    return ggb(n, lam)(x)

'''
Square of weighted norm of C(n, lam)
'''
def gam(n, lam):
    r = np.sqrt(np.pi)
    r *= gamma(n + 2*lam) * gamma(lam + 0.5) 
    r /= gamma(lam) * gamma(2 * lam) * factorial(n) * (n+lam)
    return r

def ip(uh, n, lam, a, b):
    z, wts = np.polynomial.legendre.leggauss(10*n+200);
    x = xi_inv(z, a, b)
    fx = ifft_at(x, uh).real
    Cz = C(z, n, lam)
    wz = w(z, lam)
    return np.sum(wts * fx * Cz * wz)

x = np.linspace(0, 2*pi, 16)
u = np.ones(x.shape)
uh = fft(u)
print(ip(uh, 0, 1, 1,2) / gam(0, 1))
print(ip(uh, 1, 1, 1,2) / gam(1, 1))
print(ip(uh, 2, 1, 1,2) / gam(2, 1))
print(ip(uh, 3, 1, 1,2) / gam(3, 1))
print(ip(uh, 4, 1, 1,2) / gam(4, 1))
# TODO
# Expand it in terms of the ggbs
# Patch it back into the solution.
