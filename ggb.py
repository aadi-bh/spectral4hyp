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

def wtd_inner(uh, n, lam, a, b):
    x = cgrid(2*len(uh)*(n+1), a, b)
    z = xi(x, a, b)
    dz = z[1] - z[0]
    fx = ifft_at(x, uh)
    Cz = C(z, n, lam)
    wz = w(z, lam)
    # Trapezoidal rule, but w = 0 at +-1
    return dz * np.sum(fx * Cz * wz)

x = np.linspace(0, 2*pi, 16)
u = np.ones(x.shape)
uh = fft(u)
print(wtd_inner(uh, 0, 1, 1,2) / gam(0,1))
print(wtd_inner(uh, 1, 1, 1,2))
print(wtd_inner(uh, 2, 1, 1,2))
print(wtd_inner(uh, 3, 1, 1,2))
print(wtd_inner(uh, 4, 1, 1,2))
# TODO
# Expand it in terms of the ggbs
# Patch it back into the solution.
