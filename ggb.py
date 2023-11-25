# Gegenbauer polynomials -- a Gibbs complementary basis to Fourier

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

'''
Weighted inner product over [a,b]
#### f must be function of x\in[ a, b],
#### g must be function of z\in[-1, 1].
'''
def wip(f, g, lam, a, b):
    z, wts = np.polynomial.legendre.leggauss(10*lam + 200)
    x = xi_inv(z, a, b)
    fx = f(x)
    gz = g(z)
    wz = w(z, lam)
    return np.sum(wts * fx * gz * wz)

def ip_fft(uh, n, lam, a, b):
    # f should be a function of x, g must be a function of z
    f = lambda x: ifft_at(x, uh)
    Cn = ggb(n, lam)
    return wip(f, Cn, lam, a, b)

def expand(x, f, L):
    '''
    Expands f over the GGB polys up to degree L
    and returns evaluation at x.
    '''
    degs = np.arange(0, L+1)
    a = x[0]
    b = x[-1]
    m = np.empty((len(degs), len(x)))
    for n in degs:
        Cn = ggb(n, L)
        m[n] = wip(f, Cn, L, a, b).real / gam(n, L) * Cn(xi(x, a, b))
    return np.sum(m, axis = 0)

def expand_fft(x, uh, L):
    '''
    Expands the function (f = ifft(uh)) in terms of the Gegenbauer polys
    and return the evaluation at each x.
    '''
    f = lambda x: ifft_at(x, uh)
    return expand(x, f, L)

# TODO define a function that processes everything
