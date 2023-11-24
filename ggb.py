# Gegenbauer polynomials -- the Gibbs complimentary basis to Fourier

import numpy as np
from scipy.special import gegenbauer as ggb
from scipy.special import gamma, factorial

'''
Maps interval [a,b] to [-1,1]
'''
def z(y, a, b):
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

# TODO
# Need a way to compute the inner product on an arbitrary interval
# Expand it in terms of the ggbs
# Patch it back into the solution.
