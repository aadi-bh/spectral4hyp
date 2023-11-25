# Filters!

import numpy as np

def exponential(eta, p=2):
    return np.exp(-15 * np.power(eta, 2*p))

def cesaro(eta, **kwargs):
    r = 1 - np.abs(eta)
    r *= np.where(np.abs(eta) > 1, 0, 1)
    return r

def raisedcos(eta, **kwargs):
    r = 0.5 * (1 + np.cos(np.pi * eta))
    r *= np.where(np.abs(eta) > 1, 0, 1)
    return r

def lanczos(eta, **kwargs):
    r = np.sinc(np.pi * eta)
    return r

def no_filter(eta, **kwargs):
    return np.ones(len(eta))

def apply_filter(u, sigma, **args):
    N = len(u)
    k = freqs(N)
    filter = create_filter(k, sigma, **args)
    return mask(u * filter)
