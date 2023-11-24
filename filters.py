# Filters!

import numpy as np

def exponential(eta, p=1):
    return np.exp(-35* np.power(eta, 2*p))

def cesaro(eta):
    return 1 - eta

def raisedcos(eta):
    return 0.5 * (1 + np.cos(np.pi * eta))

def lanczos(eta):
    return np.sin(np.pi * eta) / (np.pi * eta)

def no_filter(eta):
    return 1.0
