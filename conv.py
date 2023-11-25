# Convergence!

import numpy as np
from common import *

def pointwise_error(x, exact, f):
    return f(x) - exact(x)

def pointwise_error_interpolate(x, exact, fx)
    
def pointwise_error_fft(x, exact, uh):
    f = lambda x: return ifft_at(x, uh)
    return f(x) - exact(x)

