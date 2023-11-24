# Initial conditions
import numpy as np
from numpy import pi

def saw_tooth(x):
    left = np.where(x <= pi, 1, 0);
    return x * left + (x - 2 * pi) * (1-left)

def sin(x):
    return np.sin(x)

def step(x):
    y = np.where(x < np.pi, 1, 0) * np.where(x > np.pi/2, 1, 0)
    return y

def bump(x):
    return np.exp(-6 * (x-pi)**2)
