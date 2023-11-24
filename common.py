import numpy as np
from numpy.fft import *

def elrk4(SemiGroup,Nonlinear,y0,tinterval,dt,args):
    y = y0
    t, tf = tinterval
    time = [t,]
    solution = [y,]
    flag = True
    
    if (type(SemiGroup)==type([1,2]) or type(SemiGroup)==type((1,2))):
        S_half, S = SemiGroup
    elif type(SemiGroup) == type(Nonlinear):
        S_half, S = SemiGroup(y,t,dt,*args)
    else:
        print('SemiGroup must be either a tuple/list of two numpy arrays or a function that returns\
              a list/tuple of two numpy arrays.')
        print('program terminating.')
        return None
    
    while flag==True:
        t = t + dt
        k1 = dt*Nonlinear(t, y,*args)
        temp = y + k1/2.0
        temp = S_half.dot(temp)
        
        k2 = dt*Nonlinear(t, temp,*args)
        temp = S_half.dot(y) + k2/2.0
        
        k3 = dt*Nonlinear(t, temp,*args)
        temp = S.dot(y)
        temp = temp + S_half.dot(k3)
        
        k4 = dt*Nonlinear(t, temp,*args)
        temp1 = k4
        
        temp2 = S_half.dot(k3)*2
        temp1 = temp1 + temp2
        
        temp2 = S_half.dot(k2)*2
        temp1 = temp1 + temp2
        
        temp2 = S.dot(k1)
        temp1 = temp1 + temp2
        
        temp = S.dot(y)
        
        y = temp + temp1/6.0
        # solution.append(y)
        
        time.append(t)
        if t + dt > tf:
            dt = tf - t
        elif (t >= tf):
            flag = False
        elif np.any(np.isnan(y)):
            flag = False
        print("t, dt = ", t, dt)

    times = np.array(time)
    solution.append(y)
    return times, solution

def create_filter(k, sigma, args):
    K = np.max(np.abs(k))
    filter = sigma(k / K, **args)
    return filter

def mask(c, tol=1e-14):
    return c * np.where(np.abs(c)<tol, 0, 1)

def plot_resolution(c, ax, kwargs):
    k = fftshift(freqs(len(c)))
    ax.semilogy(k, np.abs(fftshift(c)), **kwargs)
    return

def freqs(n):
    return fftfreq(n, 1./ (n))
