import numpy as np
from numpy import pi
from numpy.fft import *

xmin = 0
xmax = 2 * pi

def elrk4(SemiGroup,Nonlinear,y0,tinterval,dt,args):
    y = y0
    t, tf = tinterval
    time = [t,]
    solution = [y,]
    
    if (type(SemiGroup)==type([1,2]) or type(SemiGroup)==type((1,2))):
        S_half, S = SemiGroup
    elif type(SemiGroup) == type(Nonlinear):
        S_half, S = SemiGroup(y,t,dt,*args)
    else:
        print('SemiGroup must be either a tuple/list of two numpy arrays or a function that returns\
              a list/tuple of two numpy arrays.')
        print('program terminating.')
        return None
    
    flag = True
    dt = min(dt, tf - t)
    while flag==True and t < tf:
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
        if (t >= tf):
            flag = False
        elif t + dt > tf:
            dt = tf - t
        elif np.any(np.isnan(y)):
            flag = False
        print("t, dt = ", t, dt)

    times = np.array(time)
    solution.append(y)
    return times, solution

def create_filter(k, sigma, **args):
    K = np.max(np.abs(k))
    filter = sigma(k / K, **args)
    return filter

def mask(c, tol=1e-14):
    return c * np.where(np.abs(c)<tol, 0, 1)

def pad(c, m):
    # ASSUME c is fft(u) 
    N = len(c)
    newN = 2*m + N
    r = fftshift(c)
    r = np.r_[np.zeros(m), r, np.zeros(m)]
    r *= newN / N
    r = ifftshift(r)
    return r

def unpad(c, m):
    N = len(c)
    newN = N - 2 * m
    r = fftshift(c)
    r = r[m:m + newN]
    r *= newN / N
    r = ifftshift(r)
    return r

def freqs(n):
    return fftfreq(n, 1.0/n)

def cgrid(n, xmin=0, xmax=2*np.pi):
    dx = (xmax - xmin) / n
    return 0.5 * dx + np.arange(xmin, xmax, dx)

def ifft_at(z, uh):
    '''
    Evaluates the given truncated fourier series
    at each value of x
    '''
    N = len(uh)
    k = freqs(N)
    dx = (xmax-xmin)/N
    dz = z[1] - z[0]
    # Need to shift because the first sample was at dx/2
    # TODO Don't completely understand that yet, but it makes sense
    y = z - dx/2
    xk = np.tensordot(y, k, axes = 0)
    return np.exp(1j * xk)@uh / N

