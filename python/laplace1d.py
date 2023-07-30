# -*- coding: utf-8 -*-


from w2 import BFM
from time import time
import numpy as np
import numpy.ma as ma
from scipy.fftpack import dctn, idctn
from scipy.fft import fft2,ifft2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (13, 8)
plt.rcParams['image.cmap'] = 'viridis'

from scipy.special import gamma

import pickle

# %% Function definitions

# Initialize Fourier kernel
def initialize_kernel(n1, n2):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel[0,0] = 1     # to avoid dividing by zero
    return kernel

# 2d DCT
def dct2(a):
    return dctn(a, norm='ortho')
    
# 2d IDCT
def idct2(a):
    return idctn(a, norm='ortho')

# Grid of size n1 x n2
n1 = 1024   # x axis
n2 = 1024   # y axis

#x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1), np.linspace(0.5/n2,1-0.5/n1,n2))
#x, y = np.meshgrid(np.linspace(i-,1,n1), np.linspace(0,1,n2))
x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1), np.linspace(0.5/n2,1-0.5/n1,n2))

xx, yy = (x,y)#np.meshgrid(np.linspace(-np.pi,np.pi,n1), np.linspace(-np.pi,np.pi,n2))

phi = np.empty((n1,n2))
for i in range(n1):
    for j in range(n2):
        v = (x[i,j]-0.5)**2 + (y[i,j]-0.5)**2

        if v < 1:
            phi[i,j] = np.exp(-1 / (1- v   )) -1 #np.cos(x) + np.sin(y)

        else:
            phi[i,j] = 0 -1
#psi = 0.5 * (x*x + y*y)

N = 100
xmin = 0; xmax = 1
Lx = xmax - xmin
dx = Lx/(N-1)

x = np.linspace(xmin,xmax,N)
omega = 2*np.pi*np.fft.fftfreq(N,d=dx)

#phi = -16*np.pi*np.pi*np.sin(4*np.pi*x) 
phi = -4*(np.pi**2)*np.cos(2*np.pi*x)

fphi = np.fft.fft(phi)
lapl = np.empty(N)
for i in range(len(x)):
    if omega[i] == 0:
        lapl[i] = 0

    else:
        lapl[i] = fphi[i] / (omega[i]**2)

iphi = np.fft.ifft(lapl)
print(iphi)
out = np.real(iphi)
print(out)
print(omega)
import matplotlib.pyplot as plt
plt.plot(x,out)
#plt.plot(x,np.sin(4*np.pi*x))
plt.show()

print("##")

sys.exit(0)






phi = np.cos(2*np.pi*x) + np.sin(2*np.pi*y)

print(phi)

fphi = fft2(phi)
#print(ifft2(fphi))

lapl = fphi / (((x**2 + y**2)*128*4*np.pi*np.pi)**2)
lapl[0,0] = 0

#print(fphi)

iphi = ifft2(lapl)

print(iphi)

new_phi = np.real(iphi)
#new_phi = iphi

import matplotlib.pyplot as plt
plt.imshow(phi)
plt.show()
plt.imshow(new_phi)
plt.show()


"""
rho -= nu
workspace = dct2(rho) / kernel
workspace[0,0] = 0
workspace = idct2(workspace)

phi += sigma * workspace
h1 = np.sum(workspace * rho) / (n1*n2)

return h1

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

print(x)
print(np.roll(x,shift=-n1//2, axis=[0,1]))
print("\n\n")
print(y)
print(np.roll(y,shift=-n1//2, axis=[1,0]))
print(np.roll(y,shift=n1//2, axis=[0,1]))




sys.exit(0)

rolls_x = [x, np.roll(x,shift=-n1//2, axis=0)]
rolls_y = [y]

#roll1 = 



# Initialize densities

def peaks(z):
    return 1 + np.cos(2*np.pi*(z[0])) + np.cos(2*np.pi*(z[1])) /4.0

mu = np.zeros((n2, n1))

for i in range(n2):
    for j in range(n1):
        mu[i,j] = peaks([x[i,j],y[i,j]])

#r = 0.125
#mu[(x-0.5)**2 + (y-0.5)**2 < r**2] = 1
nu = np.ones((n2, n1))
#idx = (((x-0.25)**2 + (y-0.5)**2 < r**2) & (x < 0.25) ) 
#idx = idx | (((x-0.75)**2 + (y-0.5)**2 < r**2) & (x > 0.75))
#idx = idx | ((x < 0.751) & (x > 0.249) & (y < 0.51) & (y > 0.49))
#nu[idx] = 1

# Normalize
mu *= n1*n2 / np.sum(mu)
nu *= n1*n2 / np.sum(nu)


# Plot mu and nu
fig, ax = plt.subplots(1, 2)
ax[0].imshow(mu, origin='lower', extent=(0,1,0,1))
ax[0].set_title("$\\mu$")
ax[1].imshow(nu, origin='lower', extent=(0,1,0,1))
ax[1].set_title("$\\nu$");


# %% Run the back-and-forth solver

# Number of iterations for BFM
numIters = 5#50

# Initial step size
sigma = 4/np.maximum(mu.max(), nu.max())

tic = time()

# Initialize BFM method
bf = BFM(n1, n2, mu)
compute_ot(phi, psi, bf, sigma)

toc = time()
print(f'\nElapsed time: {toc-tic:.2f}s')



# %% Visualizations


my, mx = ma.masked_array(np.gradient(psi-0.5*(x*x+y*y), 1/n2, 1/n1), mask=((mu==0), (mu==0)))

fig, ax = plt.subplots()
ax.contourf(x, y, mu+nu)
ax.set_aspect('equal')
skip = (slice(None,None,n1//16), slice(None,None,n2//16))
ax.quiver(x[skip], y[skip], mx[skip], my[skip], color='yellow', angles='xy', scale_units='xy', scale=1);

# %% 
# The discontinuity of the optimal map is hard to see as a quiver plot. So let's instead display only the x-component of the map.

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x + mx, origin='lower', extent=(0,1,0,1), cmap='plasma')

x_masked = ma.masked_array(x, mask=(nu==0))
ax[1].imshow(x_masked, origin='lower', extent=(0,1,0,1), cmap='plasma')



# %% Displacement interpolation

# Plotting interpolation
def plot_interpolation(mu, nu, phi, psi, n_fig=6):
    fig, ax = plt.subplots(1, n_fig, figsize=(20,8))
    [axi.axis('off') for axi in ax.ravel()]
    vmax = mu.max()
    ax[0].imshow(mu, vmax=vmax)
    ax[0].set_title("$t=0$")
    ax[n_fig-1].imshow(nu, vmax=vmax)
    ax[n_fig-1].set_title("$t=1$")

    interpolate = np.zeros_like(mu)
    rho_fwd = np.zeros_like(mu)
    rho_bwd = np.zeros_like(mu)

    for i in range(1,n_fig-1):
        t = i / (n_fig - 1)
        psi_t = (1-t) * 0.5 * (x*x + y*y) + t * psi
        phi_t = t * 0.5 * (x*x + y*y) + (1-t) * phi

        bf.pushforward(rho_fwd, psi_t, mu)
        bf.pushforward(rho_bwd, phi_t, nu)
        interpolate = (1-t) * rho_fwd + t * rho_bwd
        ax[i].imshow(interpolate, vmax=vmax)  
        ax[i].set_title(f"$t={i}/{n_fig-1}$")

plot_interpolation(mu, nu, phi, psi)
plt.show()

"""
