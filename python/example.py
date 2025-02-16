# -*- coding: utf-8 -*-


from w2 import BFM
from time import time
import numpy as np
import numpy.ma as ma
from scipy.fftpack import dctn, idctn
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (13, 8)
plt.rcParams['image.cmap'] = 'viridis'

from scipy.special import gamma

import pickle

# %% Function definitions

# Initialize Fourier kernel
def initialize_kernel(n1, n2):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    kernel = 2*np.pi* (xx**2 + yy**2)  #2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel[0,0] = 1     # to avoid dividing by zero
    return kernel

# 2d DCT
def dct2(a):
    return dctn(a, norm='ortho')
    
# 2d IDCT
def idct2(a):
    return idctn(a, norm='ortho')


#### Fundamental solution of Laplacian for d=2
##def fund_sol(y):
###    return np.linalg.norm(x) * (1.0 / (   
##    return np.log(np.linalg.norm(y)) / (2 * np.pi)
##
#### inverse Laplacian on torus.
##def inv_laplace(f):
##    n1, n2 = f.shape
##
##    grid = np.linspace(-10,10,10000)
##
##    out = np.empty((n1,n2))
##
##    for i in range(n1):
##        for j in range(n2):
##            out[i,j] = np.sum( fund_sol(grid[k])


# Update phi as 
#       ϕ ← ϕ + σ Δ⁻¹(ρ − ν)
# and return the error 
#       ∫(−Δ)⁻¹(ρ−ν) (ρ−ν)
# Modifies phi and rho

def b_to_k(phi):
    phi = 0.5*(x*x+y*y) - phi
    return None

def update_potential(phi, rho, nu, kernel, sigma):
    n1, n2 = nu.shape

    rho -= nu
    print(rho)
    workspace = np.fft.fft2(rho) 
    omega_x = np.fft.fftfreq(workspace[0].shape[0],d=dx)
    omega_y = np.fft.fftfreq(workspace[1].shape[0],d=dy)

    for i in range(n1):
        for j in range(n2):
            if omega_x[i] == 0 and omega_y[j] == 0:
                workspace[i,j] = 0

            else:                
                workspace[i,j] /= (omega_x[i]**2 + omega_y[j]**2)

#    workspace[0,0] = 0
#    workspace
    iphi = np.real( np.fft.ifft2(workspace,axes=[0,1]))
    #workspace = idct2(workspace)
    print(iphi)
    #phi += sigma * (0.5*(x*x+y*y) - iphi)
    phi += sigma * iphi
    h1 = np.sum(workspace * rho) / (n1*n2)
    
    return h1

# Compute the dual value
# 
#       ∫ (½|x|² − ϕ(x)) ν(x)dx  +  ∫ (½|x|² − ψ(x)) μ(x)dx 
# 
def compute_w2(phi, psi, mu, nu, x, y):
    n1, n2 = mu.shape
    return np.sum(0.5 * (x*x+y*y) * (mu + nu) - nu*phi - mu*psi)/(n1*n2)

# Parameters for Armijo-Goldstein
scaleDown = 0.95
scaleUp   = 1/scaleDown
upper = 0.75
lower = 0.25
# Armijo-Goldstein
def stepsize_update(sigma, value, oldValue, gradSq):
    diff = value - oldValue

    if diff > gradSq * sigma * upper:
        return sigma * scaleUp
    elif diff < gradSq * sigma * lower:
        return sigma * scaleDown
    return sigma

#def max_per(phi,psi):


# Back-and-forth solver
def compute_ot(phi, psi, bf, sigma):

    kernel = initialize_kernel(n1, n2)
    rho = np.copy(mu)

    oldValue = compute_w2(phi, psi, mu, nu, x, y)

    for k in range(numIters+1):

        print(phi)
        gradSq = update_potential(phi, rho, nu, kernel, sigma)
        print("!!!")
        print(phi)
        b_to_k(phi)
        b_to_k(psi)


        print(phi)
        #p_phi,p_psi = max_per(phi,psi)
        bf.ctransform(psi, phi)

        #p_phi,p_psi = max_per(phi,psi)
        bf.ctransform(phi, psi)
    
        print(phi)

        value = compute_w2(phi, psi, mu, nu, x, y)

        sigma = stepsize_update(sigma, value, oldValue, gradSq)
        oldValue = value

        bf.pushforward(rho, phi, nu)

        b_to_k(phi)
        b_to_k(psi)

        gradSq = update_potential(psi, rho, mu, kernel, sigma)

        b_to_k(phi)
        b_to_k(psi)

        #p_phi,p_psi = max_per(phi,psi)
        bf.ctransform(psi, phi)

        #p_phi,p_psi = max_per(phi,psi)
        bf.ctransform(phi, psi)

        bf.pushforward(rho, psi, mu)

        value = compute_w2(phi, psi, mu, nu, x, y)
        sigma = stepsize_update(sigma, value, oldValue, gradSq)
        oldValue = value

        if k % 5 == 0:
            print(f'iter {k:4d},   W2 value: {value:.6e},   H1 err: {gradSq:.2e}')

        b_to_k(phi)
        b_to_k(psi)

# %% Example: Caffarelli's counterexample

# Caffarelli's counterexample illustrates that the optimal map can be discontinous when the target domain is nonconvex.
# Reference: Luis A. Caffarelli. The regularity of mappings with a convex potential. J. Amer. Math. Soc. 5, 1 (1992), 99–104.


# Define the problem data and initial values

# Grid of size n1 x n2
n1 = 128# 1024   # x axis
n2 = 128#1024   # y axis

dx = 1/(n1-1)
dy = 1/(n2-1)

#x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1), np.linspace(0.5/n2,1-0.5/n1,n2))
x, y = np.meshgrid(np.linspace(0,1,n1), np.linspace(0,1,n2))
phi = np.zeros((n1,n2)) #0.5 * (x*x + y*y)
psi = np.zeros((n1,n2))#0.5 * (x*x + y*y)


#np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#
#print(x)
#print(np.roll(x,shift=-n1//2, axis=[0,1]))
#print("\n\n")
#print(y)
#print(np.roll(y,shift=-n1//2, axis=[1,0]))
#print(np.roll(y,shift=n1//2, axis=[0,1]))
#
#
#
#
#sys.exit(0)
#
#rolls_x = [x, np.roll(x,shift=-n1//2, axis=0)]
#rolls_y = [y]
#
##roll1 = 
#


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
