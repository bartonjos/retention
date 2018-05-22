#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:24:06 2018

@author: bartonjo

Test script to numerically solve the trap-filling diffusion ODE model.
"""


import numpy as np
from scipy.special import erfc
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


# define the system of odes and bcs as functions
def fun(x, y):
    f = -2*x*y[1]/D * (1 + R/(1-y[0]+tol)**2)
    return np.vstack((y[1], f))

def bc(ya,yb):
    return np.array([ya[0]-1, yb[0]])


# define the initial solution mesh and the initial guess for y
D = 5.72e-12  # for deuterium in W at 150 C
R = 2880.  # for 5 traps .5eV and 225ppm   .15   # for 20 ppm 1eV trap 
tol = 1e-10
xmax = 4. * np.sqrt(D/(1+R))
x = np.linspace(0, xmax, 5)

y_a = np.zeros((2, x.size))  # 2 rows b/c y_guess and y_guess'
y_a[0,:] = erfc(x*np.sqrt((1+R)/D))
y_a[1,:] = -2*np.sqrt((1+R)/np.pi/D) * np.exp(-x**2/(D/(1+R)))


# solve the bvp
res_a = solve_bvp(fun, bc, x, y_a)


# interpolate the solution on more grid points
x_plot = np.linspace(0, xmax, 100)
y_plot_a = res_a.sol(x_plot)[0]


# plot solutions
plt.figure()
plt.plot(x_plot[:-1]*1e6*2*np.sqrt(40), y_plot_a[:-1]*1,
         '-*', label='y_a')
#plt.legend()
plt.yscale('log')
#plt.ylim([1e-5,3])
plt.xlabel('x [microns]', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.tick_params(labelsize=18)
plt.subplots_adjust(bottom=.14, left=.16)
plt.show()




## bonus
#Nw = 6.31e28
#Ctio = 5*225e-6*Nw
#Ai = R*Ctio
#zeta = x_plot[:-1]
#Ci = y_plot_a[:-1]
#dCs = Ai/(1-Ci+tol)**2*np.gradient(Ci)/np.gradient(zeta)
#dz = zeta[3]-zeta[2]
#Ds = 2*np.sum(zeta[1:]*dCs[1:])*dz/dCs  # diffusivity at zeta[1]
