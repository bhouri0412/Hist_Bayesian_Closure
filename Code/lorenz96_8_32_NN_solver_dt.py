#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:37:56 2022

@author: mohamedazizbhouri
"""

import jax.numpy as np

import os
current_dirs_parent = os.path.dirname(os.getcwd())
fold = current_dirs_parent+'/Results/Hist_results'

from integrate_L96_2t_with_coupling import integrate_L96_2t_with_coupling

K = 8 # Number of globa-scale variables X
J = 32 # Number of local-scale Y variables per single global-scale X variable
F = 15.0 # Focring
b = 10.0 # ratio of amplitudes
c = 10.0 # time-scale ratio
h = 1.0 # Coupling coefficient

nt = 20000 # Number of time steps
si = 0.005 # Sampling time interval
dt = 0.005 # Time step

def s(k, K):
    """A non-dimension coordinate from -1..+1 corresponding to k=0..K"""
    return 2*k/K - 1
k = np.arange(K)
j = np.arange(J * K)
Xinit = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
Yinit = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)

# model spin up
X, Y, t, _ = integrate_L96_2t_with_coupling(Xinit, Yinit, si, nt, F, h, b, c, dt=dt)

#Actual run
X2, Y2, t2, _ = integrate_L96_2t_with_coupling(X[-1,:], Y[-1,:], 4*si, nt//4, F, h, b, c, dt=4*dt)
X, Y, t, _ = integrate_L96_2t_with_coupling(X[-1,:], Y[-1,:], si, nt, F, h, b, c, dt=dt)

np.save(fold+'/X2.npy',X2)
