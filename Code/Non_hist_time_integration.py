#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:37:56 2022

@author: mohamedazizbhouri
"""

import jax.numpy as np

def forward_pass(Hor, W_in, num_par_NN, K, L, sigma_X, mu_X):
    num_layers = len(L)
    H = (Hor-mu_X)/sigma_X
    Ho = H[:,0:1]
    Wl = W_in[:num_par_NN]
    for k in range(0,num_layers-2):
        W = np.reshape(Wl[0:L[k] * L[k+1]], (L[k], L[k+1]))
        Wl = Wl[L[k] * L[k+1]:]
        b = np.reshape(Wl[0:L[k+1]], (1, L[k+1]))
        Wl = Wl[L[k+1]:]
        Ho = np.tanh(np.add(np.matmul(Ho, W), b))
    W = np.reshape(Wl[0:L[num_layers-2] * L[num_layers-1]], (L[num_layers-2], L[num_layers-1]))
    Wl = Wl[L[num_layers-2] * L[num_layers-1]:]
    b = np.reshape(Wl[0:L[num_layers-1]], (1, L[num_layers-1]))
    Ho = np.add(np.matmul(Ho, W), b)
    for kk in range(K-1):
        Hl = H[:,kk+1:kk+2]
        Wl = W_in[(kk+1)*num_par_NN:(kk+2)*num_par_NN]
        for k in range(0,num_layers-2):
            W = np.reshape(Wl[0:L[k] * L[k+1]], (L[k], L[k+1]))
            Wl = Wl[L[k] * L[k+1]:]
            b = np.reshape(Wl[0:L[k+1]], (1, L[k+1]))
            Wl = Wl[L[k+1]:]
            Hl = np.tanh(np.add(np.matmul(Hl, W), b))
        W = np.reshape(Wl[0:L[num_layers-2] * L[num_layers-1]], (L[num_layers-2], L[num_layers-1]))
        Wl = Wl[L[num_layers-2] * L[num_layers-1]:]
        b = np.reshape(Wl[0:L[num_layers-1]], (1, L[num_layers-1]))
        Hl = np.add(np.matmul(Hl, W), b)
        Ho = np.concatenate((Ho,Hl),axis=1)
    return Ho

def L96_2t_xdot(Xt, W, num_par_NN, K, L, sigma_X, mu_X, F):    
    Xdot = np.roll(Xt, 1, axis = 1) * (np.roll(Xt, -1, axis = 1) - np.roll(Xt, 2, axis = 1)) - Xt + F + forward_pass(Xt, W, num_par_NN, K, L, sigma_X, mu_X)    
    return Xdot

def stepper(Xt, W, num_par_NN, K, L, dt, sigma_X, mu_X, F):
    # RK4 update of Xt
    Xdot1 = L96_2t_xdot(Xt, W, num_par_NN, K, L, sigma_X, mu_X, F)
    Xdot2 = L96_2t_xdot(Xt + 0.5 * dt * Xdot1, W, num_par_NN, K, L, sigma_X, mu_X, F)
    Xdot3 = L96_2t_xdot(Xt + 0.5 * dt * Xdot2, W, num_par_NN, K, L, sigma_X, mu_X, F)
    Xdot4 = L96_2t_xdot(Xt + dt * Xdot3, W, num_par_NN, K, L, sigma_X, mu_X, F)
    Xt = Xt + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
    return Xt

def integrate_L96_2t_with_NN(X0, si, nt, params, model, F, t0=0, dt=0.001):
    xhist = []
    X = X0.copy()
    xhist.append(X[0,:])
    ns = 1
    for n in range(nt):
        if n%50 == 0:
            print(n,nt)
        for s in range(ns):
            Xdot1 = L96_2t_xdot(X, params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            Xdot2 = L96_2t_xdot(
                    X + 0.5 * dt * Xdot1, params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot3 = L96_2t_xdot(
                    X + 0.5 * dt * Xdot2, params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot4 = L96_2t_xdot(
                    X + dt * Xdot3, params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            X = X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        xhist.append(X[0,:])
    return np.array(xhist)