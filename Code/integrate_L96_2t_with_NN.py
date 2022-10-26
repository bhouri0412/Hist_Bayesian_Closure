#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:37:56 2022

@author: mohamedazizbhouri
"""

import jax.numpy as np

def forward_pass_2(Hor, Horh1, Horh2, W_in, num_par_NN, K, L, sigma_X, mu_X):
    num_layers = len(L)
    H = (Hor-mu_X)/sigma_X
    Hh1 = (Horh1-mu_X)/sigma_X
    Hh2 = (Horh2-mu_X)/sigma_X
    Ho = np.concatenate((H[:,0:1], Hh1[:,0:1], Hh2[:,0:1]), axis=1)
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
        Hl = np.concatenate((H[:,kk+1:kk+2], Hh1[:,kk+1:kk+2], Hh2[:,kk+1:kk+2]), axis=1)
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

def forward_pass_3(Hor, Horh1, Horh2, Horh3, W_in, num_par_NN, K, L, sigma_X, mu_X):
    num_layers = len(L)
    H = (Hor-mu_X)/sigma_X
    Hh1 = (Horh1-mu_X)/sigma_X
    Hh2 = (Horh2-mu_X)/sigma_X
    Hh3 = (Horh3-mu_X)/sigma_X
    Ho = np.concatenate((H[:,0:1], Hh1[:,0:1], Hh2[:,0:1], Hh3[:,0:1]), axis=1)
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
        Hl = np.concatenate((H[:,kk+1:kk+2], Hh1[:,kk+1:kk+2], Hh2[:,kk+1:kk+2], Hh3[:,kk+1:kk+2]), axis=1)
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

def forward_pass_4(Hor, Horh1, Horh2, Horh3, Horh4, W_in, num_par_NN, K, L, sigma_X, mu_X):
    num_layers = len(L)
    H = (Hor-mu_X)/sigma_X
    Hh1 = (Horh1-mu_X)/sigma_X
    Hh2 = (Horh2-mu_X)/sigma_X
    Hh3 = (Horh3-mu_X)/sigma_X
    Hh4 = (Horh4-mu_X)/sigma_X
    Ho = np.concatenate((H[:,0:1], Hh1[:,0:1], Hh2[:,0:1], Hh3[:,0:1], Hh4[:,0:1]), axis=1)
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
        Hl = np.concatenate((H[:,kk+1:kk+2], Hh1[:,kk+1:kk+2], Hh2[:,kk+1:kk+2], Hh3[:,kk+1:kk+2], Hh4[:,kk+1:kk+2]), axis=1)
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

def forward_pass_5(Hor, Horh1, Horh2, Horh3, Horh4, Horh5, W_in, num_par_NN, K, L, sigma_X, mu_X):
    num_layers = len(L)
    H = (Hor-mu_X)/sigma_X
    Hh1 = (Horh1-mu_X)/sigma_X
    Hh2 = (Horh2-mu_X)/sigma_X
    Hh3 = (Horh3-mu_X)/sigma_X
    Hh4 = (Horh4-mu_X)/sigma_X
    Hh5 = (Horh5-mu_X)/sigma_X
    Ho = np.concatenate((H[:,0:1], Hh1[:,0:1], Hh2[:,0:1], Hh3[:,0:1], Hh4[:,0:1], Hh5[:,0:1]), axis=1)
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
        Hl = np.concatenate((H[:,kk+1:kk+2], Hh1[:,kk+1:kk+2], Hh2[:,kk+1:kk+2], Hh3[:,kk+1:kk+2], Hh4[:,kk+1:kk+2], Hh5[:,kk+1:kk+2]), axis=1)
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

def forward_pass_10(Hor, Horh1, Horh2, Horh3, Horh4, Horh5, Horh6, Horh7, Horh8, Horh9, Horh10, W_in, num_par_NN, K, L, sigma_X, mu_X):
    num_layers = len(L)
    H = (Hor-mu_X)/sigma_X
    Hh1 = (Horh1-mu_X)/sigma_X
    Hh2 = (Horh2-mu_X)/sigma_X
    Hh3 = (Horh3-mu_X)/sigma_X
    Hh4 = (Horh4-mu_X)/sigma_X
    Hh5 = (Horh5-mu_X)/sigma_X
    Hh6 = (Horh6-mu_X)/sigma_X
    Hh7 = (Horh7-mu_X)/sigma_X
    Hh8 = (Horh8-mu_X)/sigma_X
    Hh9 = (Horh9-mu_X)/sigma_X
    Hh10 = (Horh10-mu_X)/sigma_X
    Ho = np.concatenate((H[:,0:1], Hh1[:,0:1], Hh2[:,0:1], Hh3[:,0:1], Hh4[:,0:1], Hh5[:,0:1],
                         Hh6[:,0:1], Hh7[:,0:1], Hh8[:,0:1], Hh9[:,0:1], Hh10[:,0:1]), axis=1)
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
        Hl = np.concatenate((H[:,kk+1:kk+2], Hh1[:,kk+1:kk+2], Hh2[:,kk+1:kk+2], Hh3[:,kk+1:kk+2], Hh4[:,kk+1:kk+2], Hh5[:,kk+1:kk+2],
                             Hh6[:,kk+1:kk+2], Hh7[:,kk+1:kk+2], Hh8[:,kk+1:kk+2], Hh9[:,kk+1:kk+2], Hh10[:,kk+1:kk+2]), axis=1)
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

def L96_2t_xdot_2(Xt, Xth1, Xth2, W, num_par_NN, K, L, sigma_X, mu_X, F):    
    Xdot = np.roll(Xt, 1, axis = 1) * (np.roll(Xt, -1, axis = 1) - np.roll(Xt, 2, axis = 1)) - Xt + F + forward_pass_2(Xt, Xth1, Xth2, W, num_par_NN, K, L, sigma_X, mu_X) 
    return Xdot
def L96_2t_xdot_3(Xt, Xth1, Xth2, Xth3, W, num_par_NN, K, L, sigma_X, mu_X, F):     
    Xdot = np.roll(Xt, 1, axis = 1) * (np.roll(Xt, -1, axis = 1) - np.roll(Xt, 2, axis = 1)) - Xt + F + forward_pass_3(Xt, Xth1, Xth2, Xth3, W, num_par_NN, K, L, sigma_X, mu_X) 
    return Xdot
def L96_2t_xdot_4(Xt, Xth1, Xth2, Xth3, Xth4, W, num_par_NN, K, L, sigma_X, mu_X, F):     
    Xdot = np.roll(Xt, 1, axis = 1) * (np.roll(Xt, -1, axis = 1) - np.roll(Xt, 2, axis = 1)) - Xt + F + forward_pass_4(Xt, Xth1, Xth2, Xth3, Xth4, W, num_par_NN, K, L, sigma_X, mu_X)    
    return Xdot
def L96_2t_xdot_5(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, W, num_par_NN, K, L, sigma_X, mu_X, F):     
    Xdot = np.roll(Xt, 1, axis = 1) * (np.roll(Xt, -1, axis = 1) - np.roll(Xt, 2, axis = 1)) - Xt + F + forward_pass_5(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, W, num_par_NN, K, L, sigma_X, mu_X)    
    return Xdot
def L96_2t_xdot_10(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10, W, num_par_NN, K, L, sigma_X, mu_X, F):    
    Xdot = np.roll(Xt, 1, axis = 1) * (np.roll(Xt, -1, axis = 1) - np.roll(Xt, 2, axis = 1)) - Xt + F + \
            forward_pass_10(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10, W, num_par_NN, K, L, sigma_X, mu_X)   
    return Xdot

def integrate_L96_2t_with_NN_2(X0, si, nt, params, model, F, t0=0, dt=0.001):
    xhist = []
    X = X0.copy()
    xhist.append(X[0,:])
    for i in range(X.shape[0]-1):
        xhist.append(X[i+1,:])
    ns = 1
    for n in range(nt):
        if n%50 == 0:
            print(n,nt)
        for s in range(ns):
            # RK4 update of X
            Xdot1 = L96_2t_xdot_2(xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            Xdot2 = L96_2t_xdot_2(
                    xhist[-2][None,:] + 0.5 * dt * Xdot1, xhist[-3][None,:], xhist[-5][None,:], params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot3 = L96_2t_xdot_2(
                    xhist[-2][None,:] + 0.5 * dt * Xdot2, xhist[-3][None,:], xhist[-5][None,:], params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot4 = L96_2t_xdot_2(
                    xhist[-2][None,:] + dt * Xdot3, xhist[-2][None,:], xhist[-4][None,:], params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            X = xhist[-2][None,:] + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        xhist.append(X[0,:])
    return np.array(xhist)

def integrate_L96_2t_with_NN_3(X0, si, nt, params, model, F, t0=0, dt=0.001):
    xhist = []
    X = X0.copy()
    xhist.append(X[0,:])
    for i in range(X.shape[0]-1):
        xhist.append(X[i+1,:])
    ns = 1
    for n in range(nt):
        if n%50 == 0:
            print(n,nt)
        for s in range(ns):
            # RK4 update of X
            Xdot1 = L96_2t_xdot_3(xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], xhist[-8][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            Xdot2 = L96_2t_xdot_3(
                    xhist[-2][None,:] + 0.5 * dt * Xdot1, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot3 = L96_2t_xdot_3(
                    xhist[-2][None,:] + 0.5 * dt * Xdot2, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot4 = L96_2t_xdot_3(
                    xhist[-2][None,:] + dt * Xdot3, xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            X = xhist[-2][None,:] + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        xhist.append(X[0,:])
    return np.array(xhist)

def integrate_L96_2t_with_NN_4(X0, si, nt, params, model, F, t0=0, dt=0.001):
    xhist = []
    X = X0.copy()
    xhist.append(X[0,:])
    for i in range(X.shape[0]-1):
        xhist.append(X[i+1,:])
    ns = 1
    for n in range(nt):
        if n%50 == 0:
            print(n,nt)
        for s in range(ns):
            # RK4 update of X
            Xdot1 = L96_2t_xdot_4(xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], xhist[-8][None,:], xhist[-10][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            Xdot2 = L96_2t_xdot_4(
                    xhist[-2][None,:] + 0.5 * dt * Xdot1, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:], xhist[-9][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot3 = L96_2t_xdot_4(
                    xhist[-2][None,:] + 0.5 * dt * Xdot2, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:], xhist[-9][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot4 = L96_2t_xdot_4(
                    xhist[-2][None,:] + dt * Xdot3, xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], xhist[-8][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            X = xhist[-2][None,:] + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        xhist.append(X[0,:])
    return np.array(xhist)

def integrate_L96_2t_with_NN_5(X0, si, nt, params, model, F, t0=0, dt=0.001):
    xhist = []
    X = X0.copy()
    xhist.append(X[0,:])
    for i in range(X.shape[0]-1):
        xhist.append(X[i+1,:])
    ns = 1
    for n in range(nt):
        if n%50 == 0:
            print(n,nt)
        for s in range(ns):
            # RK4 update of X
            Xdot1 = L96_2t_xdot_5(xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], xhist[-8][None,:], xhist[-10][None,:], xhist[-12][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            Xdot2 = L96_2t_xdot_5(
                    xhist[-2][None,:] + 0.5 * dt * Xdot1, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:], xhist[-9][None,:], xhist[-11][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot3 = L96_2t_xdot_5(
                    xhist[-2][None,:] + 0.5 * dt * Xdot2, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:], xhist[-9][None,:], xhist[-11][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            Xdot4 = L96_2t_xdot_5(
                    xhist[-2][None,:] + dt * Xdot3, xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], xhist[-8][None,:], xhist[-10][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F
                    )
            X = xhist[-2][None,:] + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        xhist.append(X[0,:])
    return np.array(xhist)

def integrate_L96_2t_with_NN_10(X0, si, nt, params, model, F, t0=0, dt=0.001):
    xhist = []
    X = X0.copy()
    xhist.append(X[0,:])
    for i in range(X.shape[0]-1):
        xhist.append(X[i+1,:])
    ns = 1
    for n in range(nt):
        if n%50 == 0:
            print(n,nt)
        for s in range(ns):
            # RK4 update of X
            Xdot1 = L96_2t_xdot_10(xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], xhist[-8][None,:], xhist[-10][None,:], 
                                xhist[-12][None,:], xhist[-14][None,:], xhist[-16][None,:], xhist[-18][None,:], xhist[-20][None,:], 
                                xhist[-22][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            Xdot2 = L96_2t_xdot_10(xhist[-2][None,:] + 0.5 * dt * Xdot1, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:], 
                                xhist[-9][None,:], xhist[-11][None,:], xhist[-13][None,:], xhist[-15][None,:], xhist[-17][None,:], 
                                xhist[-19][None,:], xhist[-21][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            Xdot3 = L96_2t_xdot_10(xhist[-2][None,:] + 0.5 * dt * Xdot2, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:], 
                                xhist[-9][None,:], xhist[-11][None,:], xhist[-13][None,:], xhist[-15][None,:], xhist[-17][None,:], 
                                xhist[-19][None,:], xhist[-21][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            Xdot4 = L96_2t_xdot_10(xhist[-2][None,:] + dt * Xdot3, xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], 
                                xhist[-8][None,:], xhist[-10][None,:], xhist[-12][None,:], xhist[-14][None,:], xhist[-16][None,:], 
                                xhist[-18][None,:], xhist[-20][None,:], model.params, model.num_par_NN, model.K, model.L, model.sigma_X, model.mu_X, F)
            X = xhist[-2][None,:] + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        xhist.append(X[0,:])
    return np.array(xhist)
