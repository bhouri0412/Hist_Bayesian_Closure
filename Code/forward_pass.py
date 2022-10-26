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
