#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:37:56 2022

@author: mohamedazizbhouri
"""

import jax.numpy as np
from jax import jit, random, grad, vmap
from jax.example_libraries import optimizers
from functools import partial
import itertools
from tqdm import tqdm

from integrate_L96_2t_with_coupling import integrate_L96_2t_with_coupling

n_hist = 2
if n_hist == 2:
    from forward_pass import forward_pass_2 as forward_pass
    from stepper import stepper_2 as stepper
elif n_hist == 3:
    from forward_pass import forward_pass_3 as forward_pass
    from stepper import stepper_3 as stepper
elif n_hist == 4:
    from forward_pass import forward_pass_4 as forward_pass
    from stepper import stepper_4 as stepper
elif n_hist == 5:
    from forward_pass import forward_pass_5 as forward_pass
    from stepper import stepper_5 as stepper
elif n_hist == 10:
    from forward_pass import forward_pass_10 as forward_pass
    from stepper import stepper_10 as stepper

import os
current_dirs_parent = os.path.dirname(os.getcwd())
fold = current_dirs_parent+'/Results/Hist_results'

K = 8 # Number of globa-scale variables X
J = 32 # Number of local-scale Y variables per single global-scale X variable
F = 15.0 # Focring
b = 10.0 # ratio of amplitudes
c = 10.0 # time-scale ratio
h = 1.0 # Coupling coefficient
noise = 0.03

nt_pre = 20000 # Number of time steps for model spinup
nt = 20000  # Number of time steps
si = 0.005  # Sampling time interval
dt = 0.005  # Time step

# Initial conditions
def s(k, K):
    """A non-dimension coordinate from -1..+1 corresponding to k=0..K"""
    return 2*k/K - 1
k = np.arange(K)
j = np.arange(J * K)
Xinit = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
Yinit = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)

# Solving true model
X, Y, t2, _ = integrate_L96_2t_with_coupling(Xinit, Yinit, si, nt_pre+nt, F, h, b, c, dt=dt)
X = X[nt_pre:,:]
Y = Y[nt_pre:,:]

# Sub-sampling (tmeporal sparsity)
X = X[::2,:]
dt = dt*2
si = si*2

# Corrupting data with noise
X_train  = X + noise*X.std(0)*random.normal(random.PRNGKey(1234), X.shape)  
np.save(fold+'/X_train', X_train)

# First training routine where we target state at the next time-step
n_fut = 1

# Prepare start and end points given n_hist and n_fut
Xt = []
for i in range(2*n_hist+1+1):
    Xt.append(X_train[i:-2*n_hist-2+i-n_fut+1,:])
Xt = np.transpose(np.array(Xt), (1, 0, 2)) # nt-2*n_hist-1 x 2*n_hist+2 x K
Xtpdt = X_train[2*n_hist+2+n_fut-1:,:] # nt-2*n_hist-1 x K
Ndata = Xt.shape[0]

# MLP NN architecture
L = [n_hist+1, 128, 128, 128, 128, 128, 128, 1]
def get_NN_par_num(L):
    l = len(L)
    NN_par_num = 0
    for k in range(l-1):
        NN_par_num = NN_par_num + L[k] * L[k+1] + L[k+1]
    return NN_par_num 

class ODEfit:
    def __init__(self, Xt, Xtpdt, L, dt, sigma_X, mu_X):  
        self.Xtpdt = Xtpdt
        self.Xt = Xt
        self.K = Xt.shape[2]
        self.dt = 2*dt
        self.sigma_X = sigma_X
        self.mu_X = mu_X
        self.L = L
        self.num_par_NN = get_NN_par_num(L)
        self.key = random.PRNGKey(1234)
        def initialize_NN(layers):      
            # Xavier initialization
            def xavier_init(size):
                in_dim = size[0]
                out_dim = size[1]
                xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
                self.key, _ = random.split(self.key)
                return random.normal(self.key, (in_dim*out_dim,1)) * xavier_stddev   
            num_layers = len(layers)
            l = 0
            W_NN = xavier_init( size=[layers[l], layers[l+1]] )
            W_NN = np.concatenate( ( W_NN, np.zeros((layers[l+1],1)) ) , axis=0 )
            for l in range(1,num_layers-1):
                W_NN = np.concatenate( ( W_NN, xavier_init( size=[layers[l], layers[l+1]] ) ) , axis=0 )
                W_NN = np.concatenate( ( W_NN, np.zeros((layers[l+1],1)) ) , axis=0 )
            return W_NN
        self.params = initialize_NN(self.L)
        for i in range(self.K-1):
            self.params = np.concatenate( (self.params, initialize_NN(self.L)), axis=0)
        # Set optimizer initialization and update functions
        self.learning_rate = 1e-4
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(self.learning_rate)
        self.opt_state = self.opt_init(self.params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch_Xt, batch_Xtpdt, idx):
        batch_Xt = batch_Xt[idx,:,:] # batch_size x 2*n_hist+1 x K
        batch_Xtpdt = batch_Xtpdt[idx,:] # batch_size x K
        pred = stepper(batch_Xt, params, self.num_par_NN, self.K, self.L, self.dt, self.sigma_X, self.mu_X, F)
        loss = np.mean((pred - batch_Xtpdt)**2)
        return loss
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, Xt, Xtpdt, idx):
        params = self.get_params(opt_state)
        g = grad(self.loss, argnums=0)(params, Xt, Xtpdt, idx)
        return self.opt_update(i, g, opt_state)
    # Optimize parameters in a loop
    def train(self, nIter = 10000, batch_size = 64):
        key = random.PRNGKey(1234)
        for it in tqdm(range(nIter)):
            key, _ = random.split(key)
            idx = random.choice(key, self.Xt.shape[0], (batch_size,), replace=False)
            self.opt_state = self.step(next(self.itercount), self.opt_state, self.Xt, self.Xtpdt, idx)            
            if it % 500 == 0:
                self.params = self.get_params(self.opt_state)
                loss_value = self.loss(self.params, self.Xt, self.Xtpdt, idx)
                self.loss_log.append(np.sqrt(loss_value))
                print(it, loss_value)
                print(self.params[0:self.params.shape[0]:self.num_par_NN])
            if it == nIter:
                break
        self.loss_log = np.array(self.loss_log)
        self.params = self.get_params(self.opt_state)

class ODEfit_transfer:
    def __init__(self, Xt, Xtpdt, L, dt, sigma_X, mu_X):  
        self.Xtpdt = Xtpdt
        self.Xt = Xt
        self.K = Xt.shape[2]
        self.dt = 2*dt
        self.sigma_X = sigma_X
        self.mu_X = mu_X
        self.L = L
        self.num_par_NN = get_NN_par_num(L)
        self.key = random.PRNGKey(1234)
        def initialize_NN(layers):      
            # Xavier initialization
            def xavier_init(size):
                in_dim = size[0]
                out_dim = size[1]
                xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
                self.key, _ = random.split(self.key)
                return random.normal(self.key, (in_dim*out_dim,1)) * xavier_stddev   
            num_layers = len(layers)
            l = 0
            W_NN = xavier_init( size=[layers[l], layers[l+1]] )
            W_NN = np.concatenate( ( W_NN, np.zeros((layers[l+1],1)) ) , axis=0 )
            for l in range(1,num_layers-1):
                W_NN = np.concatenate( ( W_NN, xavier_init( size=[layers[l], layers[l+1]] ) ) , axis=0 )
                W_NN = np.concatenate( ( W_NN, np.zeros((layers[l+1],1)) ) , axis=0 )
            return W_NN
        self.params = initialize_NN(self.L)
        for i in range(self.K-1):
            self.params = np.concatenate( (self.params, initialize_NN(self.L)), axis=0)
        # Set optimizer initialization and update functions
        self.learning_rate = 1e-4
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(self.learning_rate)
        self.opt_state = self.opt_init(self.params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch_Xt, batch_Xtpdt, idx):
        batch_Xt = batch_Xt[idx,:,:] 
        batch_Xtpdt = batch_Xtpdt[idx,:]
        pred = stepper(batch_Xt, params, self.num_par_NN, self.K, self.L, self.dt, self.sigma_X, self.mu_X, F)
        batch_Xt = np.concatenate((batch_Xt,pred[:,None,:]),axis=1)
        for iloc in range(n_fut_transfer-1):
            pred = stepper(batch_Xt, params, self.num_par_NN, self.K, self.L, self.dt, self.sigma_X, self.mu_X, F)
            batch_Xt = np.concatenate((batch_Xt,pred[:,None,:]),axis=1)
        loss = np.mean((pred - batch_Xtpdt)**2)
        return loss
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, Xt, Xtpdt, idx):
        params = self.get_params(opt_state)
        g = grad(self.loss, argnums=0)(params, Xt, Xtpdt, idx)
        return self.opt_update(i, g, opt_state)
    # Optimize parameters in a loop
    def train(self, nIter = 10000, batch_size = 64):
        key = random.PRNGKey(1234)
        for it in tqdm(range(nIter)):
            key, _ = random.split(key)
            idx = random.choice(key, self.Xt.shape[0], (batch_size,), replace=False)
            self.opt_state = self.step(next(self.itercount), self.opt_state, self.Xt, self.Xtpdt, idx)            
            if it % 500 == 0:
                self.params = self.get_params(self.opt_state)
                loss_value = self.loss(self.params, self.Xt, self.Xtpdt, idx)
                self.loss_log.append(np.sqrt(loss_value))
                print(it, loss_value)
                print(self.params[0:self.params.shape[0]:self.num_par_NN])
            if it == nIter:
                break
        self.loss_log = np.array(self.loss_log)
        self.params = self.get_params(self.opt_state)

mu_X = np.zeros(X_train.shape[1])
sigma_X = np.max(np.abs(X_train), axis=0)

model = ODEfit(Xt, Xtpdt, L, dt, sigma_X, mu_X)

batch_size = 512
model.train(nIter = 15000, batch_size = batch_size)

# (Second) transfer learning training routine where we target state at the next n_fut_transfer time-step
n_fut_transfer = 5

# Prepare start and end points given n_hist and n_fut_transfer
Xt_transfer = []
for i in range(2*n_hist+1+1):
    Xt_transfer.append(X_train[i:-2*n_hist-2+i-n_fut_transfer+1,:])
Xt_transfer = np.transpose(np.array(Xt_transfer), (1, 0, 2)) # nt-2*n_hist-n_fut_transfer x 2*n_hist+2 x K

Xtpdt_transfer = X_train[2*n_hist+2+n_fut_transfer-1:,:] # nt-2*n_hist-n_fut_transfer x K
Ndata = Xt_transfer.shape[0]

model_transfer = ODEfit_transfer(Xt_transfer, Xtpdt_transfer, L, dt, sigma_X, mu_X)
model_transfer.params = model.params
model_transfer.opt_state = model_transfer.opt_init(model_transfer.params)

model_transfer.train(nIter = 30000, batch_size =  batch_size)
np.save(fold+'/parameters_pre', model_transfer.params)

if n_hist == 2:
    from integrate_L96_2t_with_NN import integrate_L96_2t_with_NN_2 as integrate_L96_2t_with_NN
elif n_hist == 3:
    from integrate_L96_2t_with_NN import integrate_L96_2t_with_NN_3 as integrate_L96_2t_with_NN
elif n_hist == 4:
    from integrate_L96_2t_with_NN import integrate_L96_2t_with_NN_4 as integrate_L96_2t_with_NN
elif n_hist == 5:
    from integrate_L96_2t_with_NN import integrate_L96_2t_with_NN_5 as integrate_L96_2t_with_NN
elif n_hist == 10:
    from integrate_L96_2t_with_NN import integrate_L96_2t_with_NN_10 as integrate_L96_2t_with_NN

compute_results_det = 1
if compute_results_det == 1:
    
    nt = 1000
    X_int, Y_int, t, _ = integrate_L96_2t_with_coupling(X[0,:], Y[0,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    X_int = X_int[::2,:]
    Y_int = Y_int[::2,:]
    t = t[::2]

    X_int2dt = X_int[::2,:]
    Y_int2dt = Y_int[::2,:]
    t_2dt = t[::2]
    Xt_dt = X_int2dt[2:,:]
    Xth1_dt = X_int2dt[1:-1,:]
    Xth2_dt = X_int2dt[:-2,:]
    
    NN_out = forward_pass(Xt_dt, Xth1_dt, Xth2_dt, model_transfer.params[:,0], model_transfer.num_par_NN, model_transfer.K, model_transfer.L, model_transfer.sigma_X, model_transfer.mu_X) 

    Xpred_int = integrate_L96_2t_with_NN(X_int[0:2*n_hist+2,:], si, nt-2*n_hist-1, model_transfer.params, model_transfer, F, 0, 2*dt)
    
    exact_out_int = []
    
    for ii in range(K):
        exact_out = - h*c/b*(Y_int2dt[2:,ii*J+0]+Y_int2dt[2:,ii*J+1]+Y_int2dt[2:,ii*J+2]+Y_int2dt[2:,ii*J+3]+
                             Y_int2dt[2:,ii*J+4]+Y_int2dt[2:,ii*J+5]+Y_int2dt[2:,ii*J+6]+Y_int2dt[2:,ii*J+7]+
                             Y_int2dt[2:,ii*J+8]+Y_int2dt[2:,ii*J+9]+Y_int2dt[2:,ii*J+10]+Y_int2dt[2:,ii*J+11]+
                             Y_int2dt[2:,ii*J+12]+Y_int2dt[2:,ii*J+13]+Y_int2dt[2:,ii*J+14]+Y_int2dt[2:,ii*J+15]+
                             Y_int2dt[2:,ii*J+16]+Y_int2dt[2:,ii*J+17]+Y_int2dt[2:,ii*J+18]+Y_int2dt[2:,ii*J+19]+
                             Y_int2dt[2:,ii*J+20]+Y_int2dt[2:,ii*J+21]+Y_int2dt[2:,ii*J+22]+Y_int2dt[2:,ii*J+23]+
                             Y_int2dt[2:,ii*J+24]+Y_int2dt[2:,ii*J+25]+Y_int2dt[2:,ii*J+26]+Y_int2dt[2:,ii*J+27]+
                             Y_int2dt[2:,ii*J+28]+Y_int2dt[2:,ii*J+29]+Y_int2dt[2:,ii*J+30]+Y_int2dt[2:,ii*J+31])   
        exact_out_int.append(exact_out)

    exact_out_int = np.array(exact_out_int)
    
    ####### Extrap ######
    
    Xpred_init = X[-2-2*n_hist:,:]
    
    X_ext, Y_ext, t_ext, _ = integrate_L96_2t_with_coupling(X[-1,:], Y[-1,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    X_ext = X_ext[::2,:]
    Y_ext = Y_ext[::2,:]
    t_ext = t_ext[::2]
    
    X_ext2dt = X_ext[::2,:]
    Y_ext2dt = Y_ext[::2,:]
    t_2dt_ext = t_ext[::2]
    Xt_dt = X_ext2dt[2:,:]
    Xth1_dt = X_ext2dt[1:-1,:]
    Xth2_dt = X_ext2dt[:-2,:]
    
    NN_out_ext = forward_pass(Xt_dt, Xth1_dt, Xth2_dt, model_transfer.params[:,0], model_transfer.num_par_NN, model_transfer.K, model_transfer.L, model_transfer.sigma_X, model_transfer.mu_X) 

    Xpred_ext = integrate_L96_2t_with_NN(Xpred_init, si, nt, model_transfer.params, model_transfer, F, 0, 2*dt)
    Xpred_ext = Xpred_ext[2*n_hist+1:,:]
    
    exact_out_ext = []
    
    for ii in range(K):
        exact_out = - h*c/b*(Y_ext2dt[2:,ii*J+0]+Y_ext2dt[2:,ii*J+1]+Y_ext2dt[2:,ii*J+2]+Y_ext2dt[2:,ii*J+3]+
                             Y_ext2dt[2:,ii*J+4]+Y_ext2dt[2:,ii*J+5]+Y_ext2dt[2:,ii*J+6]+Y_ext2dt[2:,ii*J+7]+
                             Y_ext2dt[2:,ii*J+8]+Y_ext2dt[2:,ii*J+9]+Y_ext2dt[2:,ii*J+10]+Y_ext2dt[2:,ii*J+11]+
                             Y_ext2dt[2:,ii*J+12]+Y_ext2dt[2:,ii*J+13]+Y_ext2dt[2:,ii*J+14]+Y_ext2dt[2:,ii*J+15]+
                             Y_ext2dt[2:,ii*J+16]+Y_ext2dt[2:,ii*J+17]+Y_ext2dt[2:,ii*J+18]+Y_ext2dt[2:,ii*J+19]+
                             Y_ext2dt[2:,ii*J+20]+Y_ext2dt[2:,ii*J+21]+Y_ext2dt[2:,ii*J+22]+Y_ext2dt[2:,ii*J+23]+
                             Y_ext2dt[2:,ii*J+24]+Y_ext2dt[2:,ii*J+25]+Y_ext2dt[2:,ii*J+26]+Y_ext2dt[2:,ii*J+27]+
                             Y_ext2dt[2:,ii*J+28]+Y_ext2dt[2:,ii*J+29]+Y_ext2dt[2:,ii*J+30]+Y_ext2dt[2:,ii*J+31])
        exact_out_ext.append(exact_out)

    exact_out_ext = np.array(exact_out_ext)

    err_int_det = np.linalg.norm(X_int-Xpred_int) / np.linalg.norm(X_int)
    err_ext_det = np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
    print('Relative interpolation norm det: ',err_int_det)
    print('Relative extrapolation norm det: ',err_ext_det)
    
    err_int_NN_det = np.linalg.norm(exact_out_int-NN_out.T) / np.linalg.norm(exact_out_int)
    err_ext_NN_det = np.linalg.norm(exact_out_ext-NN_out_ext.T) / np.linalg.norm(exact_out_ext)
    print('Relative interpolation norm Closure det: ',err_int_NN_det)
    print('Relative extrapolation norm Closure det: ',err_ext_NN_det)

    np.save(fold+'/X_pred_int_det', Xpred_int)
    np.save(fold+'/X_int_det', X_int)
    np.save(fold+'/NN_int_det', NN_out)
    np.save(fold+'/exact_out_int_det', exact_out_int)
    np.save(fold+'/t_det', t)
    np.save(fold+'/t_2dt_det', t_2dt)
    
    np.save(fold+'/X_pred_ext_det', Xpred_ext)
    np.save(fold+'/X_ext_det', X_ext)
    np.save(fold+'/NN_ext_det', NN_out_ext)
    np.save(fold+'/exact_out_ext_det', exact_out_ext)
    np.save(fold+'/t_ext_det', t_ext)
    np.save(fold+'/t_2dt_ext_det', t_2dt_ext)

