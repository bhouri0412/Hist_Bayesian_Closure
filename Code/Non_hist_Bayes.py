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
from Non_hist_time_integration import forward_pass, stepper, integrate_L96_2t_with_NN

import os
current_dirs_parent = os.path.dirname(os.getcwd())
fold = current_dirs_parent+'/Results/Non_Hist_results'

K = 8 # Number of globa-scale variables X
J = 32 # Number of local-scale Y variables per single global-scale X variable
F = 20.0 # Focring
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

# Prepare start and end points for first training routine where we target state at the next time-step
Xt = X_train[:-1,:]
Xtpdt = X_train[1:,:]
Ndata = Xt.shape[0]

# MLP NN architecture
L = [1, 128, 128, 128, 128, 128, 128, 1]
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
        self.K = Xt.shape[1]
        self.dt = dt
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
        self.learning_rate = 1e-4 # 1e-4 # 1e-3 # 3e-2
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(self.learning_rate)
        self.opt_state = self.opt_init(self.params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch_Xt, batch_Xtpdt, idx):
        batch_Xt = batch_Xt[idx,:] # batch_size x K
        batch_Xtpdt = batch_Xtpdt[idx,:] # batch_size x n_fut x K
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
        self.K = Xt.shape[1]
        self.dt = dt
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
        self.learning_rate = 1e-4 # 1e-4 # 1e-3 # 3e-2
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(self.learning_rate)
        self.opt_state = self.opt_init(self.params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch_Xt, batch_Xtpdt, idx):
        batch_Xt = batch_Xt[idx,:] # batch_size x K
        batch_Xtpdt = batch_Xtpdt[idx,:] # batch_size x n_fut x K
        pred = stepper(batch_Xt, params, self.num_par_NN, self.K, self.L, self.dt, self.sigma_X, self.mu_X, F)
        for iloc in range(n_fut_transfer-1):
            pred = stepper(pred, params, self.num_par_NN, self.K, self.L, self.dt, self.sigma_X, self.mu_X, F)
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
n_fut_transfer = 4

# Prepare start and end points given n_hist and n_fut_transfer
Xt_reinf = X_train[:-n_fut_transfer,:] # nt-n_fut+1 x K
Xtpdt_reinf = X_train[n_fut_transfer:,:]
Ndata = Xt_reinf.shape[0]

model_transfer = ODEfit_transfer(Xt_reinf, Xtpdt_reinf, L, dt, sigma_X, mu_X)
model_transfer.params = model.params
model_transfer.opt_state = model_transfer.opt_init(model_transfer.params)

model_transfer.train(nIter = 30000, batch_size =  batch_size)
np.save(fold+'/parameters_pre', model_transfer.params)

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
    
    NN_out = forward_pass(X_int2dt, model_transfer.params[:,0], model_transfer.num_par_NN, model_transfer.K, model_transfer.L, model_transfer.sigma_X, model_transfer.mu_X) 

    Xpred_int = integrate_L96_2t_with_NN(X[0:1,:], si, nt, model_transfer.params, model_transfer, F, 0, dt)
    
    exact_out_int = []
    
    for ii in range(K):
        exact_out = - h*c/b*(Y_int2dt[:,ii*J+0]+Y_int2dt[:,ii*J+1]+Y_int2dt[:,ii*J+2]+Y_int2dt[:,ii*J+3]+
                             Y_int2dt[:,ii*J+4]+Y_int2dt[:,ii*J+5]+Y_int2dt[:,ii*J+6]+Y_int2dt[:,ii*J+7]+
                             Y_int2dt[:,ii*J+8]+Y_int2dt[:,ii*J+9]+Y_int2dt[:,ii*J+10]+Y_int2dt[:,ii*J+11]+
                             Y_int2dt[:,ii*J+12]+Y_int2dt[:,ii*J+13]+Y_int2dt[:,ii*J+14]+Y_int2dt[:,ii*J+15]+
                             Y_int2dt[:,ii*J+16]+Y_int2dt[:,ii*J+17]+Y_int2dt[:,ii*J+18]+Y_int2dt[:,ii*J+19]+
                             Y_int2dt[:,ii*J+20]+Y_int2dt[:,ii*J+21]+Y_int2dt[:,ii*J+22]+Y_int2dt[:,ii*J+23]+
                             Y_int2dt[:,ii*J+24]+Y_int2dt[:,ii*J+25]+Y_int2dt[:,ii*J+26]+Y_int2dt[:,ii*J+27]+
                             Y_int2dt[:,ii*J+28]+Y_int2dt[:,ii*J+29]+Y_int2dt[:,ii*J+30]+Y_int2dt[:,ii*J+31])
        exact_out_int.append(exact_out)
        
    exact_out_int = np.array(exact_out_int)
    
    ####### Extrap ######
    
    X_ext, Y_ext, t_ext, _ = integrate_L96_2t_with_coupling(X[-1,:], Y[-1,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    X_ext = X_ext[::2,:]
    Y_ext = Y_ext[::2,:]
    t_ext = t_ext[::2]
    
    X_ext2dt = X_ext[::2,:]
    Y_ext2dt = Y_ext[::2,:]
    t_2dt_ext = t_ext[::2]
    
    NN_out_ext = forward_pass(X_ext2dt, model_transfer.params[:,0], model_transfer.num_par_NN, model_transfer.K, model_transfer.L, model_transfer.sigma_X, model_transfer.mu_X) 

    Xpred_ext = integrate_L96_2t_with_NN(X_ext[0:1,:], si, nt, model_transfer.params, model_transfer, F, 0, dt)
    
    exact_out_ext = []
    
    for ii in range(K):
        exact_out = - h*c/b*(Y_ext2dt[:,ii*J+0]+Y_ext2dt[:,ii*J+1]+Y_ext2dt[:,ii*J+2]+Y_ext2dt[:,ii*J+3]+
                             Y_ext2dt[:,ii*J+4]+Y_ext2dt[:,ii*J+5]+Y_ext2dt[:,ii*J+6]+Y_ext2dt[:,ii*J+7]+
                             Y_ext2dt[:,ii*J+8]+Y_ext2dt[:,ii*J+9]+Y_ext2dt[:,ii*J+10]+Y_ext2dt[:,ii*J+11]+
                             Y_ext2dt[:,ii*J+12]+Y_ext2dt[:,ii*J+13]+Y_ext2dt[:,ii*J+14]+Y_ext2dt[:,ii*J+15]+
                             Y_ext2dt[:,ii*J+16]+Y_ext2dt[:,ii*J+17]+Y_ext2dt[:,ii*J+18]+Y_ext2dt[:,ii*J+19]+
                             Y_ext2dt[:,ii*J+20]+Y_ext2dt[:,ii*J+21]+Y_ext2dt[:,ii*J+22]+Y_ext2dt[:,ii*J+23]+
                             Y_ext2dt[:,ii*J+24]+Y_ext2dt[:,ii*J+25]+Y_ext2dt[:,ii*J+26]+Y_ext2dt[:,ii*J+27]+
                             Y_ext2dt[:,ii*J+28]+Y_ext2dt[:,ii*J+29]+Y_ext2dt[:,ii*J+30]+Y_ext2dt[:,ii*J+31])
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

class ODEfit_HMC:
    def __init__(self, Xt, Xtpdt, L, dt, sigma_X, mu_X):  
        self.Xtpdt = Xtpdt
        self.Xt = Xt
        self.K = Xt.shape[1]
        self.dt = dt
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
        self.learning_rate = 1e-4 # 1e-4 # 1e-3 # 3e-2
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(self.learning_rate)
        self.opt_state = self.opt_init(self.params)
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch_Xt, batch_Xtpdt, idx):
        batch_Xt = batch_Xt[idx,:] # batch_size x K
        batch_Xtpdt = batch_Xtpdt[idx,:] # batch_size x n_fut x K
        pred = stepper(batch_Xt, params, self.num_par_NN, self.K, self.L, self.dt, self.sigma_X, self.mu_X, F)
        loss = np.mean((pred - batch_Xtpdt)**2)
        return loss
    def compute_gradients_and_update(self, batch_size_HMC, key):
        idx = random.choice(key, self.Xt.shape[0], (batch_size_HMC,), replace=False)
        self.params = self.get_params(self.opt_state)
        loss_value = self.loss(self.params, self.Xt, self.Xtpdt, idx)
        g = grad(self.loss, argnums=0)(self.params, self.Xt, self.Xtpdt, idx)
        return loss_value, g[:,0]
    
model_HMC = ODEfit_HMC(Xt, Xtpdt, L, dt, sigma_X, mu_X)

model_HMC.params = model_transfer.params
model_HMC.opt_state = model_HMC.opt_init(model_HMC.params)
Ndata = model_HMC.Xt.shape[0]

batch_size_HMC_original = Ndata
batch_size_HMC = Ndata 
niters_HMC_pre = 0 # number of HMC burnouts
niters_HMC = 4000 # number of HMC sampling steps that are kept
L = 10 # leap frog step number
parameters = [] # book keeping the parameters
loggammalist = [] # book keeping the loggamma
loglambdalist = [] # book keeping the loggamma
loglikelihood = [] # book keeping the loggamma
p_list = [] # book keeping HMC decisions
model_HMC.key = random.PRNGKey(1234)

# HMC chain estimates settings
N_samples = niters_HMC
N_total = niters_HMC # last N_total of HMC are considered to get the N_samples samples
N_MAP = niters_HMC # last N_MAP of HMC are considered to look for MAP
N_end = niters_HMC # the first N_end of HMC are considered for all subsequent results and plots

# initial weight
w_temp = model_HMC.params[:,0]
print("initial_w", w_temp.shape)
model_HMC.key, _ = random.split(model_HMC.key)
loggamma_temp = 4. + random.normal(model_HMC.key)
model_HMC.key, _ = random.split(model_HMC.key)
loglambda_temp = random.normal(model_HMC.key)

loss_original, _ = model_HMC.compute_gradients_and_update(batch_size_HMC_original, model_HMC.key) # compute the initial Hamiltonian
loggamma_temp = np.log(batch_size_HMC_original / loss_original)
print("This is initial guess", loggamma_temp, "with loss", loss_original)
epsilon = 0.0005 # leap frog step size
epsilon_max = epsilon 
epsilon_min = epsilon
loggamma_temp_origin = loggamma_temp
if loggamma_temp > 6.:
    loggamma_temp = 6.
    epsilon_max = epsilon
    epsilon_min = epsilon
loggamma_temp_init = loggamma_temp

def kinetic_energy(V, loggamma_v, loglambda_v):
    q = (np.sum(-V**2) - loggamma_v**2 - loglambda_v**2)/2.0
    return q

def compute_gradient_param(dWeights, weights, loggamma, loglambda, para_num):
    dWeights = np.exp(loggamma)/2.0 * dWeights + np.exp(loglambda) * np.sign(weights)
    return dWeights

def compute_gradient_hyper(loss, weights, loggamma, loglambda, batch_size_HMC, para_num):
    grad_loggamma = np.exp(loggamma) * (loss/2.0 + 1.0) - (batch_size_HMC/2.0 + 1.0)
    grad_loglambda = np.exp(loglambda) * (np.sum(np.abs(weights)) + 1.0) - (para_num + 1.0)
    return grad_loggamma, grad_loglambda

def compute_Hamiltonian(loss, weights, loggamma, loglambda, batch_size_HMC, para_num):
    H = np.exp(loggamma)*(loss/2.0 + 1.0) + np.exp(loglambda)*(np.sum(np.abs(weights)) + 1.0)\
    - (batch_size_HMC/2.0 + 1.0) * loggamma - (para_num + 1.0) * loglambda
    return H

def leap_frog(v_in, w_in, loggamma_in, loglambda_in, loggamma_v_in, loglambda_v_in):
    # assign weights from the previous step to the model
    model_HMC.params = w_in[:,None]
    model_HMC.opt_state = model_HMC.opt_init(model_HMC.params)
    
    # leap frog updates
    v_new = v_in
    loggamma_v_new = loggamma_v_in
    loglambda_v_new = loglambda_v_in

    loggamma_new = loggamma_in
    loglambda_new = loglambda_in
    w_new = w_in

    for m in range(L):
        # First half of the leap frog
        model_HMC.key, _ = random.split(model_HMC.key)
        loss, dWeights = model_HMC.compute_gradients_and_update(batch_size_HMC, model_HMC.key) # evaluate the gradient
        
        dWeights = compute_gradient_param(dWeights, model_HMC.get_params(model_HMC.opt_state)[:,0], loggamma_new, loglambda_new, model_HMC.K*model_HMC.num_par_NN)
        grad_loggamma, grad_loglambda = compute_gradient_hyper(loss, w_new, loggamma_new, loglambda_new, batch_size_HMC, model_HMC.K*model_HMC.num_par_NN)
        
        loggamma_v_new = loggamma_v_new - epsilon/2*grad_loggamma
        loglambda_v_new = loglambda_v_new - epsilon/2*grad_loglambda
        v_new = v_new - epsilon/2*(dWeights)
        w_new = model_HMC.get_params(model_HMC.opt_state)[:,0] + epsilon * v_new
        
        model_HMC.params = w_new[:,None]
        model_HMC.opt_state = model_HMC.opt_init(model_HMC.params)
        
        loggamma_new = loggamma_new + epsilon * loggamma_v_new
        loglambda_new = loglambda_new + epsilon * loglambda_v_new
          
        # Second half of the leap frog
        model_HMC.key, _ = random.split(model_HMC.key)
        loss, dWeights = model_HMC.compute_gradients_and_update(batch_size_HMC, model_HMC.key)
        dWeights = compute_gradient_param(dWeights, model_HMC.get_params(model_HMC.opt_state)[:,0], loggamma_new, loglambda_new, model_HMC.K*model_HMC.num_par_NN)
        grad_loggamma, grad_loglambda = compute_gradient_hyper(loss, w_new, loggamma_new, loglambda_new, batch_size_HMC, model_HMC.K*model_HMC.num_par_NN)

        v_new = v_new - epsilon/2*(dWeights)
        loggamma_v_new = loggamma_v_new - epsilon/2*grad_loggamma
        loglambda_v_new = loglambda_v_new - epsilon/2*grad_loglambda

        return v_new, w_new, loggamma_new, loglambda_new, loggamma_v_new, loglambda_v_new
    
def compute_epsilon(step):
    coefficient = np.log(epsilon_max/epsilon_min)
    return epsilon_max * np.exp( - step * coefficient / (niters_HMC+niters_HMC_pre))

# training steps
for step in range(niters_HMC+niters_HMC_pre):
    epsilon = compute_epsilon(step)

    model_HMC.key, _ = random.split(model_HMC.key)
    v_initial = random.normal(model_HMC.key, (model_HMC.K*model_HMC.num_par_NN,)) # np.random.randn(tot_num_param,) # initialize the velocity
    model_HMC.key, _ = random.split(model_HMC.key)
    loggamma_v_initial = random.normal(model_HMC.key)   
    model_HMC.key, _ = random.split(model_HMC.key)
    loglambda_v_initial = random.normal(model_HMC.key)   
    
    model_HMC.key, _ = random.split(model_HMC.key)
    loss_initial, _ = model_HMC.compute_gradients_and_update(batch_size_HMC, model_HMC.key) # compute the initial Hamiltonian
    loss_initial = compute_Hamiltonian(loss_initial, w_temp, loggamma_temp, loglambda_temp, batch_size_HMC, model_HMC.K*model_HMC.num_par_NN)

    v_new, w_new, loggamma_new, loglambda_new, loggamma_v_new, loglambda_v_new = \
        leap_frog(v_initial, w_temp, loggamma_temp, loglambda_temp, loggamma_v_initial, loglambda_v_initial)
    # compute the final Hamiltonian
    model_HMC.key, _ = random.split(model_HMC.key)
    loss_finial, _ = model_HMC.compute_gradients_and_update(batch_size_HMC, model_HMC.key)
    loss_finial = compute_Hamiltonian(loss_finial, w_new, loggamma_new, loglambda_new, batch_size_HMC, model_HMC.K*model_HMC.num_par_NN)

    # making decisions
    p_temp = np.exp(-loss_finial + loss_initial + \
                    kinetic_energy(v_new, loggamma_v_new, loglambda_v_new) - kinetic_energy(v_initial, loggamma_v_initial, loglambda_v_initial))

    p = min(1, p_temp)
    model_HMC.key, _ = random.split(model_HMC.key)
    p_decision = random.uniform(model_HMC.key)  
    if step >= niters_HMC_pre:
        p_list.append(p > p_decision)
    if p > p_decision:
        w_temp = w_new
        if step >= niters_HMC_pre:
            parameters.append(w_new)
            loggammalist.append(loggamma_new)
            loglambdalist.append(loglambda_new)
            loglikelihood.append(loss_finial)
        loggamma_temp = loggamma_new
        loglambda_temp = loglambda_new
    else:
        model_HMC.params = w_temp[:,None]
        model_HMC.opt_state = model_HMC.opt_init(model_HMC.params)
        if step >= niters_HMC_pre:
            parameters.append(w_temp)
            loggammalist.append(loggamma_temp)
            loglambdalist.append(loglambda_temp)
            loglikelihood.append(loss_initial)
        
    if step % 20 == 0:
        print(step)
        print('probability', p)
        print(p > p_decision)

p_list = np.array(p_list)
np.save(fold+'/p_list', p_list)

parameters = np.array(parameters)
loggammalist = np.array(loggammalist)
loglikelihood = np.array(loglikelihood)
loglambdalist = np.array(loglambdalist)
     
#### Trajectory plots below ####

integrate_L96_2t_with_NN_v = vmap(integrate_L96_2t_with_NN, in_axes = (None, None, None, 0, None, None, None, None))
forward_pass_v = vmap(forward_pass, in_axes = (None, 0, None, None, None, None, None))

compute_results_Bayes = 1
if compute_results_Bayes == 1:
    
    nt = 1000
    X_int, Y_int, t, _ = integrate_L96_2t_with_coupling(X[0,:], Y[0,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    X_int = X_int[::2,:]
    Y_int = Y_int[::2,:]
    t = t[::2]

    sigma_X_HMC = np.std(X_train, axis=0)
    parameters = parameters[:N_end,:]
    precision = loggammalist[:N_end]
    precision_theta = loglambdalist[:N_end]
    loglikelihood = loglikelihood[:N_end]
    precision = np.exp(precision) * K
    precision_theta = np.exp(precision_theta) * model_HMC.K*model_HMC.num_par_NN
    print("precision", precision[-1])
    
    loglikelihood = loglikelihood[-N_MAP:]
    idx_MAP = np.argmin(loglikelihood)
    print(idx_MAP, "index")
    
    MAP = parameters[idx_MAP, :]
    
    X_int2dt = X_int[::2,:]
    Y_int2dt = Y_int[::2,:]
    t_2dt = t[::2]
    
    Xpred_int = integrate_L96_2t_with_NN_v(X_int[0:1,:], si, nt, parameters, model_transfer, F, 0, dt)
    Xpred_int_MAP = Xpred_int[idx_MAP,:,:]
    
    precision_X = np.repeat(precision[:,None], Xpred_int.shape[1], axis = 1) # N_HMC x nt
    precision_X = np.repeat(precision_X[:,:,None], Xpred_int.shape[2], axis = 2) # N_HMC x nt x K
    Sigma_data = np.ones_like(Xpred_int) / np.sqrt(precision_X) # N_HMC x nt x K
    
    model_HMC.key, _ = random.split(model_HMC.key)
    rr = np.repeat( random.normal(model_HMC.key, (precision.shape[0],1)) , Xpred_int.shape[1], axis=1) # N_HMC x nt 
    rr = np.repeat(rr[:,:,None], Xpred_int.shape[2], axis = 2) # N_HMC x nt x K
    
    Xpred_int = Xpred_int + (Sigma_data * sigma_X_HMC) * rr
    
    NN_int = forward_pass_v(X_int2dt, parameters, model_transfer.num_par_NN, model_transfer.K, model_transfer.L, model_transfer.sigma_X, model_transfer.mu_X) 
    NN_MAP = NN_int[idx_MAP,:,:]
    
    mu_pred = np.mean(Xpred_int, axis = 0)
    Sigma_pred = np.var(Xpred_int, axis = 0)
    
    NN_mu = np.mean(NN_int, axis = 0)
    NN_Sigma = np.var(NN_int, axis = 0)
    
    exact_out_int = []
    
    for ii in range(K):
        exact_out = - h*c/b*(Y_int2dt[:,ii*J+0]+Y_int2dt[:,ii*J+1]+Y_int2dt[:,ii*J+2]+Y_int2dt[:,ii*J+3]+
                             Y_int2dt[:,ii*J+4]+Y_int2dt[:,ii*J+5]+Y_int2dt[:,ii*J+6]+Y_int2dt[:,ii*J+7]+
                             Y_int2dt[:,ii*J+8]+Y_int2dt[:,ii*J+9]+Y_int2dt[:,ii*J+10]+Y_int2dt[:,ii*J+11]+
                             Y_int2dt[:,ii*J+12]+Y_int2dt[:,ii*J+13]+Y_int2dt[:,ii*J+14]+Y_int2dt[:,ii*J+15]+
                             Y_int2dt[:,ii*J+16]+Y_int2dt[:,ii*J+17]+Y_int2dt[:,ii*J+18]+Y_int2dt[:,ii*J+19]+
                             Y_int2dt[:,ii*J+20]+Y_int2dt[:,ii*J+21]+Y_int2dt[:,ii*J+22]+Y_int2dt[:,ii*J+23]+
                             Y_int2dt[:,ii*J+24]+Y_int2dt[:,ii*J+25]+Y_int2dt[:,ii*J+26]+Y_int2dt[:,ii*J+27]+
                             Y_int2dt[:,ii*J+28]+Y_int2dt[:,ii*J+29]+Y_int2dt[:,ii*J+30]+Y_int2dt[:,ii*J+31])  
        exact_out_int.append(exact_out)
        
    exact_out_int = np.array(exact_out_int)
    
    ####### Extrap ######
    
    X_ext, Y_ext, t_ext, _ = integrate_L96_2t_with_coupling(X[-1,:], Y[-1,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    X_ext = X_ext[::2,:]
    Y_ext = Y_ext[::2,:]
    t_ext = t_ext[::2]

    Xpred_ext = integrate_L96_2t_with_NN_v(X_ext[0:1,:], si, nt, parameters, model_transfer, F, 0, dt)
    
    X_ext2dt = X_ext[::2,:]
    Y_ext2dt = Y_ext[::2,:]
    t_2dt_ext = t_ext[::2]
    
    Xpred_ext_MAP = Xpred_ext[idx_MAP,:,:]
    
    precision_X = np.repeat(precision[:,None], Xpred_ext.shape[1], axis = 1) # N_HMC x nt
    precision_X = np.repeat(precision_X[:,:,None], Xpred_ext.shape[2], axis = 2) # N_HMC x nt x K
    Sigma_data = np.ones_like(Xpred_ext) / np.sqrt(precision_X) # N_HMC x nt x K
    
    model_HMC.key, _ = random.split(model_HMC.key)
    rr = np.repeat( random.normal(model_HMC.key, (precision.shape[0],1)) , Xpred_ext.shape[1], axis=1) # N_HMC x nt 
    rr = np.repeat(rr[:,:,None], Xpred_ext.shape[2], axis = 2) # N_HMC x nt x K
    
    Xpred_ext = Xpred_ext + (Sigma_data * sigma_X_HMC) * rr
    
    NN_ext = forward_pass_v(X_ext2dt, parameters, model_transfer.num_par_NN, model_transfer.K, model_transfer.L, model_transfer.sigma_X, model_transfer.mu_X) 
    NN_MAP_ext = NN_ext[idx_MAP,:,:]
    
    mu_pred_ext = np.mean(Xpred_ext, axis = 0)
    Sigma_pred_ext = np.var(Xpred_ext, axis = 0)
    
    NN_mu_ext = np.mean(NN_ext, axis = 0)
    NN_Sigma_ext = np.var(NN_ext, axis = 0)
    exact_out_ext = []
    
    for ii in range(K):
        exact_out = - h*c/b*(Y_ext2dt[:,ii*J+0]+Y_ext2dt[:,ii*J+1]+Y_ext2dt[:,ii*J+2]+Y_ext2dt[:,ii*J+3]+
                             Y_ext2dt[:,ii*J+4]+Y_ext2dt[:,ii*J+5]+Y_ext2dt[:,ii*J+6]+Y_ext2dt[:,ii*J+7]+
                             Y_ext2dt[:,ii*J+8]+Y_ext2dt[:,ii*J+9]+Y_ext2dt[:,ii*J+10]+Y_ext2dt[:,ii*J+11]+
                             Y_ext2dt[:,ii*J+12]+Y_ext2dt[:,ii*J+13]+Y_ext2dt[:,ii*J+14]+Y_ext2dt[:,ii*J+15]+
                             Y_ext2dt[:,ii*J+16]+Y_ext2dt[:,ii*J+17]+Y_ext2dt[:,ii*J+18]+Y_ext2dt[:,ii*J+19]+
                             Y_ext2dt[:,ii*J+20]+Y_ext2dt[:,ii*J+21]+Y_ext2dt[:,ii*J+22]+Y_ext2dt[:,ii*J+23]+
                             Y_ext2dt[:,ii*J+24]+Y_ext2dt[:,ii*J+25]+Y_ext2dt[:,ii*J+26]+Y_ext2dt[:,ii*J+27]+
                             Y_ext2dt[:,ii*J+28]+Y_ext2dt[:,ii*J+29]+Y_ext2dt[:,ii*J+30]+Y_ext2dt[:,ii*J+31])
        
        exact_out_ext.append(exact_out)
        
    exact_out_ext = np.array(exact_out_ext)
    
    lower_int = mu_pred - 2.0*np.sqrt(Sigma_pred)
    upper_int = mu_pred + 2.0*np.sqrt(Sigma_pred)
    pts_out_int = (lower_int>X_int) + (upper_int<X_int)
    frac_out_int = np.sum(pts_out_int)/pts_out_int.shape[0]/pts_out_int.shape[1]
    
    lower_NN_int = NN_mu - 2.0*np.sqrt(NN_Sigma)
    upper_NN_int = NN_mu + 2.0*np.sqrt(NN_Sigma)
    pts_out_NN_int = (lower_NN_int.T>exact_out_int) + (upper_NN_int.T<exact_out_int)
    frac_out_NN_int = np.sum(pts_out_NN_int)/pts_out_NN_int.shape[0]/pts_out_NN_int.shape[1]
    
    lower_ext = mu_pred_ext - 2.0*np.sqrt(Sigma_pred_ext)
    upper_ext = mu_pred_ext + 2.0*np.sqrt(Sigma_pred_ext)
    pts_out_ext = (lower_ext>X_ext) + (upper_ext<X_ext)
    frac_out_ext = np.sum(pts_out_ext)/pts_out_ext.shape[0]/pts_out_ext.shape[1]
    
    lower_NN_ext = NN_mu_ext - 2.0*np.sqrt(NN_Sigma_ext)
    upper_NN_ext = NN_mu_ext + 2.0*np.sqrt(NN_Sigma_ext)
    pts_out_NN_ext = (lower_NN_ext.T>exact_out_ext) + (upper_NN_ext.T<exact_out_ext)
    frac_out_NN_ext = np.sum(pts_out_NN_ext)/pts_out_NN_ext.shape[0]/pts_out_NN_ext.shape[1]
    
    err_int = np.linalg.norm(X_int-Xpred_int_MAP) / np.linalg.norm(X_int)
    err_ext = np.linalg.norm(X_ext-Xpred_ext_MAP) / np.linalg.norm(X_ext)
    print('Relative interpolation norm MAP: ',err_int)
    print('Relative extrapolation norm MAP: ',err_ext)
    
    err_int = np.linalg.norm(X_int-mu_pred) / np.linalg.norm(X_int)
    err_ext = np.linalg.norm(X_ext-mu_pred_ext) / np.linalg.norm(X_ext)
    print('Relative interpolation norm mean: ',err_int)
    print('Relative extrapolation norm mean: ',err_ext)
    
    print('Iterpolation out-point franction: ',frac_out_int)
    print('Extrapolation out-point franction: ',frac_out_ext)
    
    err_int = np.linalg.norm(exact_out_int-NN_MAP.T) / np.linalg.norm(exact_out_int)
    err_ext = np.linalg.norm(exact_out_ext-NN_MAP_ext.T) / np.linalg.norm(exact_out_ext)
    print('Relative interpolation norm Closure MAP: ',err_int)
    print('Relative extrapolation norm Closure MAP: ',err_ext)
    
    err_int = np.linalg.norm(exact_out_int-NN_mu.T) / np.linalg.norm(exact_out_int)
    err_ext = np.linalg.norm(exact_out_ext-NN_mu_ext.T) / np.linalg.norm(exact_out_ext)
    print('Relative interpolation norm Closure mean: ',err_int)
    print('Relative extrapolation norm Closure mean: ',err_ext)
    
    print('Iterpolation out-point franction Closure: ',frac_out_NN_int)
    print('Extrapolation out-point franction Closure: ',frac_out_NN_ext)
    
    if compute_results_det == 1:
        print('Relative interpolation norm det: ',err_int_det)
        print('Relative extrapolation norm det: ',err_ext_det)
        print('Relative interpolation norm Closure det: ',err_int_NN_det)
        print('Relative extrapolation norm Closure det: ',err_ext_NN_det)
    
    np.save(fold+'/loggammalist', loggammalist)
    np.save(fold+'/loglikelihood', loglikelihood)
    np.save(fold+'/loglambdalist', loglambdalist)
  
    np.save(fold+'/X_pred_int', Xpred_int)
    np.save(fold+'/X_int', X_int)
    np.save(fold+'/NN_int', NN_int)
    np.save(fold+'/exact_out_int', exact_out_int)
    np.save(fold+'/t', t)
    np.save(fold+'/t_2dt', t_2dt)
    
    np.save(fold+'/X_pred_ext', Xpred_ext)
    np.save(fold+'/X_ext', X_ext)
    np.save(fold+'/NN_ext', NN_ext)
    np.save(fold+'/exact_out_ext', exact_out_ext)
    np.save(fold+'/t_ext', t_ext)
    np.save(fold+'/t_2dt_ext', t_2dt_ext)
    
    np.save(fold+'/X_train', X_train[:nt+1,:])
