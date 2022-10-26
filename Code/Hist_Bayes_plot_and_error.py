#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:37:56 2022

@author: mohamedazizbhouri
"""

import jax.numpy as np
from jax import random

from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16,
                     'lines.linewidth': 2,
                     'axes.labelsize': 20,  # fontsize for x and y labels (was 10)
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 20,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex"#,        # change this if using xetex or lautex
#                     "text.usetex": True,                # use LaTeX to write all text
                     })

import os
current_dirs_parent = os.path.dirname(os.getcwd())
fold_save = current_dirs_parent+'/Results/Hist_plots'

fold_res = current_dirs_parent+'/Results/Hist_results'

is_plot_extrap = 1
is_plot_interp = 1

# plotting settings
N_end = 4000 # the first N_end of HMC are considered for all subsequent results and plots
N_total = 4000 # last N_total of HMC are considered to get the N_samples samples
N_samples = 4000 
N_MAP = 4000 # last N_MAP of HMC are considered to look for MAP

X_train = np.load(fold_res+'/X_train.npy')

loggammalist = np.load(fold_res+'/loggammalist.npy')[:N_end]
loglikelihood = np.load(fold_res+'/loglikelihood.npy')[:N_end]
loglambdalist = np.load(fold_res+'/loglambdalist.npy')[:N_end]
  
Xpred_int = np.load(fold_res+'/X_pred_int.npy')[:N_end,:,:]
NN_int = np.load(fold_res+'/NN_int.npy')[:N_end,:,:]
Xpred_ext = np.load(fold_res+'/X_pred_ext.npy')[:N_end,:,:]
NN_ext = np.load(fold_res+'/NN_ext.npy')[:N_end,:,:]

loggammalist = loggammalist[- N_total:]
loglikelihood = loglikelihood[- N_total:]
loglambdalist = loglambdalist[- N_total:]

Xpred_int = Xpred_int[- N_total:,:,:]
NN_int = NN_int[- N_total:,:,:]
Xpred_ext = Xpred_ext[- N_total:,:,:]
NN_ext = NN_ext[- N_total:,:,:]

X_int = np.load(fold_res+'/X_int.npy')
exact_out_int = np.load(fold_res+'/exact_out_int.npy')
t = np.load(fold_res+'/t.npy')*5
t_2dt = np.load(fold_res+'/t_2dt.npy')*5

X_ext = np.load(fold_res+'/X_ext.npy')
exact_out_ext = np.load(fold_res+'/exact_out_ext.npy')
t_ext = np.load(fold_res+'/t_ext.npy')*5
t_2dt_ext = np.load(fold_res+'/t_2dt_ext.npy')*5

key = random.PRNGKey(1234)
idx = random.choice(key, N_total, (N_samples,), replace=False)

#### Trajectory plots below ####

loglikelihood = loglikelihood[-N_MAP:]
idx_MAP = np.argmin(loglikelihood)

X_int2dt = X_int[::2,:]
t_2dt = t[::2]
Xt_dt = X_int2dt[2:,:]
Xth1_dt = X_int2dt[1:-1,:]
Xth2_dt = X_int2dt[:-2,:]

Xpred_int_MAP = Xpred_int[idx_MAP,:,:]

NN_MAP = NN_int[idx_MAP,:,:]

Xpred_int = Xpred_int[idx,:,:]
NN_int = NN_int[idx,:,:]

mu_pred = np.mean(Xpred_int, axis = 0)
Sigma_pred = np.var(Xpred_int, axis = 0)

NN_mu = np.mean(NN_int, axis = 0)
NN_Sigma = np.var(NN_int, axis = 0)

if is_plot_interp == 1:
    
    plt.figure(figsize=(22,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            
            exact_out = exact_out_int[ii,:]
            
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_2dt[2:], NN_int[0,:,ii], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory from posterior", linewidth = 0.5)
            plt.plot(t_2dt[2:], NN_int[:,:,ii].T, '-', color = "gray", alpha = 0.8, linewidth = 0.5)
            plt.plot(t_2dt[2:], exact_out, 'r--', label = "True")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'Closure for $X_'+str(ii+1)+'$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=2, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.6,-3.8))
              
    plt.savefig(fold_save+'/interp_hist_Bayes_samples_closure.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
    plt.close()
    
    plt.figure(figsize=(22,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            exact_out = exact_out_int[ii,:]
            
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_2dt[2:], exact_out, 'r--', label = "True")
            plt.plot(t_2dt[2:], NN_mu[:,ii], 'b-', label = "Mean prediction")
            lower_0 = NN_mu[:,ii] - 2.0*np.sqrt(NN_Sigma[:,ii])
            upper_0 = NN_mu[:,ii] + 2.0*np.sqrt(NN_Sigma[:,ii])
            plt.fill_between(t_2dt[2:].flatten(), lower_0.flatten(), upper_0.flatten(), 
                             facecolor='orange', alpha=0.5, label="Two std band")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'Closure for $X_'+str(ii+1)+'$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.7,-3.8))
                
    plt.savefig(fold_save+'/interp_hist_Bayes_mean_closure.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
    plt.close()
    
    plt.figure(figsize=(18,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t, Xpred_int[0,:,ii], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory from posterior", linewidth = 0.5)
            plt.plot(t, Xpred_int[:,:,ii].T, '-', color = "gray", alpha = 0.8, linewidth = 0.5)
            plt.plot(t, X_int[:,ii], 'r-', label = "True")
            plt.plot(t, X_train[:,ii], 'k--', label = "Training data")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'$X_'+str(ii+1)+'(t)$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.9,-3.8))
                
    plt.savefig(fold_save+'/interp_hist_Bayes_samples_X.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1) 
    plt.close()
    
    plt.figure(figsize=(18,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            
            ax1 = plt.subplot(4, 2, ii+1)
            
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t, X_int[:,ii], 'r-', label = "True")
            plt.plot(t, X_train[:,ii], 'k--', label = "Training data")
            plt.plot(t, mu_pred[:,ii], 'b--', label = "Mean prediction")
            lower_0 = mu_pred[:,ii] - 2.0*np.sqrt(Sigma_pred[:,ii])
            upper_0 = mu_pred[:,ii] + 2.0*np.sqrt(Sigma_pred[:,ii])
            plt.fill_between(t.flatten(), lower_0.flatten(), upper_0.flatten(), 
                             facecolor='orange', alpha=0.5, label="Two std band")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'$X_'+str(ii+1)+'(t)$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=4, frameon=False, prop={'size': 20}, bbox_to_anchor=(2.05,-3.8))
                
    plt.savefig(fold_save+'/interp_hist_Bayes_mean_X.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
    plt.close()
        
####### Extrap ######

Xpred_ext_MAP = Xpred_ext[idx_MAP,:,:]

NN_MAP_ext = NN_ext[idx_MAP,:,:]

Xpred_ext = Xpred_ext[idx,:,:]
NN_ext = NN_ext[idx,:,:]

mu_pred_ext = np.mean(Xpred_ext, axis = 0)
Sigma_pred_ext = np.var(Xpred_ext, axis = 0)

NN_mu_ext = np.mean(NN_ext, axis = 0)
NN_Sigma_ext = np.var(NN_ext, axis = 0)

if is_plot_extrap == 1:
    
    plt.figure(figsize=(22,24))
    
    for kk in range(4):
        for jj in range(2):
                
            ii = kk+jj*4
                
            exact_out = exact_out_ext[ii,:]
                
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_2dt_ext[2:], NN_ext[0,:,ii], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory from posterior", linewidth = 0.5)
            plt.plot(t_2dt_ext[2:], NN_ext[:,:,ii].T, '-', color = "gray", alpha = 0.8, linewidth = 0.5)
            plt.plot(t_2dt_ext[2:], exact_out, 'r--', label = "True")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'Closure for $X_'+str(ii+1)+'$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.6,-3.8))
                    
    plt.savefig(fold_save+'/hist_Bayes_samples_closure.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)   
    plt.close()
    
    plt.figure(figsize=(22,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            exact_out = exact_out_ext[ii,:]
            
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_2dt_ext[2:], exact_out, 'r--', label = "True")
            plt.plot(t_2dt_ext[2:], NN_mu_ext[:,ii], 'b-', label = "Mean prediction")
            lower_0 = NN_mu_ext[:,ii] - 2.0*np.sqrt(NN_Sigma_ext[:,ii])
            upper_0 = NN_mu_ext[:,ii] + 2.0*np.sqrt(NN_Sigma_ext[:,ii])
            plt.fill_between(t_2dt_ext[2:].flatten(), lower_0.flatten(), upper_0.flatten(), 
                             facecolor='orange', alpha=0.5, label="Two std band")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'Closure for $X_'+str(ii+1)+'$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.7,-3.8))
                
    plt.savefig(fold_save+'/hist_Bayes_mean_closure.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
    plt.close()
    
    plt.figure(figsize=(18,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_ext, Xpred_ext[0,:,ii], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory from posterior", linewidth = 0.5)
            plt.plot(t_ext, Xpred_ext[:,:,ii].T, '-', color = "gray", alpha = 0.8, linewidth = 0.5)
            plt.plot(t_ext, X_ext[:,ii], 'r-', label = "True")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'$X_'+str(ii+1)+'(t)$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.7,-3.8))
                
    plt.savefig(fold_save+'/hist_Bayes_samples_X.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)   
    plt.close()
        
    plt.figure(figsize=(18,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            
            ax1 = plt.subplot(4, 2, ii+1)
            
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_ext, X_ext[:,ii], 'r-', label = "True")
            plt.plot(t_ext, mu_pred_ext[:,ii], 'b--', label = "Mean prediction")
            lower_0 = mu_pred_ext[:,ii] - 2.0*np.sqrt(Sigma_pred_ext[:,ii])
            upper_0 = mu_pred_ext[:,ii] + 2.0*np.sqrt(Sigma_pred_ext[:,ii])
            plt.fill_between(t_ext.flatten(), lower_0.flatten(), upper_0.flatten(), 
                             facecolor='orange', alpha=0.5, label="Two std band")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'$X_'+str(ii+1)+'(t)$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.8,-3.8))
                
    plt.savefig(fold_save+'/hist_Bayes_mean_X.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
    plt.close()
    
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
print('Relative MAP interpolation error for X : ',err_int)
print('Relative MAP extrapolation error for X : ',err_ext)

err_int = np.linalg.norm(X_int-mu_pred) / np.linalg.norm(X_int)
err_ext = np.linalg.norm(X_ext-mu_pred_ext) / np.linalg.norm(X_ext)
print('Relative mean interpolation error for X: ',err_int)
print('Relative mean extrapolation error for X: ',err_ext)

print('Iterpolation out-point franction for X: ',frac_out_int)
print('Extrapolation out-point franction for X: ',frac_out_ext)

err_int = np.linalg.norm(exact_out_int-NN_MAP.T) / np.linalg.norm(exact_out_int)
err_ext = np.linalg.norm(exact_out_ext-NN_MAP_ext.T) / np.linalg.norm(exact_out_ext)
print('Relative MAP interpolation error for Closure: ',err_int)
print('Relative MAP extrapolation error for Closure: ',err_ext)

err_int = np.linalg.norm(exact_out_int-NN_mu.T) / np.linalg.norm(exact_out_int)
err_ext = np.linalg.norm(exact_out_ext-NN_mu_ext.T) / np.linalg.norm(exact_out_ext)
print('Relative mean interpolation error for Closure: ',err_int)
print('Relative mean extrapolation error for Closure: ',err_ext)

print('Iterpolation out-point franction for Closure: ',frac_out_NN_int)
print('Extrapolation out-point franction for Closure: ',frac_out_NN_ext)
