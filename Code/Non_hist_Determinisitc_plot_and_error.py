#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:37:56 2022

@author: mohamedazizbhouri
"""

import jax.numpy as np

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
fold_save = current_dirs_parent+'/Results/Non_Hist_plots'

fold_res = current_dirs_parent+'/Results/Non_Hist_results'

is_plot_extrap = 1
is_plot_interp = 1

X_train = np.load(fold_res+'/X_train.npy')

Xpred_int = np.load(fold_res+'/X_pred_int_det.npy')
NN_int = np.load(fold_res+'/NN_int_det.npy')
Xpred_ext = np.load(fold_res+'/X_pred_ext_det.npy')
NN_ext = np.load(fold_res+'/NN_ext_det.npy')

X_int = np.load(fold_res+'/X_int_det.npy')
exact_out_int = np.load(fold_res+'/exact_out_int_det.npy')
t = np.load(fold_res+'/t_det.npy')*5
t_2dt = np.load(fold_res+'/t_2dt_det.npy')*5

X_ext = np.load(fold_res+'/X_ext_det.npy')
exact_out_ext = np.load(fold_res+'/exact_out_ext_det.npy')
t_ext = np.load(fold_res+'/t_ext_det.npy')*5
t_2dt_ext = np.load(fold_res+'/t_2dt_ext_det.npy')*5

#### Trajectory plots below ####

X_int2dt = X_int[::2,:]
t_2dt = t[::2]
Xt_dt = X_int2dt[2:,:]
Xth1_dt = X_int2dt[1:-1,:]
Xth2_dt = X_int2dt[:-2,:]   

if is_plot_interp == 1:
    
    plt.figure(figsize=(22,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            
            exact_out = exact_out_int[ii,:]
            
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_2dt, exact_out, 'r-', label = "True")
            plt.plot(t_2dt, NN_int[:,ii], 'b--', label = "Prediction")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'Closure for $X_'+str(ii+1)+'$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=2, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.4,-3.8))
              
    plt.savefig(fold_save+'/interp_non_hist_det_closure.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
    plt.close()
    
    plt.figure(figsize=(18,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t, X_int[:,ii], 'r-', label = "True")
            plt.plot(t, Xpred_int[:,ii], 'b--', label = "Prediction")
            plt.plot(t, X_train[:,ii], 'k--', label = "Training data") # 'ro'
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'$X_'+str(ii+1)+'(t)$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.8,-3.8))
                
    plt.savefig(fold_save+'/interp_non_hist_det_X.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1) 
    plt.close()
                
####### Extrap ######

if is_plot_extrap == 1:
    
    plt.figure(figsize=(22,24))
    
    for kk in range(4):
        for jj in range(2):
                
            ii = kk+jj*4
                
            exact_out = exact_out_ext[ii,:]
                
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_2dt_ext, exact_out, 'r-', label = "True")
            plt.plot(t_2dt_ext, NN_ext[:,ii], 'b--', label = "Prediction")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'Closure for $X_'+str(ii+1)+'$',fontsize=26)
            if kk==0 and jj==0:
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.4,-3.8))
                    
    plt.savefig(fold_save+'/non_hist_det_closure.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)   
    plt.close()
    
    plt.figure(figsize=(18,24))
    
    for kk in range(4):
        for jj in range(2):
            
            ii = kk+jj*4
            
            ax1 = plt.subplot(4, 2, ii+1)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.plot(t_ext, X_ext[:,ii], 'r-', label = "True")
            plt.plot(t_ext, Xpred_ext[:,ii], 'b--', label = "Prediction")
            plt.xlabel('$t$ (ATM days)',fontsize=26)
            plt.ylabel(r'$X_'+str(ii+1)+'(t)$',fontsize=26)
            if kk==0 and jj==0:
#                    plt.legend(frameon=False, prop={'size': 20}, bbox_to_anchor=(x,y))
                plt.legend(ncol=3, frameon=False, prop={'size': 20}, bbox_to_anchor=(1.5,-3.8))
                
    plt.savefig(fold_save+'/non_hist_det_X.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)   
    plt.close()
                
err_int = np.linalg.norm(X_int-Xpred_int) / np.linalg.norm(X_int)
err_ext = np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
print('Relative interpolation error for X: ',err_int)
print('Relative extrapolation error for X: ',err_ext)

err_int = np.linalg.norm(exact_out_int-NN_int.T) / np.linalg.norm(exact_out_int)
err_ext = np.linalg.norm(exact_out_ext-NN_ext.T) / np.linalg.norm(exact_out_ext)
print('Relative interpolation error for Closure: ',err_int)
print('Relative extrapolation error for Closure: ',err_ext)
