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
fold_save = current_dirs_parent+'/Results/RMSE_plots'

fold_res_h = current_dirs_parent+'/Results/Hist_results'
fold_res_nh = current_dirs_parent+'/Results/Non_Hist_results'

# plotting settings
nt = 1000

X_train = np.load(fold_res_h+'/X_train.npy')
X2 = np.load(fold_res_h+'/X2.npy')

Xpred_int_det_h = np.load(fold_res_h+'/X_pred_int_det.npy')
Xpred_ext_det_h = np.load(fold_res_h+'/X_pred_ext_det.npy')
Xpred_int_det_nh = np.load(fold_res_nh+'/X_pred_int_det.npy')
Xpred_ext_det_nh = np.load(fold_res_nh+'/X_pred_ext_det.npy')

X_int = np.load(fold_res_h+'/X_int.npy')
t = np.load(fold_res_h+'/t.npy')*5

X_ext = np.load(fold_res_h+'/X_ext.npy')
t_ext = np.load(fold_res_h+'/t_ext.npy')*5

loggammalist_h = np.load(fold_res_h+'/loggammalist.npy')
loglikelihood_h = np.load(fold_res_h+'/loglikelihood.npy')
loglambdalist_h = np.load(fold_res_h+'/loglambdalist.npy')
Xpred_int_h = np.load(fold_res_h+'/X_pred_int.npy')
Xpred_ext_h = np.load(fold_res_h+'/X_pred_ext.npy')
loglikelihood_h = loglikelihood_h
idx_MAP_h = np.argmin(loglikelihood_h)
Xpred_int_MAP_h = Xpred_int_h[idx_MAP_h,:,:]
Xpred_ext_MAP_h = Xpred_ext_h[idx_MAP_h,:,:]
mu_pred_int_h = np.mean(Xpred_int_h, axis = 0)
mu_pred_ext_h = np.mean(Xpred_ext_h, axis = 0)

loggammalist_nh = np.load(fold_res_nh+'/loggammalist.npy')
loglikelihood_nh = np.load(fold_res_nh+'/loglikelihood.npy')
loglambdalist_nh = np.load(fold_res_nh+'/loglambdalist.npy')
Xpred_int_nh = np.load(fold_res_nh+'/X_pred_int.npy')
Xpred_ext_nh = np.load(fold_res_nh+'/X_pred_ext.npy')
loglikelihood_nh = loglikelihood_nh
idx_MAP_nh = np.argmin(loglikelihood_nh)
Xpred_int_MAP_nh = Xpred_int_nh[idx_MAP_nh,:,:]
Xpred_ext_MAP_nh = Xpred_ext_nh[idx_MAP_nh,:,:]
mu_pred_int_nh = np.mean(Xpred_int_nh, axis = 0)
mu_pred_ext_nh = np.mean(Xpred_ext_nh, axis = 0)

#### Trajectory plots below ####

err_det_int_h = []
err_det_int_nh = []
err_det_ext_h = []
err_det_ext_nh = []

err_mean_int_h = []
err_mean_int_nh = []
err_MAP_int_h = []
err_MAP_int_nh = []
err_mean_ext_h = []
err_mean_ext_nh = []
err_MAP_ext_h = []
err_MAP_ext_nh = []

err_X2 = []

for i in range(nt+1):
    if i%50 == 0:
        print(i)
    err_det_int_h.append( np.linalg.norm(X_int[:i,:]-Xpred_int_det_h[:i,:]) / np.linalg.norm(X_int[:i,:]) )
    err_det_int_nh.append( np.linalg.norm(X_int[:i,:]-Xpred_int_det_nh[:i,:]) / np.linalg.norm(X_int[:i,:]) )
    err_det_ext_h.append( np.linalg.norm(X_ext[:i,:]-Xpred_ext_det_h[:i,:]) / np.linalg.norm(X_ext[:i,:]) )
    err_det_ext_nh.append( np.linalg.norm(X_ext[:i,:]-Xpred_ext_det_nh[:i,:]) / np.linalg.norm(X_ext[:i,:]) )
    
    err_mean_int_h.append( np.linalg.norm(X_int[:i,:]-mu_pred_int_h[:i,:]) / np.linalg.norm(X_int[:i,:]) )
    err_mean_int_nh.append( np.linalg.norm(X_int[:i,:]-mu_pred_int_nh[:i,:]) / np.linalg.norm(X_int[:i,:]) )
    err_mean_ext_h.append( np.linalg.norm(X_ext[:i,:]-mu_pred_ext_h[:i,:]) / np.linalg.norm(X_ext[:i,:]) )
    err_mean_ext_nh.append( np.linalg.norm(X_ext[:i,:]-mu_pred_ext_nh[:i,:]) / np.linalg.norm(X_ext[:i,:]) )

    err_MAP_int_h.append( np.linalg.norm(X_int[:i,:]-Xpred_int_MAP_h[:i,:]) / np.linalg.norm(X_int[:i,:]) )
    err_MAP_int_nh.append( np.linalg.norm(X_int[:i,:]-Xpred_int_MAP_nh[:i,:]) / np.linalg.norm(X_int[:i,:]) )
    err_MAP_ext_h.append( np.linalg.norm(X_ext[:i,:]-Xpred_ext_MAP_h[:i,:]) / np.linalg.norm(X_ext[:i,:]) )
    err_MAP_ext_nh.append( np.linalg.norm(X_ext[:i,:]-Xpred_ext_MAP_nh[:i,:]) / np.linalg.norm(X_ext[:i,:]) )
for i in range(nt):    
    err_X2.append( np.linalg.norm(X_ext[i:i+1,:]-X2[:i,:]) / np.linalg.norm(X_ext[i:i+1,:]) )
    
err_det_int_h = np.array(err_det_int_h)    
err_det_int_nh = np.array(err_det_int_nh)    
err_det_ext_h = np.array(err_det_ext_h)    
err_det_ext_nh = np.array(err_det_ext_nh)    

err_mean_int_h = np.array(err_mean_int_h)    
err_mean_int_nh = np.array(err_mean_int_nh)    
err_mean_ext_h = np.array(err_mean_ext_h)    
err_mean_ext_nh = np.array(err_mean_ext_nh)    

err_MAP_int_h = np.array(err_MAP_int_h)    
err_MAP_int_nh = np.array(err_MAP_int_nh)    
err_MAP_ext_h = np.array(err_MAP_ext_h)    
err_MAP_ext_nh = np.array(err_MAP_ext_nh)    

err_X2 = np.array(err_X2)

###############################################
############## Truth 2\Delta t ################
###############################################

plt.figure(figsize=(12,6.5))
plt.yscale("log") 
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t[:-1], err_det_ext_h[1:], 'k--', label = "Deterministic Param.")
plt.plot(t[:-1], err_mean_ext_h[1:], '--', color='hotpink', label = "Bayesian Param. (Mean)")
plt.plot(t[:-1], err_MAP_ext_h[1:], 'r--', label = "Bayesian Param. (MAP)")
plt.plot(t[:23], err_X2[:23], 'b-', label = "\"Truth\" Model w. $2\Delta t$")
plt.axvline(x = t[22], color = 'orange', label = 'Last stable point')
plt.xlabel('$t$ (ATM days)',fontsize=26)
plt.ylabel('RMSE $(t)$',fontsize=26)
plt.legend(frameon=False, prop={'size': 20})
plt.savefig(fold_save+'/RMSE_truth.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
plt.close()

###############################################
##################### det #####################
###############################################
plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t, err_det_int_h, 'r--', label = "History-based")
plt.plot(t, err_det_int_nh, 'b-', label = "Non-history-based")
plt.xlabel('$t$ (ATM days)',fontsize=26)
plt.ylabel('RMSE $(t)$',fontsize=26)
plt.legend(frameon=False, prop={'size': 20})
plt.savefig(fold_save+'/Interp_RMSE_det.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
plt.close()
      
plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t, err_det_ext_h, 'r--', label = "History-based")
plt.plot(t, err_det_ext_nh, 'b-', label = "Non-history-based")
plt.xlabel('$t$ (ATM days)',fontsize=26)
plt.ylabel('RMSE $(t)$',fontsize=26)
plt.legend(frameon=False, prop={'size': 20})
plt.savefig(fold_save+'/RMSE_det.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
plt.close()
                    
err_int = np.linalg.norm(X_int-Xpred_int_det_h) / np.linalg.norm(X_int)
err_ext = np.linalg.norm(X_ext-Xpred_ext_det_h) / np.linalg.norm(X_ext)
print('Relative deterministic interpolation error for X - History: ',err_int)
print('Relative deterministic extrapolation error for X - History: ',err_ext)
 
err_int = np.linalg.norm(X_int-Xpred_int_det_nh) / np.linalg.norm(X_int)
err_ext = np.linalg.norm(X_ext-Xpred_ext_det_nh) / np.linalg.norm(X_ext)
print('Relative deterministic interpolation error for X - Non-history: ',err_int)
print('Relative deterministic extrapolation error for X - Non-history: ',err_ext)

###############################################
##################### mean ####################
###############################################

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t, err_mean_int_h, 'r--', label = "History-based")
plt.plot(t, err_mean_int_nh, 'b-', label = "Non-history-based")
plt.xlabel('$t$ (ATM days)',fontsize=26)
plt.ylabel('RMSE $(t)$',fontsize=26)
plt.legend(frameon=False, prop={'size': 20})
plt.savefig(fold_save+'/Interp_RMSE_mean.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
plt.close()
      
plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t, err_mean_ext_h, 'r--', label = "History-based")
plt.plot(t, err_mean_ext_nh, 'b-', label = "Non-history-based")
plt.xlabel('$t$ (ATM days)',fontsize=26)
plt.ylabel('RMSE $(t)$',fontsize=26)
plt.legend(frameon=False, prop={'size': 20})
plt.savefig(fold_save+'/RMSE_mean.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
plt.close()
                    
err_int = np.linalg.norm(X_int-mu_pred_int_h) / np.linalg.norm(X_int)
err_ext = np.linalg.norm(X_ext-mu_pred_ext_h) / np.linalg.norm(X_ext)
print('Relative mean interpolation error for X - History: ',err_int)
print('Relative mean extrapolation error for X - History: ',err_ext)
 
err_int = np.linalg.norm(X_int-mu_pred_int_nh) / np.linalg.norm(X_int)
err_ext = np.linalg.norm(X_ext-mu_pred_ext_nh) / np.linalg.norm(X_ext)
print('Relative mean interpolation error for X - Non-history: ',err_int)
print('Relative mean extrapolation error for X - Non-history: ',err_ext)

###############################################
##################### MAP #####################
###############################################

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t, err_MAP_int_h, 'r--', label = "History-based")
plt.plot(t, err_MAP_int_nh, 'b-', label = "Non-history-based")
plt.xlabel('$t$ (ATM days)',fontsize=26)
plt.ylabel('RMSE $(t)$',fontsize=26)
plt.legend(frameon=False, prop={'size': 20})
plt.savefig(fold_save+'/Interp_RMSE_MAP.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
plt.close()
      
plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t, err_MAP_ext_h, 'r--', label = "History-based")
plt.plot(t, err_MAP_ext_nh, 'b-', label = "Non-history-based")
plt.xlabel('$t$ (ATM days)',fontsize=26)
plt.ylabel('RMSE $(t)$',fontsize=26)
plt.legend(frameon=False, prop={'size': 20})
plt.savefig(fold_save+'/RMSE_MAP.png', dpi = 300, bbox_inches='tight',pad_inches = 0.1)
plt.close()
                    
err_int = np.linalg.norm(X_int-Xpred_int_MAP_h) / np.linalg.norm(X_int)
err_ext = np.linalg.norm(X_ext-Xpred_ext_MAP_h) / np.linalg.norm(X_ext)
print('Relative MAP interpolation error for X - History: ',err_int)
print('Relative MAP extrapolation error for X - History: ',err_ext)
 
err_int = np.linalg.norm(X_int-Xpred_int_MAP_nh) / np.linalg.norm(X_int)
err_ext = np.linalg.norm(X_ext-Xpred_ext_MAP_nh) / np.linalg.norm(X_ext)
print('Relative MAP interpolation error for X - Non-history: ',err_int)
print('Relative MAP extrapolation error for X - Noh-history: ',err_ext)
