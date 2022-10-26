Hist_Bayes_Closure Code Guide
@author: mohamedazizbhouri

###################################################
################ Code and Results #################
###################################################

The folder "code" contains the implementation of the Bayesian and deterministic history- and non-history-based parameterization for Loren '96 model as detailed in the paper "History-Based, Bayesian, Closure for Stochastic Parameterization: Application to Lorenz '96". Below is a list of the code scripts.

1(a)- The code "Hist_Bayes.py" contains the implementation of the Bayesian history-based parameterization for Loren '96 model and saves the simulation results in folder "Results/Hist_results". 

1(b)- The code "Hist_Deterministic.py" contains the implementation of the deterministic history-based parameterization for Loren '96 model and saves the simulation results in folder "Results/Hist_results". 

1(c)- Both "Hist_Bayes.py" and "Hist_Deterministic.py" call the code "integrate_L96_2t_with_coupling.py" to generate the "true" data, and the codes "forward_pass.py" and "stepper.py" to compute the parameterization and the time-stepping of the parameterized model respectively. They also call "integrate_L96_2t_with_NN.npy" to simulate a trajectory of the parameterized model.

1(d)- The codes "Hist_Determinisitc_plot_and_error.py" and "Hist_Bayes_plot_and_error.py" outputs the errors and generates the plots using the simulation results saved in folder "Results/Hist_results" for the deterministic and Bayesian history-based models respectively. The plots are saved in the folder "Results/Hist_plots".

2(a)- The code "Non_hist_Bayes.py" contains the implementation of the Bayesian non-history-based parameterization for Loren '96 model and saves the simulation results in folder "Results/Non_Hist_results". 

2(b)- The code "Non_hist_Deterministic.py" contains the implementation of the deterministic non-history-based parameterization for Loren '96 model and saves the simulation results in folder "Results/Non_Hist_results".

2(c)- Both "Non_hist_Bayes.py" and "Non_hist_Deterministic.py" call the code "integrate_L96_2t_with_coupling.py" to generate the "true" data. They call the code "Non_hist_time_integration.npy" to access the functions "forward_pass" and "stepper" needed to to compute the parameterization and the time-stepping of the parameterized model respectively, and also the code "integrate_L96_2t_with_NN" needed to simulate a trajectory of the parameterized model.

2(d)- The codes "Non_hist_Determinisitc_plot_and_error.py" and "Non_hist_Bayes_plot_and_error.py" outputs the errors and generates the plots using the simulation results saved in folder "Results/Non_Hist_results" for the deterministic and Bayesian non-history-based models respectively. The plots are saved in the folder "Results/Non_Hist_plots".

3- The code "true_model_diff_dt.py" solves the "true" Lorenz '96 model with different time steps for stability analysis. The results are saved in folder "Results/Hist_plots".

4- The code "RMSE_plot.py" computes the temporal evolution of RMSE based on "true", history- and non-history based models results saved in "Results/Hist_plots" and "Results/Non_Hist_plots" and saves the plots in folder "Results/RMSE_plots"

###################################################
#################### Remarks ######################
###################################################

Rk1: Due to the size of some data files exceeding the Github limit, the easiest way to download the codes and all data is to use the following Google Drive link to the whole repository: https://drive.google.com/drive/folders/1vxlzLLah7HB0zEx46bbDmq7dM2_Y-FcD?usp=sharing

Rk2: The Github repository contains all data except the HMC-based trajectories for the resolved variables and closure due to the size limit. The corresponding files can be found in the Google Drive repository mentioned above. In directory "Results/Hist_results", the files "X_pred_ext.npy" and "NN_ext.npy" correspond to the history-based HMC trajectories starting from the last training point for the resolved variables and closure respectively, while the files "X_pred_int.npy" and "NN_int.npy" correspond to the same trajectories but starting from the first training point. In directory "Results/Non_Hist_results", the files "X_pred_ext.npy" and "NN_ext.npy" correspond to the non-history-based HMC trajectories starting from the last training point for the resolved variables and closure respectively, while the files "X_pred_int.npy" and "NN_int.npy" correspond to the same trajectories but starting from the first training point.

Rk3: The code was run using the jax version 0.3.13 and the jaxlib version 0.3.10

