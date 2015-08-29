#! /usr/bin/env python

# Copyright (C) 2014-2015 The BET Development Team

"""
This example takes uniformly distributed samples of parameters and
output data from a simple groundwater contaminant transport model,
and calculates solutions to the stochastic inverse problem.
The parameter domain is 5D, where the uncertain parameters are the x and y 
locations of the source, the horizontal dispersivity, the flow angle,
and the contaminant flux. There are 11 measured QoIs in the data space 
available. By changing the choice of QoIs that we use to solve the stochastic
inverse problem, we see the effect of geometric distinctness. 
Probabilities in the parameter space are 
calculated using the MC assumption.  1D and 2D marginals are calculated,
smoothed, and plotted. The samples are then ranked by probability density
and the volume of high-probability regions are calculated. Probabilistic predictions of other QoI are made.

"""

import numpy as np
import bet.calculateP as calculateP
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.postProcess.postTools as postTools


# Labels and descriptions of the uncertain parameters
labels = ['Source $y$ coordinate [L]', 'Source $x$ coordinate [L]', 'Dispersivity x [L]', 'Flow Angle [degrees]', 'Contaminant flux [M/T]']

# Load data from files
lam_domain = np.loadtxt("files/lam_domain.txt.gz") #parameter domain
ref_lam = np.loadtxt("files/lam_ref.txt.gz") #reference parameter set
Q_ref = np.loadtxt("files/Q_ref.txt.gz") #reference QoI set
samples = np.loadtxt("files/samples.txt.gz") # uniform samples in parameter domain
dataf = np.loadtxt("files/data.txt.gz") # data from model

QoI_indices=[0,1,2,3] # Indices for output data with which you want to invert
bin_ratio = 0.25 #ratio of length of data region to invert

data = dataf[:,QoI_indices]
Q_ref=Q_ref[QoI_indices]

dmax = data.max(axis=0)
dmin = data.min(axis=0)
dscale = bin_ratio*(dmax-dmin)
Qmax = Q_ref + 0.5*dscale
Qmin = Q_ref -0.5*dscale
def rho_D(x):
  return np.all(np.logical_and(np.greater(x,Qmin), np.less(x,Qmax)),axis=1)

# Plot the data domain
plotD.show_data(data, Q_ref = Q_ref, rho_D=rho_D, showdim=2)

# Whether or not to use deterministic description of simple function approximation of
# ouput probability
deterministic_discretize_D = True
if deterministic_discretize_D == True:
  (d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle(data=data,
                                                                              Q_ref=Q_ref, 
                                                                              bin_ratio=bin_ratio, 
                                                                              center_pts_per_edge = 1)
else:
  (d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.unif_unif(data=data,
                                                                 Q_ref=Q_ref, 
                                                                 M=50, 
                                                                 bin_ratio=bin_ratio, 
                                                                 num_d_emulate=1E5)
  
# calculate probablities making Monte Carlo assumption
(P,  lam_vol, io_ptr) = calculateP.prob(samples=samples,
                                        data=data,
                                        rho_D_M=d_distr_prob,
                                        d_distr_samples=d_distr_samples)

# calculate 2D marginal probabilities
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(P_samples = P, samples = samples, lam_domain = lam_domain, nbins = 10)

# smooth 2D marginal probabilites for plotting (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D,bins, sigma=1.0)

# plot 2D marginal probabilities
plotP.plot_2D_marginal_probs(marginals2D, bins, lam_domain, filename = "contaminant_map",
                             plot_surface=False,
                             lam_ref = ref_lam,
                             lambda_label=labels,
                             interactive=False)

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(P_samples = P, samples = samples, lam_domain = lam_domain, nbins = 20)

# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=1.0)

# plot 1d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, lam_domain, filename = "contaminant_map", interactive=False, lam_ref=ref_lam, lambda_label=labels)

percentile = 1.0
# Sort samples by highest probability density and sample highest percentile percent samples
(num_samples, P_high, samples_high, lam_vol_high, data_high)= postTools.sample_highest_prob(top_percentile=percentile, P_samples=P, samples=samples, lam_vol=lam_vol,data = data,sort=True)

# print the number of samples that make up the  highest percentile percent samples and
# ratio of the volume of the parameter domain they take up
print (num_samples, np.sum(lam_vol_high))

# Propogate the probability measure through a different QoI map
(_, P_pred, _, _ , data_pred)= postTools.sample_highest_prob(top_percentile=percentile, P_samples=P, samples=samples, lam_vol=lam_vol,data = dataf[:,7],sort=True)

# Plot 1D pdf of predicted QoI
# calculate 1d marginal probs
(bins_pred, marginals1D_pred) = plotP.calculate_1D_marginal_probs(P_samples = P_pred, samples = data_pred, lam_domain = np.array([[np.min(data_pred),np.max(data_pred)]]), nbins = 20)

# plot 1d pdf 
plotP.plot_1D_marginal_probs(marginals1D_pred, bins_pred, lam_domain= np.array([[np.min(data_pred),np.max(data_pred)]]), filename = "contaminant_prediction", interactive=False)
