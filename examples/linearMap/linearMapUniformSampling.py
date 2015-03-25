#! /usr/bin/env python
"""
This example generates uniform samples on a 3D grid
and evaluates a linear map to a 2d space. Probabilities
in the paramter space are calculated using emulated points.
1D and 2D marginals are calculated, smoothed, and plotted.
"""

import numpy as np
import bet.calculateP as calculateP
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP

# parameter domain
lam_domain= np.array([[0.0, 1.0],
                      [0.0, 1.0],
                      [0.0, 1.0]])
# number of uniform samples in each direction
n0 = 30
n1 = 30
n2 = 30

# QoI map
Q_map = np.array([[0.506, 0.463],[0.253, 0.918], [0.085, 0.496]])

# reference QoI
Q_ref =  np.array([0.422, 0.9385])

# reference parameters
ref_lam = [0.5, 0.5, 0.5]

#set up samples
vec0=list(np.linspace(lam_domain[0][0], lam_domain[0][1], n0))
vec1 = list(np.linspace(lam_domain[1][0], lam_domain[1][1], n1))
vec2 = list(np.linspace(lam_domain[2][0], lam_domain[2][1], n1))
vecv0, vecv1, vecv2 = np.meshgrid(vec0, vec1, vec2, indexing='ij')
samples=np.vstack((vecv0.flat[:], vecv1.flat[:], vecv2.flat[:])).transpose()

# calc data
data= np.dot(samples,Q_map)

# uniform simple function approx
(d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.unif_unif(data=data,
                                                               Q_ref=Q_ref, M=10, bin_ratio=0.2, num_d_emulate=1E4)
# create emulated points
lambda_emulate = calculateP.emulate_iid_lebesgue(lam_domain=lam_domain,
                                                 num_l_emulate = 1E4)
# calculate probablities
(P,  lambda_emulate, io_ptr, emulate_ptr) = calculateP.prob_emulated(samples=samples,
                                                                     data=data, rho_D_M = d_distr_prob, d_distr_samples = d_distr_samples,
                                                                     lam_domain=lam_domain, lambda_emulate=lambda_emulate, d_Tree=d_Tree)
# calculate 2d marginal probs
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(P_samples = P, samples = lambda_emulate, lam_domain = lam_domain, nbins = [10, 10, 10])
# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D,bins, sigma=0.1)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, lam_domain, filename = "linearMap",
                             plot_surface=False)

# calculate 2d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(P_samples = P, samples = lambda_emulate, lam_domain = lam_domain, nbins = [10, 10, 10])
# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.01)
# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, lam_domain, filename = "linearMap")





