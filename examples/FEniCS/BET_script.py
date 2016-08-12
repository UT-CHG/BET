#! /usr/bin/env python

# Copyright (C) 2014-2016 The BET Development Team

r"""
This example requires the following external packages not shipped
with BET:

  (1)  An installation of FEniCS that can be run using the same
  python as used for installing BET. See http://fenicsproject.org/
  for more information.

This example generates samples for a KL expansion associated with
a covariance defined by ``cov`` in computeSaveKL.py on an L-shaped mesh
that defines the permeability field for a Poisson equation solved in
myModel.py.

The quantities of interest (QoI) are defined as two spatial
averages of the solution to the PDE.

The user defines the dimension of the parameter space (corresponding
to the number of KL terms) and the number of samples in this space.
"""

import numpy as np
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam
from myModel import my_model
from Compute_Save_KL import computeSaveKL


# Interface BET to the model.
sampler = bsam.sampler(my_model)

# Define the number of KL terms to use to represent permeability field
num_KL_terms = 2
# Compute and save the KL expansion -- can comment out after running once
computeSaveKL(num_KL_terms)

# Initialize input parameter sample set object
input_samples = samp.sample_set(num_KL_terms)

# Set parameter domain
KL_term_min = -3.0
KL_term_max = 3.0
input_samples.set_domain(np.repeat([[KL_term_min, KL_term_max]],
                                   num_KL_terms,
                                   axis=0))



'''
Suggested changes for user:

Try with and without random sampling.

If using regular sampling, try different numbers of samples
per dimension (be careful if the dimension is not 2).
'''
# Generate samples on the parameter space
randomSampling = False
if randomSampling is True:
    input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E2)
else:
    input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=[10, 10])

'''
A standard Monte Carlo (MC) assumption is that every Voronoi cell
has the same volume. If a regular grid of samples was used, then
the standard MC assumption is true.
'''
MC_assumption = True
# Estimate volumes of Voronoi cells associated with the parameter samples
if MC_assumption is False:
    input_samples.estimate_volume(n_mc_points=1E5)
else:
    input_samples.estimate_volume_mc()

# Create the discretization object using the input samples
my_discretization = sampler.compute_QoI_and_create_discretization(
    input_samples, savefile='FEniCS_Example.txt.gz')

'''
Suggested changes for user:

Try different reference parameters.
'''
# Define the reference parameter
#param_ref = np.zeros((1,num_KL_terms))
param_ref = np.ones((1,num_KL_terms))

# Compute the reference QoI
Q_ref = my_model(param_ref)

# Create some plots of input and output discretizations
plotD.scatter_2D(input_samples, ref_sample=param_ref[0,:],
                 filename='FEniCS_ParameterSamples',
                 file_extension = '.eps')
if Q_ref.size == 2:
    plotD.show_data_domain_2D(my_discretization, Q_ref=Q_ref[0,:],
            file_extension=".eps")

'''
Suggested changes for user:

Try different ways of discretizing the probability measure on D defined
as a uniform probability measure on a rectangle or interval depending
on choice of QoI_num in myModel.py.
'''
randomDataDiscretization = False
if randomDataDiscretization is False:
    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref[0,:], rect_scale=0.1,
        cells_per_dimension=3)
else:
    simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref[0,:], rect_scale=0.1,
        M=50, num_d_emulate=1E5)

# calculate probablities
calculateP.prob(my_discretization)

########################################
# Post-process the results
########################################
# calculate 2d marginal probs
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                        nbins=20)
# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.5)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples,
                             filename="FEniCS",
                             lam_ref=param_ref[0,:],
                             file_extension=".eps",
                             plot_surface=False)

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                        nbins=20)
# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.5)
# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples,
                             filename="FEniCS",
                             lam_ref=param_ref[0,:],
                             file_extension=".eps")



