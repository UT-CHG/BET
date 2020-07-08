#! /usr/bin/env python

# Copyright (C) 2014-2020 The BET Development Team

"""
This example solves a stochastic inverse problem for a
linear 3-to-2 map with a density-based method.
We refer to the map as the QoI map,
or just a QoI. We refer to the range of the QoI map as
the data space.
The 3-D input space is sampled with initial i.i.d. uniform
random samples.
We refer to the input space as the parameter space,
and use parameter to refer to a particular
point (e.g., a particular random sample) in this space.
The model, a 3-to-2 linear map is solved for these parameters to generate
"predicted" data.

The parameter space is also sampled with a different ("data-generating") random variable, and the linear map
is applied to generate artificial "observed" data.
We solve the density-based approach problem defined by the predicted inputs and outputs and the
observed output data.
In this problem, the initial uniform probability on the parameter space is updated to a new probability measure
based on the data-consistent inversion framework.
This can be compared to the data-generating distribution through plots and a variety of distance metrics.
"""

import numpy as np
import bet.postProcess.plotP as plotP
import bet.calculateP.calculateR as calculateR
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.postProcess.compareP as compP
from myModel import my_model

# Define the sampler that will be used to create the discretization
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem.
# The sampler and my_model is the interface of BET to the model,
# and it allows BET to create input/output samples of the model.
sampler = bsam.sampler(my_model)

# Initialize 3-dimensional input parameter sample set object
input_samples = samp.sample_set(3)

# Set parameter domain
input_samples.set_domain(np.repeat([[0.0, 1.0]], 3, axis=0))

num_samples = int(1E3)
'''
Suggested changes for user:

Try num_samples = 1E3 and 1E4.
What happens when num_samples = 1E2?
'''

# Generate samples on the parameter space
input_samples = sampler.random_sample_set('uniform', input_samples, num_samples=num_samples)

# Create the prediction discretization object using the input samples
disc_predict = sampler.compute_qoi_and_create_discretization(input_samples)

# Generate observed data
sampler_obs = bsam.sampler(my_model)

# Initialize 3-dimensional input parameter sample set object
input_samples_obs = samp.sample_set(3)

# Set parameter domain
input_samples_obs.set_domain(np.repeat([[0.0, 1.0]], 3, axis=0))

# Generate samples on the parameter space
beta_a = 2.0  # a parameter for beta distribution
beta_b = 2.0  # b parameter for beta distribution

'''
Suggested changes for user:

Try changing beta_a and beta_b to shift the data-generating distribution.
Try beta_a = 5, beta_b = 2.
Try beta_a = 0.5, beta_b = 3

Both parameters must be non-negative.
'''

input_samples_obs = sampler_obs.random_sample_set(['beta', {'a': beta_a, 'b': beta_b}],
                                                  input_samples_obs, num_samples=int(1E3))
disc_obs = sampler_obs.compute_qoi_and_create_discretization(input_samples_obs)

# Set probability set for predictions
disc_predict.set_output_observed_set(disc_obs.get_output_sample_set())


# Calculate initial total variation of marginals
comp_init = compP.compare(disc_predict, disc_obs, set1_init=True, set2_init=True)
print("Initial TV of Marginals")
for i in range(3):
    print(comp_init.distance_marginal_quad(i=i, compare_factor=0.2, rtol=1.0e-3, maxiter=1000))

print("------------------------------------------------------")

invert_to = 'kde'  # 'multivariate_gaussian', 'expon', or 'beta'

'''
Suggested changes for user:

Try changing the type of probability measure to invert to from
'kde': Gaussian kernel density estimate (generally the best and most robust choice)
'multivariate_gaussian': fit a multivariate Gaussian distribution
'beta': fit a beta distribution
'expon': fit an exponential distribution (useful if beta_a or beta_b <= 1)

'''

if invert_to == 'kde':
    # Invert to weighted KDE
    print("Weighted Kernel Density Estimate")
    calculateR.invert_to_kde(disc_predict)
elif invert_to == 'multivariate_gaussian':
    # Invert to multivariate Gaussian
    print("Multivariate Gaussian")
    calculateR.invert_to_gmm(disc_predict)
elif invert_to == 'beta':
    # Invert and fit Beta distribution
    print("Beta Distribution")
    calculateR.invert_to_random_variable(disc_predict, rv='beta')
elif invert_to == 'expon':
    # Invert and fit Beta distribution
    print("Beta Distribution")
    calculateR.invert_to_random_variable(disc_predict, rv='expon')


# Calculate Total Variation between updated marginals and data-generating marginals
for i in range(3):
    plotP.plot_1d_marginal_densities(sets=(disc_predict.get_input_sample_set(),
                                           disc_obs.get_input_sample_set()), i=i,
                                     sets_label_initial=['Initial', 'Data-Generating'],
                                     sets_label=['Updated', ''])
# Calculate updated total variation
comp_init = compP.compare(disc_predict, disc_obs, set1_init=False, set2_init=True)
print("Updated TV of Marginals")
for i in range(3):
    print(comp_init.distance_marginal_quad(i=i, compare_factor=0.2, rtol=1.0e-3, maxiter=100))
