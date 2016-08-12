#! /usr/bin/env python

# Copyright (C) 2014-2016 The BET Development Team

r"""
This example generates samples on a 2D grid and evaluates
a nonlinear map to a 1d or 2d space. The maps are defined
as quantities of interest (QoI) defined as spatial
observations of the solution to the elliptic PDE .. math::
  :nowrap:

  \begin{cases}
    -\nabla \cdot (A(\lambda)\cdot\nabla u) &= f(x,y;\lambda), \ (x,y)\in\Omega, \\
    u|_{\partial \Omega} &= 0,
  \end{cases}

where :math:`A(\lambda)=\text{diag}(1/\lambda_1,1/\lambda_2)`,
:math: `f(x,y;\lambda) = \pi^2 \sin(\pi x\lambda_1)\sin(\pi y \lambda_2)`,
and :math:`\Omega=[0,1]\times[0,1]`.

Probabilities
in the parameter space are calculated using emulated points.
1D and 2D marginals are calculated, smoothed, and plotted.
"""


import numpy as np
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam

from myModel import my_model

# Define the sampler that will be used to create the discretization
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem.
# The sampler and my_model is the interface of BET to the model,
# and it allows BET to create input/output samples of the model.
sampler = bsam.sampler(my_model)

# Initialize 2-dimensional input parameter sample set object
input_samples = samp.sample_set(2)

# Set parameter domain
input_samples.set_domain(np.array([[3.0, 6.0],
                                   [1.0, 5.0]]))



'''
Suggested changes for user:

Try with and without random sampling.

If using random sampling, try num_samples = 1E3 and 1E4.
What happens when num_samples = 1E2?
Try using 'lhs' instead of 'random' in the random_sample_set.

If using regular sampling, try different numbers of samples
per dimension.
'''
# Generate samples on the parameter space
randomSampling = False
if randomSampling is True:
    input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E4)
else:
    input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=[50, 50])

'''
Suggested changes for user:

A standard Monte Carlo (MC) assumption is that every Voronoi cell
has the same volume. If a regular grid of samples was used, then
the standard MC assumption is true.

See what happens if the MC assumption is not assumed to be true, and
if different numbers of points are used to estimate the volumes of
the Voronoi cells.
'''
MC_assumption = True
# Estimate volumes of Voronoi cells associated with the parameter samples
if MC_assumption is False:
    input_samples.estimate_volume(n_mc_points=1E5)
else:
    input_samples.estimate_volume_mc()

# Create the discretization object using the input samples
my_discretization = sampler.compute_QoI_and_create_discretization(input_samples,
                                               savefile = 'NonlinearExample.txt.gz')

'''
Suggested changes for user:

Try different reference parameters.
'''
# Define the reference parameter
param_ref = np.array([5.5, 4.5])
#param_ref = np.array([4.5, 3.0])
#param_ref = np.array([3.5, 1.5])

# Compute the reference QoI
Q_ref =  my_model(param_ref)

# Create some plots of input and output discretizations
plotD.scatter_2D(input_samples, ref_sample = param_ref,
                 filename = 'nonlinearMapParameterSamples',
                 file_extension='.eps')
if Q_ref.size == 2:
    plotD.show_data_domain_2D(my_discretization, Q_ref = Q_ref, file_extension=".eps")

'''
Suggested changes for user:

Try different ways of discretizing the probability measure on D defined
as a uniform probability measure on a rectangle or interval depending
on choice of QoI_num in myModel.py.
'''
randomDataDiscretization = False
if randomDataDiscretization is False:
    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.25,
        cells_per_dimension = 3)
else:
    simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.25,
        M=50, num_d_emulate=1E5)

# calculate probablities
calculateP.prob(my_discretization)

########################################
# Post-process the results
########################################
'''
Suggested changes for user:

At this point, the only thing that should change in the plotP.* inputs
should be either the nbins values or sigma (which influences the kernel
density estimation with smaller values implying a density estimate that
looks more like a histogram and larger values smoothing out the values
more).

There are ways to determine "optimal" smoothing parameters (e.g., see CV, GCV,
and other similar methods), but we have not incorporated these into the code
as lower-dimensional marginal plots generally have limited value in understanding
the structure of a high dimensional non-parametric probability measure.
'''
# calculate 2d marginal probs
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                        nbins = [20, 20])
# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.5)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "nomlinearMap",
                             lam_ref = param_ref, file_extension = ".eps", plot_surface=False)

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                        nbins = [20, 20])
# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.5)
# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "nonlinearMap",
                             lam_ref = param_ref, file_extension = ".eps")
