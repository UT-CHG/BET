#! /usr/bin/env python

# Copyright (C) 2014-2016 The BET Development Team

"""
This example solves a stochastic inverse problem for a
linear 3-to-2 map. We refer to the map as the QoI map,
or just a QoI. We refer to the range of the QoI map as
the data space.
The 3-D input space is discretized with i.i.d. uniform
random samples or a regular grid of samples.
We refer to the input space as the
parameter space, and use parameter to refer to a particular
point (e.g., a particular random sample) in this space.
A reference parameter is used to define a reference QoI datum
and a uniform probability measure is defined on a small box
centered at this datum.
The measure on the data space is discretized either randomly
or deterministically, and this discretized measure is then
inverted by BET to determine a probability measure on the
parameter space whose support contains the measurable sets
of probable parameters.
We use emulation to estimate the measures of sets defined by
the random discretizations.
1D and 2D marginals are calculated, smoothed, and plotted.
"""

import numpy as np
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam
from myModel import my_model

# Initialize 3-dimensional input parameter sample set object
input_samples = samp.sample_set(3)

# Set parameter domain
input_samples.set_domain(np.repeat([[0.0, 1.0]], 3, axis=0))

# Define the sampler that will be used to create the discretization
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem
sampler = bsam.sampler(my_model)

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
    input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E3)
else:
    input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=[15, 15, 10])

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
                                               savefile = '3to2_discretization.txt.gz')

'''
Suggested changes for user:

Try different reference parameters.
'''
# Define the reference parameter
param_ref = np.array([0.5, 0.5, 0.5])
#param_ref = np.array([0.75, 0.75, 0.5])
#param_ref = np.array([0.75, 0.75, 0.75])
#param_ref = np.array([0.5, 0.5, 0.75])

# Compute the reference QoI
Q_ref =  my_model(param_ref)

# Create some plots of input and output discretizations
plotD.scatter_2D_multi(input_samples, ref_sample= param_ref, showdim = 'all', filename = 'linearMapParameterSamples')
plotD.scatter_rhoD(my_discretization, ref_sample = Q_ref, io_flag='output')

'''
Suggested changes for user:

Try different ways of discretizing the probability measure on D defined as a uniform
probability measure on a rectangle (since D is 2-dimensional) centered at Q_ref whose
size is determined by scaling the circumscribing box of D.
'''
randomDataDiscretization = False
if randomDataDiscretization is False:
    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.25,
        center_pts_per_edge = 3)
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
                                                        nbins = [10, 10, 10])

# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.2)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "linearMap",
                             lam_ref=param_ref, file_extension = ".eps", plot_surface=False)

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                        nbins = [10, 10, 10])
# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.2)
# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "linearMap",
                             lam_ref=param_ref, file_extension = ".eps")
