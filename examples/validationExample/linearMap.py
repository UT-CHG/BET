#! /usr/bin/env python

# Copyright (C) 2014-2016 The BET Development Team

"""
This 2D linear example verifies that geometrically distinct QoI can
recreate a probability measure on the input parameter space
used to define the output probability measure. 
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

# Initialize 3-dimensional input parameter sample set object
input_samples = samp.sample_set(2)

# Set parameter domain
input_samples.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))

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
    input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=[30, 30])

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
                                               savefile = 'Validation_discretization.txt.gz')

'''
Compute the output distribution simple function approximation by
propagating a different set of samples to implicitly define a Voronoi
discretization of the data space, corresponding to an implicitly defined
set of contour events defining a discretization of the input parameter
space. The probabilities of the Voronoi cells in the data space (and
thus the probabilities of the corresponding contour events in the
input parameter space) are determined by Monte Carlo sampling using
a set of i.i.d. uniform samples to bin into these cells.

Suggested changes for user:

See the effect of using different values for num_samples_discretize_D.
Choosing

  num_samples_discretize_D = 1

produces exactly the right answer and is equivalent to assigning a
uniform probability to each data sample above (why?).

Try setting this to 2, 5, 10, 50, and 100. Can you explain what you
are seeing? To see an exaggerated effect, try using random sampling
above with n_samples set to 1E2.
'''
num_samples_discretize_D = 1
num_iid_samples = 1E5

Partition_set = samp.sample_set(2)
Monte_Carlo_set = samp.sample_set(2)

Partition_set.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))
Monte_Carlo_set.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))

Partition_discretization = sampler.create_random_discretization('random',
                                                            Partition_set,
                                                            num_samples=num_samples_discretize_D)

Monte_Carlo_discretization = sampler.create_random_discretization('random',
                                                            Monte_Carlo_set,
                                                            num_samples=num_iid_samples)

# Compute the simple function approximation to the distribution on the data space
simpleFunP.user_partition_user_distribution(my_discretization,
                                            Partition_discretization,
                                            Monte_Carlo_discretization)

# Calculate probabilities
calculateP.prob(my_discretization)

########################################
# Post-process the results (optional)
########################################
# Show some plots of the different sample sets
plotD.scatter_2D(my_discretization._input_sample_set, filename = 'Parameter_Samples.eps')
plotD.scatter_2D(my_discretization._output_sample_set, filename = 'QoI_Samples.eps')
plotD.scatter_2D(my_discretization._output_probability_set, filename = 'Data_Space_Discretization.eps')
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
                                                        nbins = [30, 30])

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "validation_raw",
                             file_extension = ".eps", plot_surface=False)

# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.1)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "validation_smooth",
                             file_extension = ".eps", plot_surface=False)

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                        nbins = [30, 30])

# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "validation_raw",
                             file_extension = ".eps")

# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.1)

# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "validation_smooth",
                             file_extension = ".eps")




