#! /usr/bin/env python

# Copyright (C) 2014-2016 The BET Development Team

r"""

Creates uniformly distributed samples on a unit cube and evaluates a
3d to 2d nonlinear map. The map has normal error added to it, and the
difference between the true model and the model with error is added
as an error estimate. Jacobians are also added.
Assuming an output probability as uniform on a 
rectangle, the stochastic inverse problem is solved. The error in the
solution over a defined box due to sampling is approximated with 
error bounds. The error due to solving the model inexactly is also 
approximated.

Piecewise constant and piecewise linear (using Jacobians) surrogates
are generated and used to calculate probabilities and error estimates.
Effectivity ratios are calculated.

"""

import numpy as np
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.calculateP.calculateError as calculateError
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import bet.surrogates as surrogates
from bet.Comm import comm
from lbModel import lb_model_exact, lb_model


# Define the reference parameter
param_ref = np.array([[0.5, 0.5, 0.5]])
(Q_ref, _) = lb_model_exact(param_ref)

# Interface BET to the approximate model and create discretization object.
sampler = bsam.sampler(lb_model, error_estimates=True, jacobians=True)
input_samples = samp.sample_set(3)
input_samples.set_domain(np.array([[0.0, 1.0],
                                   [0.0, 1.0],
                                   [0.0, 1.0]]))
my_disc = sampler.create_random_discretization("random", input_samples, num_samples=1000)

# Define output probability
rect_domain = np.array([[0.5, 1.5], [1.25, 2.25]])
simpleFunP.regular_partition_uniform_distribution_rectangle_domain( data_set=my_disc, rect_domain=rect_domain)

# Make emulated input sets
emulated_inputs = bsam.random_sample_set('r',
                                         my_disc._input_sample_set._domain,
                                         num_samples = 10001,
                                         globalize=False)

emulated_inputs2 = bsam.random_sample_set('r',
                                         my_disc._input_sample_set._domain,
                                         num_samples = 10001,
                                          globalize=False)


# Make exact discretization
my_disc_exact = my_disc.copy()
my_disc_exact._output_sample_set._values_local = my_disc._output_sample_set._values_local + my_disc._output_sample_set._error_estimates_local
my_disc_exact._output_sample_set._error_estimates_local = np.zeros(my_disc_exact._output_sample_set._error_estimates_local.shape)
my_disc_exact._output_sample_set.local_to_global()
my_disc_exact.set_emulated_input_sample_set(emulated_inputs2)

# Solve stochastic inverse problems
my_disc.set_emulated_input_sample_set(emulated_inputs)
calculateP.prob_with_emulated_volumes(my_disc)
calculateP.prob_with_emulated_volumes(my_disc_exact)

# Set in input space to calculate errors for
s_set = samp.rectangle_sample_set(3)
s_set.setup(maxes=[[0.75, 0.75, 0.75]], mins=[[0.25, 0.25, 0.25]])
s_set.set_region(np.array([0,1]))

# Calculate true probabilty using linear surrogate on true data
sur = surrogates.piecewise_polynomial_surrogate(my_disc_exact)
sur_disc_lin = sur.generate_for_input_set(emulated_inputs, order=1)
(Pt, _) = sur.calculate_prob_for_sample_set_region(s_set, 
                                                         regions=[0])
P_true = Pt[0]
if comm.rank == 0:
    print "True probability: ",  P_true

# Approximate probabilty
P_approx  = calculateP.prob_from_sample_set(my_disc._input_sample_set,
                                             s_set)[0]
if comm.rank == 0:
    print "Approximate probability: ",  P_approx

# Calculate sampling error
samp_er = calculateError.sampling_error(my_disc)
(up, low) = samp_er.calculate_for_sample_set_region(s_set, 
                                                    region=0, emulated_set=emulated_inputs)
if comm.rank == 0:
    print "Sampling error upper bound: ", up
    print "Sampling error lower bound: ", low


# Calculate modeling error
mod_er = calculateError.model_error(my_disc)
er_est = mod_er.calculate_for_sample_set_region(s_set, 
                    region=0, emulated_set=emulated_inputs)
if comm.rank == 0:
    print "Modeling error estimate: ", er_est

# Make piecewise constant surrogate of discretization
sur = surrogates.piecewise_polynomial_surrogate(my_disc)
sur_disc_constant = sur.generate_for_input_set(emulated_inputs, order=0)
# Calculate probability and error estimate
(P2, er_est2) = sur.calculate_prob_for_sample_set_region(s_set, 
                                                         regions=[0])
if comm.rank == 0:
    print "Piecewise constant surrogate probability: ", P2[0]
    print "Piecewise constant error estimate ",  er_est2[0]
    print "Piecewise constant corrected probability: ", P2[0]-er_est2[0]
    print "Piecewise constant effectivity ratio: ", er_est2[0]/(P2[0]-P_true)

# Make piecewise linear surrogate of discretization
sur_disc_linear = sur.generate_for_input_set(emulated_inputs, order=1)
# Calculate probability and error estimate
(P3, er_est3) = sur.calculate_prob_for_sample_set_region(s_set, 
                                                         regions=[0])
if comm.rank == 0:
    print "Piecewise constant surrogate probability: ", P3[0]
    print "Piecewise constant error estimate ",  er_est3[0]
    print "Piecewise constant corrected probability: ", P3[0]-er_est3[0]
    print "Piecewise constant effectivity ratio: ", er_est3[0]/(P3[0]-P_true)

