# Copyright (C) 2016 The BET Development Team


import unittest, os, glob
import numpy as np
import numpy.testing as nptest
import bet
import bet.sample as sample
import bet.util as util
import bet.surrogates as surrogates
import bet.sampling.basicSampling as bsam
import bet.calculateP.simpleFunP as simpleFunP
from bet.Comm import comm, MPI

def linear_model1(parameter_samples):
    Q_map = np.array([[0.506, 0.463],[0.253, 0.918], [0.085, 0.496]])
    QoI_samples = np.dot(parameter_samples,Q_map)
    return QoI_samples

class Test_piecewise_polynomial_surrogate_3_to_2(unittest.TestCase):
    def setUp(self):
        param_ref = np.array([0.5, 0.5, 0.5])
        Q_ref =  linear_model1(param_ref)
        
        sampler = bsam.sampler(linear_model1)
        input_samples = sample.sample_set(3)
        input_samples.set_domain(np.repeat([[0.0, 1.0]], 3, axis=0))
        input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E2)
        disc = sampler.compute_QoI_and_create_discretization(input_samples, 
                                                             globalize=True)
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=disc, Q_ref=Q_ref, rect_scale=0.25)
        num = disc.check_nums()
        disc._input_sample_set.set_error_estimates(0.1 * np.ones((num, 2)))
        jac = np.zeros((num,2,3))
        jac[:,:,:] = np.array([[0.506, 0.463],[0.253, 0.918], [0.085, 0.496]]).transpose()
        disc._input_sample_set.set_jacobians(jac)
        self.sur = surrogates.piecewise_polynomial_surrogate(disc)

    def Test_constants(self):
        iss = bsam.random_sample_set('r',
                                     self.sur.input_disc._input_sample_set._domain,
                                     num_samples = 10,
                                     globalize=False)
        sur_disc = self.sur.generate_for_input_set(iss, order=0)
        self.assertEqual(sur_disc._output_sample_set._dim, 2)
        
        
        
