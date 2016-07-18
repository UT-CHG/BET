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

def linear_model2(parameter_samples):
    Q_map = np.array([[0.506],[0.253], [0.085]])
    QoI_samples = np.dot(parameter_samples,Q_map)
    return QoI_samples

def linear_model3(parameter_samples):
    Q_map = np.array([[0.506]])
    QoI_samples = np.dot(parameter_samples,Q_map)
    return QoI_samples

class Test_piecewise_polynomial_surrogate_3_to_2(unittest.TestCase):
    """
    Testing :meth:`bet.surrogates.piecewise_polynomial_surrogate` on a 
    3 to 2 map.

    """
    def setUp(self):
        """
        Setup map.
        """
        param_ref = np.array([0.5, 0.5, 0.5])
        Q_ref =  linear_model1(param_ref)
        
        sampler = bsam.sampler(linear_model1)
        input_samples = sample.sample_set(3)
        input_samples.set_domain(np.repeat([[0.0, 1.0]], 3, axis=0))
        input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E2)
        disc = sampler.compute_QoI_and_create_discretization(input_samples, 
                                                             globalize=True)
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=disc, Q_ref=Q_ref, rect_scale=0.5)
        num = disc.check_nums()
        disc._output_sample_set.set_error_estimates(0.01 * np.ones((num, 2)))
        jac = np.zeros((num,2,3))
        jac[:,:,:] = np.array([[0.506, 0.463],[0.253, 0.918], [0.085, 0.496]]).transpose()

        disc._input_sample_set.set_jacobians(jac)
        self.sur = surrogates.piecewise_polynomial_surrogate(disc)

    def Test_constants(self):
        """
        Test for piecewise constants.
        """
        iss = bsam.random_sample_set('r',
                                     self.sur.input_disc._input_sample_set._domain,
                                     num_samples = 10,
                                     globalize=False)
        sur_disc = self.sur.generate_for_input_set(iss, order=0)
        sur_disc.check_nums()
        self.assertEqual(sur_disc._output_sample_set._dim, 2)
        nptest.assert_array_equal(sur_disc._input_sample_set._domain,
                                  self.sur.input_disc._input_sample_set._domain)
        sur_disc._input_sample_set._values_local[0,:]
        s_set = sur_disc._input_sample_set.copy()
        sur_disc.set_io_ptr()
        regions_local = np.equal(sur_disc._io_ptr_local, 0)
        s_set.set_region_local(regions_local)
        s_set.local_to_global()
        s_set.check_num()
        self.sur.calculate_prob_for_sample_set_region(s_set, 
                                                      regions=[0],
                                                      update_input=True)
        
        
    def Test_linears(self):
        """
        Test for piecewise linears.
        """
        iss = bsam.random_sample_set('r',
                                     self.sur.input_disc._input_sample_set._domain,
                                     num_samples = 10,
                                     globalize=False)
        sur_disc = self.sur.generate_for_input_set(iss, order=1)
        sur_disc.check_nums()
        self.assertEqual(sur_disc._output_sample_set._dim, 2)
        nptest.assert_array_equal(sur_disc._input_sample_set._domain,
                                  self.sur.input_disc._input_sample_set._domain)
        sur_disc._input_sample_set._values_local[0,:]
        s_set = sur_disc._input_sample_set.copy()
        sur_disc.set_io_ptr()
        regions_local = np.equal(sur_disc._io_ptr_local, 0)
        s_set.set_region_local(regions_local)
        s_set.local_to_global()
        s_set.check_num()
        self.sur.calculate_prob_for_sample_set_region(s_set, 
                                                      regions=[0],
                                                      update_input=True)

class Test_piecewise_polynomial_surrogate_3_to_1(unittest.TestCase):
    """
    Testing :meth:`bet.surrogates.piecewise_polynomial_surrogate` on a 
    3 to 1 map.

    """
    def setUp(self):
        """
        Setup problem.
        """
        param_ref = np.array([0.5, 0.5, 0.5])
        Q_ref =  linear_model2(param_ref)
        
        sampler = bsam.sampler(linear_model2)
        input_samples = sample.sample_set(3)
        input_samples.set_domain(np.repeat([[0.0, 1.0]], 3, axis=0))
        input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E2)
        disc = sampler.compute_QoI_and_create_discretization(input_samples, 
                                                             globalize=True)
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=disc, Q_ref=Q_ref, rect_scale=0.5)
        num = disc.check_nums()
        disc._output_sample_set.set_error_estimates(0.01 * np.ones((num, 1)))
        jac = np.zeros((num,1,3))
        jac[:,:,:] = np.array([[0.506],[0.253], [0.085]]).transpose()

        disc._input_sample_set.set_jacobians(jac)
        self.sur = surrogates.piecewise_polynomial_surrogate(disc)

    def Test_constants(self):
        """
        Test for piecewise constants.
        """
        iss = bsam.random_sample_set('r',
                                     self.sur.input_disc._input_sample_set._domain,
                                     num_samples = 10,
                                     globalize=False)
        sur_disc = self.sur.generate_for_input_set(iss, order=0)
        sur_disc.check_nums()
        self.assertEqual(sur_disc._output_sample_set._dim, 1)
        nptest.assert_array_equal(sur_disc._input_sample_set._domain,
                                  self.sur.input_disc._input_sample_set._domain)
        sur_disc._input_sample_set._values_local[0,:]

        s_set = sur_disc._input_sample_set.copy()
        sur_disc.set_io_ptr()
        regions_local = np.equal(sur_disc._io_ptr_local, 0)
        s_set.set_region_local(regions_local)
        s_set.local_to_global()
        s_set.check_num()
        self.sur.calculate_prob_for_sample_set_region(s_set, 
                                                      regions=[0],
                                                      update_input=True)
        
        
    def Test_linears(self):
        """
        Test for piecewise linears.
        """
        iss = bsam.random_sample_set('r',
                                     self.sur.input_disc._input_sample_set._domain,
                                     num_samples = 10,
                                     globalize=False)
        sur_disc = self.sur.generate_for_input_set(iss, order=1)
        sur_disc.check_nums()
        self.assertEqual(sur_disc._output_sample_set._dim, 1)
        nptest.assert_array_equal(sur_disc._input_sample_set._domain,
                                  self.sur.input_disc._input_sample_set._domain)
        sur_disc._input_sample_set._values_local[0,:]

        s_set = sur_disc._input_sample_set.copy()
        sur_disc.set_io_ptr()
        regions_local = np.equal(sur_disc._io_ptr_local, 0)
        s_set.set_region_local(regions_local)
        s_set.local_to_global()
        s_set.check_num()
        self.sur.calculate_prob_for_sample_set_region(s_set, 
                                                      regions=[0],
                                                      update_input=True)

class Test_piecewise_polynomial_surrogate_1_to_1(unittest.TestCase):
    """
    Testing :meth:`bet.surrogates.piecewise_polynomial_surrogate` on a 
    1 to 1 map.

    """
    def setUp(self):
        """
        Setup maps
        """
        param_ref = np.array([0.5])
        Q_ref =  linear_model3(param_ref)
        
        sampler = bsam.sampler(linear_model3)
        input_samples = sample.sample_set(1)
        input_samples.set_domain(np.repeat([[0.0, 1.0]], 1, axis=0))
        input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E3)
        disc = sampler.compute_QoI_and_create_discretization(input_samples, 
                                                             globalize=True)
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=disc, Q_ref=Q_ref, rect_scale=0.5)
        num = disc.check_nums()
        disc._output_sample_set.set_error_estimates(0.01 * np.ones((num, 1)))
        jac = np.zeros((num,1,1))
        jac[:,:,:] = np.array([[0.506]]).transpose()

        disc._input_sample_set.set_jacobians(jac)
        self.sur = surrogates.piecewise_polynomial_surrogate(disc)

  
    def Test_constants(self):
        """
        Test methods for order 0 polynomials.
        """
        iss = bsam.random_sample_set('r',
                                     self.sur.input_disc._input_sample_set._domain,
                                     num_samples = 10,
                                     globalize=False)
        sur_disc = self.sur.generate_for_input_set(iss, order=0)
        sur_disc.check_nums()
        self.assertEqual(sur_disc._output_sample_set._dim, 1)
        nptest.assert_array_equal(sur_disc._input_sample_set._domain,
                                  self.sur.input_disc._input_sample_set._domain)
        sur_disc._input_sample_set._values_local[0,:]

        s_set = sur_disc._input_sample_set.copy()
        sur_disc.set_io_ptr()
        regions_local = np.equal(sur_disc._io_ptr_local, 0)
        s_set.set_region_local(regions_local)
        s_set.local_to_global()
        s_set.check_num()
        self.sur.calculate_prob_for_sample_set_region(s_set, 
                                                      regions=[0],
                                                      update_input=True)
        
        
    def Test_linears(self):
        """
        Test for piecewise linears.
        """
        iss = bsam.random_sample_set('r',
                                     self.sur.input_disc._input_sample_set._domain,
                                     num_samples = 10,
                                     globalize=False)
        sur_disc = self.sur.generate_for_input_set(iss, order=1)
        sur_disc.check_nums()
        self.assertEqual(sur_disc._output_sample_set._dim, 1)
        nptest.assert_array_equal(sur_disc._input_sample_set._domain,
                                  self.sur.input_disc._input_sample_set._domain)
        sur_disc._input_sample_set._values_local[0,:]

        s_set = sur_disc._input_sample_set.copy()
        sur_disc.set_io_ptr()
        regions_local = np.equal(sur_disc._io_ptr_local, 0)
        s_set.set_region_local(regions_local)
        s_set.local_to_global()
        s_set.check_num()
        self.sur.calculate_prob_for_sample_set_region(s_set, 
                                                      regions=[0],
                                                      update_input=True)
