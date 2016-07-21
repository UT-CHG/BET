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
import bet.calculateP.calculateError as calculateError
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


class calculate_error(object):
    def Test_sampling_error(self):
        """
        Testing :meth:`bet.calculateP.calculateError.sampling_error`
        """
        num = self.disc.check_nums()
        neiList = calculateError.cell_connectivity_exact(self.disc)
        for i in range(num):
            self.assertGreater(len(neiList[i]), 0)
        (B_N, C_N) = calculateError.boundary_sets(self.disc, neiList)
        s_error = calculateError.sampling_error(self.disc, exact=True)
        (upper, lower) = s_error.calculate_for_contour_events()
        for x in upper:
            if not np.isnan(x):
                self.assertGreaterEqual(x, 0.0)
        for x in lower:
            if not np.isnan(x):
                self.assertLessEqual(x, 0.0)

        s_set = self.disc._input_sample_set.copy()
        regions_local = np.equal(self.disc._io_ptr_local, 0)
        s_set.set_region_local(regions_local)
        s_set.local_to_global()
        (up, low) = s_error.calculate_for_sample_set_region(s_set, 
                                                            1)         
        self.assertAlmostEqual(up, upper[0])
        self.assertAlmostEqual(low, lower[0]) 

        (up, low) = s_error.calculate_for_sample_set_region(s_set, 
                                                            1,
                                                            emulated_set=self.disc._input_sample_set)         
        self.assertAlmostEqual(up, upper[0])
        self.assertAlmostEqual(low, lower[0])

        self.disc.set_emulated_input_sample_set(self.disc._input_sample_set)
        (up, low) = s_error.calculate_for_sample_set_region(s_set, 
                                                            1)         
        self.assertAlmostEqual(up, upper[0])
        self.assertAlmostEqual(low, lower[0])

    def Test_model_error(self):
        """
        Testing :meth:`bet.calculateP.calculateError.model_error`
        """
        num = self.disc.check_nums()
        m_error = calculateError.model_error(self.disc)
        er_est = m_error.calculate_for_contour_events()

        s_set = self.disc._input_sample_set.copy()
        regions_local = np.equal(self.disc._io_ptr_local, 0)
        s_set.set_region_local(regions_local)
        s_set.local_to_global()
        er_est2 = m_error.calculate_for_sample_set_region(s_set, 
                                                    1)
        self.assertAlmostEqual(er_est[0], er_est2)
        error_id_sum_local = np.sum(self.disc._input_sample_set._error_id_local)
        error_id_sum = comm.allreduce(error_id_sum_local, op=MPI.SUM)
        self.assertAlmostEqual(er_est2, error_id_sum)

        emulated_set=self.disc._input_sample_set
        er_est3 = m_error.calculate_for_sample_set_region(s_set, 
                                                            1,
                                                            emulated_set=emulated_set)
        self.assertAlmostEqual(er_est[0], er_est3)
        self.disc.set_emulated_input_sample_set(self.disc._input_sample_set)
        m_error = calculateError.model_error(self.disc)
        er_est4 = m_error.calculate_for_sample_set_region(s_set, 
                                                    1)
        self.assertAlmostEqual(er_est[0], er_est4)

        

class Test_3_to_2(calculate_error, unittest.TestCase):
    """
    Testing :meth:`bet.calculateP.calculateError` on a 
    3 to 2 map.
    """
    def setUp(self):
        param_ref = np.array([0.5, 0.5, 0.5])
        Q_ref =  linear_model1(param_ref)
        
        sampler = bsam.sampler(linear_model1)
        input_samples = sample.sample_set(3)
        input_samples.set_domain(np.repeat([[0.0, 1.0]], 3, axis=0))
        input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E3)
        disc = sampler.compute_QoI_and_create_discretization(input_samples, 
                                                             globalize=True)
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=disc, Q_ref=Q_ref, rect_scale=0.5)
        num = disc.check_nums()
        disc._output_sample_set.set_error_estimates(0.01 * np.ones((num, 2)))
        jac = np.zeros((num,2,3))
        jac[:,:,:] = np.array([[0.506, 0.463],[0.253, 0.918], [0.085, 0.496]]).transpose()

        disc._input_sample_set.set_jacobians(jac)
        self.disc = disc
class Test_3_to_1(calculate_error, unittest.TestCase):
    """
    Testing :meth:`bet.calculateP.calculateError` on a 
    3 to 1 map.
    """
    def setUp(self):
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
        self.disc = disc
class Test_1_to_1(calculate_error, unittest.TestCase):
    """
    Testing :meth:`bet.calculateP.calculateError` on a 
    1 to 1 map.
    """
    def setUp(self):
        param_ref = np.array([0.5])
        Q_ref =  linear_model3(param_ref)
        
        sampler = bsam.sampler(linear_model3)
        input_samples = sample.sample_set(1)
        input_samples.set_domain(np.repeat([[0.0, 1.0]], 1, axis=0))
        input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E2)
        disc = sampler.compute_QoI_and_create_discretization(input_samples, 
                                                             globalize=True)
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=disc, Q_ref=Q_ref, rect_scale=0.5)
        num = disc.check_nums()
        disc._output_sample_set.set_error_estimates(0.01 * np.ones((num, 1)))
        jac = np.zeros((num,1,1))
        jac[:,:,:] = np.array([[0.506]]).transpose()

        disc._input_sample_set.set_jacobians(jac)
        self.disc = disc

