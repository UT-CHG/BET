# Copyright (C) 2014-2020 The BET Development Team

"""
This module contains unittests for :mod:`~bet.calculateP.calculateR`
"""

import unittest
import os
import pyDOE
import numpy.testing as nptest
import numpy as np
import scipy.io as sio
import bet
import bet.sampling.basicSampling as bsam
from bet.Comm import comm
import bet.sample
from bet.sample import sample_set
from bet.sample import discretization as disc
import collections
from test.problem_setups import *


class Test_calculateR(unittest.TestCase):
    """
    Testing ``bet.calculateP.calculateR``
    """
    def setUp(self):
        self.in_dim = 1
        self.out_dim = 1
        self.vals = np.ones((10, ))
        self.vals_marg= np.ones((10, ))

    def test_kde(self):
        """
        Test ``bet.calculateP.calculateR.invert_to_kde``
        """
        disc, _ = random_kde(dim=self.in_dim, out_dim=self.out_dim, level=2)
        disc.get_input_sample_set().pdf(self.vals)
        disc.get_input_sample_set().pdf_init(self.vals)
        disc.get_input_sample_set().marginal_pdf(self.vals, i=0)
        disc.get_input_sample_set().marginal_pdf_init(self.vals, i=0)
        disc.get_input_sample_set().marginal_pdf(self.vals_marg, i=0)
        disc.get_input_sample_set().marginal_pdf_init(self.vals_marg, i=0)

    def test_rv(self):
        """
        Test ``bet.calculateP.calculateR.invert_to_random_variable``
        """
        disc, _ = random_rv(dim=self.in_dim, out_dim=self.out_dim, level=2)
        disc.get_input_sample_set().pdf(self.vals)
        disc.get_input_sample_set().pdf_init(self.vals)
        disc.get_input_sample_set().marginal_pdf(self.vals, i=0)
        disc.get_input_sample_set().marginal_pdf_init(self.vals, i=0)
        disc.get_input_sample_set().marginal_pdf(self.vals_marg, i=0)
        disc.get_input_sample_set().marginal_pdf_init(self.vals_marg, i=0)

    def test_gmm(self):
        """
        Test ``bet.calculateP.calculateR.invert_to_gmm``
        """
        disc, _ = random_gmm(dim=self.in_dim, out_dim=self.out_dim, level=2)
        disc.get_input_sample_set().pdf(self.vals)
        disc.get_input_sample_set().pdf_init(self.vals)
        disc.get_input_sample_set().marginal_pdf(self.vals, i=0)
        disc.get_input_sample_set().marginal_pdf_init(self.vals, i=0)
        disc.get_input_sample_set().marginal_pdf(self.vals_marg, i=0)
        disc.get_input_sample_set().marginal_pdf_init(self.vals_marg, i=0)

    def test_multivariate_gaussian(self):
        """
        Test ``bet.calculateP.calculateR.invert_to_multivariate_gaussian``
        """
        disc, _ = random_multivariate_gaussian(dim=self.in_dim, out_dim=self.out_dim, level=2)
        disc.get_input_sample_set().pdf(self.vals)
        disc.get_input_sample_set().pdf_init(self.vals)
        disc.get_input_sample_set().marginal_pdf(self.vals, i=0)
        disc.get_input_sample_set().marginal_pdf_init(self.vals, i=0)
        disc.get_input_sample_set().marginal_pdf(self.vals_marg, i=0)
        disc.get_input_sample_set().marginal_pdf_init(self.vals_marg, i=0)

class Test_calculateR_3to2(Test_calculateR):
    """
    Testing ``bet.calculateP.calculateR`` with a 3 to 2 map.
    """
    def setUp(self):
        self.in_dim = 3
        self.out_dim = 3
        self.vals = np.ones((10, 3))
        self.vals_marg= np.ones((10, ))


class Test_invert_to_random_variable(unittest.TestCase):
    """
    Test `bet.calculateP.calculateR.invert_to_random_variable`
    """
    def test_string(self):
        """
        Test when rv is a string.
        """
        random_rv(dim=2, out_dim=1, rv_invert='beta', level=2)

    def test_list1(self):
        """
        Test when rv is a list of length 2.
        """
        random_rv(dim=2, out_dim=1, rv_invert=['beta', {'loc': 0.25}], level=2)

    def test_list2(self):
        """
        Test when rv is a list of lists.
        """
        random_rv(dim=2, out_dim=1, rv_invert=[['beta', {'floc': 0.25}], ['norm', {}]], level=2)

    def test_sample_from_updated(self):
        disc, _ = random_rv(dim=2, out_dim=1, rv_invert=[['beta', {'floc': 0.25}], ['norm', {}]], level=2)
        new_set = bsam.sample_from_updated(disc, num_samples=100)
        assert new_set.get_dim() == 2
        assert new_set.check_num() == 100

        disc, _ = random_gmm(dim=2, out_dim=1, level=2)
        new_set = bsam.sample_from_updated(disc, num_samples=100)
        assert new_set.get_dim() == 2
        assert new_set.check_num() == 100

        disc, _ = random_kde(dim=2, out_dim=1, level=2)
        new_set = bsam.sample_from_updated(disc, num_samples=100)
        assert new_set.get_dim() == 2
        assert new_set.check_num() == 100
        disc.global_to_local()




class Test_rejection_sampling(unittest.TestCase):
    def Test_rejection_sampling(self):
        """
        Testing ``bet.calculateP.calculateR.invert_rejection_sampling``
        """
        rv = 'uniform'
        dim = 1
        out_dim = 1
        num_samples = 1000
        globalize = True
        rv2 = "norm"
        def my_model(samples):
            A = np.eye(dim, out_dim)
            return np.dot(samples, A)

        sampler1 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler1.random_sample_set(rv, dim, num_samples, globalize)
        disc1 = sampler1.compute_qoi_and_create_discretization()

        sampler2 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler2.random_sample_set(rv2, dim, num_samples, globalize)
        disc2 = sampler1.compute_qoi_and_create_discretization()

        disc1.set_output_observed_set(disc2.get_output_sample_set())
        calculateR.invert_rejection_sampling(disc1)






