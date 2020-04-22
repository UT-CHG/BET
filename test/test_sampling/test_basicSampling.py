# Copyright (C) 2014-2019 The BET Development Team

"""
This module contains unittests for :mod:`~bet.sampling.basicSampling:`
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

local_path = os.path.join(".")


class Test_random_sample_set1to1(unittest.TestCase):
    """
    Testing ``bet.sampling.basicSampling.random_sample_set``
    """
    def setUp(self):
        self.input_dim = 1
        self.output_dim = 1
        self.num = 100
        self.set = random_voronoi(rv='uniform', dim=self.input_dim, out_dim=self.output_dim,
                                  num_samples=self.num, level=1)

    def test_nums(self):
        """
        Test for correct dimensions and sizes.
        """
        assert self.set.get_dim() == self.input_dim
        assert self.set.get_values().shape[0] == self.num

class Test_random_sample_set3to2(Test_random_sample_set1to1):
    """
    Testing ``bet.sampling.basicSampling.random_sample_set``
    """
    def setUp(self):
        self.input_dim = 3
        self.output_dim = 2
        self.num = 100
        self.set = random_voronoi(rv=['beta', {'a': 1, 'b': 3}], dim=self.input_dim, out_dim=self.output_dim,
                                  num_samples=self.num, level=1)


class Test_random_sample_set2to1(Test_random_sample_set1to1):
    """
    Testing ``bet.sampling.basicSampling.random_sample_set``
    """
    def setUp(self):
        self.input_dim = 2
        self.output_dim = 1
        self.num = 1
        self.set = random_voronoi(rv=[['beta', {'a': 1, 'b': 3}], ['norm', {'scale': 3}]],
                                  dim=self.input_dim, out_dim=self.output_dim,
                                  num_samples=self.num, level=1)

class Test_regular_sample(unittest.TestCase):
    """
    Testing ``bet.sampling.basicSampling.regular_sample_set``
    """

    def setUp(self):
        self.input_dim = 2
        self.output_dim = 1
        self.num = 3
        self.set = regular_voronoi(dim=self.input_dim, out_dim=self.output_dim, num_samples_per_dim=self.num)

    def test_nums(self):
        """
        Test for correct dimensions and sizes.
        """
        assert self.set.get_dim() == self.input_dim
        assert self.set.get_values().shape[0] == self.num ** self.input_dim

    def test_domain(self):
        """
        Test that values are in correct domain.
        """
        assert np.all(np.greater_equal(self.set.get_values(), 0.0))
        assert np.all(np.less_equal(self.set.get_values(), 1.0))

class Test_lhs_sample(Test_random_sample_set1to1):
    """
    Testing ``bet.sampling.basicSampling.lhs_sample_set``
    """

    def setUp(self):
        self.input_dim = 2
        self.output_dim = 1
        self.num = 3
        self.set = lhs_voronoi(dim=self.input_dim, out_dim=self.output_dim, num_samples=self.num)

    def test_domain(self):
        """
        Test that values are in correct domain.
        """
        assert np.all(np.greater_equal(self.set.get_values(), 0.0))
        assert np.all(np.less_equal(self.set.get_values(), 1.0))


class Test_sampler(unittest.TestCase):
    """
    Testing ``bet.sampling.basicSampling.sampler``
    """
    def setUp(self):
        self.input_dim = 2
        self.output_dim = 2
        self.num = 100
        self.sampler = random_voronoi(rv='uniform', dim=self.input_dim, out_dim=self.output_dim,
                                  num_samples=self.num, level=3)

    def test_nums(self):
        """
        Test for correct dimensions and sizes.
        """
        assert self.sampler.discretization.get_input_sample_set().get_dim() == self.input_dim
        assert self.sampler.discretization.get_output_sample_set().get_dim() == self.output_dim
        assert self.sampler.discretization.check_nums() == self.num

    def test_copy(self):
        """
        Test copying
        """
        sampler2 = self.sampler.copy()
        assert sampler2.discretization == self.sampler.discretization
        assert sampler2.lb_model == self.sampler.lb_model

    def test_values(self):
        """
        Check for correct values.
        """
        nptest.assert_almost_equal(self.sampler.discretization.get_input_sample_set().get_values(),
                                   self.sampler.discretization.get_output_sample_set().get_values())


class Test_sampler_regular(Test_sampler):
    """
    Testing ``bet.sampling.basicSampling.sampler``
    """
    def setUp(self):
        self.input_dim = 2
        self.output_dim = 2
        self.num = 3
        self.sampler = regular_voronoi(dim=self.input_dim, out_dim=self.output_dim,
                                       num_samples_per_dim=self.num, level=3)

    def test_nums(self):
        """
        Test for correct dimensions and sizes.
        """
        assert self.sampler.discretization.get_input_sample_set().get_dim() == self.input_dim
        assert self.sampler.discretization.get_output_sample_set().get_dim() == self.output_dim
        assert self.sampler.discretization.check_nums() == self.num ** self.input_dim




class Test_sampler_lhs(Test_sampler):
    """
    Testing ``bet.sampling.basicSampling.sampler``
    """
    def setUp(self):
        self.input_dim = 2
        self.output_dim = 2
        self.num = 30
        self.sampler = lhs_voronoi(dim=self.input_dim, out_dim=self.output_dim, num_samples=self.num, level=3)


