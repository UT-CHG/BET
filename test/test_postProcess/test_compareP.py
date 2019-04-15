# Copyright (C) 2014-2019 The BET Development Team

import numpy as np
# import numpy.testing as nptest
import unittest
# import os
# import glob
import bet.sample as sample
import bet.postProcess.compareP as compP
import bet.sampling.basicSampler as bsam
# import bet.util as util
# from bet.Comm import comm, MPI

# local_path = os.path.join(os.path.dirname(bet.__file__), "/test")
local_path = ''

def set_unit_probs(num_samples=100,
                   dim=2,
                   delta=0.1):
    s_set = sample.sample_set(dim)
    s_set.set_domain(np.array([[0,1]*dim]))
    s = bsam.random_sample_set('r', s_set, num_samples)
    dd = delta/2.0
    probs = 1*(np.sum(np.logical_and(s._values <= (0.5+dd), 
                      s._values >= (0.5-dd)),axis=1) >= dim-1) 
    s.set_probabilities(probs)/np.sum(probs) # uniform probabilities
    s.global_to_local()

def check_densities(s_set, dim=2, delta=0.1, tol=1e-4):
    # density values should be reciprocal of delta^dim
    true_den_val = 1.0/(delta**dim)
    if np.mean(np.abs(s_set._den - true_den_val)) < tol:
        return 1
    else:
        return 0

# class Test_distance(unittest.TestCase):
#     def setUp(self):
#         self.dim = 1
#         self.integration_set = sample.sample_set(dim=self.dim)
#         self.left_set = sample.sample_set(dim=self.dim)
#         self.right_set = sample.sample_set(dim=self.dim)
#         self.num1, self.num2, self.num = 100, 100, 500
#         values = np.ones((self.num, self.dim))
#         values1 = np.ones((self.num1, self.dim))
#         values2 = np.ones((self.num2, self.dim))
#         self.integration_set.set_values(values)
#         self.left_set.set_values(values1)
#         self.right_set.set_values(values2)

#     def test_identity(self):
#         r"""
#         Ensure passing identical sets returns 0 distance.
#         """
#         compP.distance(self.left_set, self.left_set)


class Test_metrization_simple(unittest.TestCase):
    def setUp(self):
        self.dim = 3
        self.integration_set = sample.sample_set(dim=self.dim)
        self.left_set = sample.sample_set(dim=self.dim)
        self.right_set = sample.sample_set(dim=self.dim)
        self.num1, self.num2, self.num = 100, 100, 500
        values = np.ones((self.num, self.dim))
        values1 = np.ones((self.num1, self.dim))
        values2 = np.ones((self.num2, self.dim))
        self.integration_set.set_values(values)
        self.left_set.set_values(values1)
        self.right_set.set_values(values2)
        self.domain = np.tile([0, 1], [self.dim, 1])
        self.integration_set.set_domain(self.domain)
        self.left_set.set_domain(self.domain)
        self.right_set.set_domain(self.domain)

        self.mtrc = compP.metrization(sample_set_left=self.left_set,
                                      sample_set_right=self.right_set,
                                      integration_sample_set=self.integration_set)

    def test_dimension(self):
        r"""
        Check that improperly setting dimension raises warning.
        """
        dim = self.dim+1
        values = np.ones((200, dim))
        integration_set = sample.sample_set(dim=dim)
        integration_set.set_values(values)
        integration_set.set_domain(np.tile([0, 1], [dim, 1]))

        try:
            compP.metrization(sample_set_left=self.left_set,
                              sample_set_right=self.right_set,
                              integration_sample_set=self.integration_set)
        except sample.dim_not_matching:
            # setting wrong shapes should raise this error
            print('caught')
            pass

    def test_domain(self):
        r"""
        Check that improperly setting domain raises warning.
        """
        test_set = self.integration_set.copy()
        test_set.set_domain(test_set.get_domain()+0.01)
        test_metr = [compP.metrization(
            integration_sample_set=self.integration_set),
            compP.metrization(
            None, sample_set_right=self.integration_set),
            compP.metrization(
            None, sample_set_left=self.integration_set)
        ]
        for mm in test_metr:
            test_funs = [mm.set_right,
                         mm.set_left,
                         mm.set_int]
            for fun in test_funs:
                try:
                    fun(test_set)
                except AttributeError:
                    pass

    def test_copy_clip_merge_slice(self):
        r"""
        Test copying, clipping, merging, slicing
        """
        mm = self.mtrc.copy()
        mc = mm.clip(50)
        ms = self.mtrc.merge(mc)
        ms.slice([0])
        ms.slice([1, 0])
        ms.slice([1, 0, 1])  # can repeat dimensions if you want?
        if self.dim > 2:
            ms.slice([2, 0, 1])
            ms.slice([1, 2, 0, 0])
            ms.slice([1, 2])
            ms.slice([0, 1])

    def test_missing_domain(self):
        r"""
        Make sure we can initialize the function in several permutations
        if the domain is missing from the integration set
        """
        test_set = sample.sample_set(dim=self.dim)  # no domain info
        other_set = test_set.copy()  # has domain info
        other_set.set_domain(self.domain)
        mm = compP.metrization(None, other_set)
        mm = compP.metrization(None, None, other_set)
        mm = compP.metrization(test_set, other_set, None)
        mm = compP.metrization(test_set, None, other_set)
        mm = compP.metrization(test_set, None, None)
        mm.set_left(other_set)
        try:  # we are missing a set, so this should fail
            mm.check_domain()
        except AttributeError:
            pass

        self.mtrc.set_right(other_set)
        self.mtrc.check_domain()  # now we expect it to pass

        # the following should error out because not enough information
        try:
            self.mtrc = compP.metrization(None)
        except AttributeError:
            pass
        try:
            self.mtrc = compP.metrization(None, None, test_set)
        except AttributeError:
            pass
        try:
            self.mtrc = compP.metrization(test_set, None, other_set)
        except AttributeError:
            pass

    def test_no_sample_set(self):
        r"""
        Make sure we can initialize the function in several permutations
        """
        test_set = sample.sample_set(dim=self.dim)
        test_set.set_domain(self.domain)
        other_set = test_set.copy()
        self.mtrc = compP.metrization(test_set)
        self.mtrc = compP.metrization(test_set, None)
        self.mtrc = compP.metrization(test_set, None, other_set)
        self.mtrc = compP.metrization(test_set, other_set, None)
        self.mtrc = compP.metrization(test_set, None, None)

    # TO DO: test left and right missing domains, inferred from others.
    def test_set_ptr_left(self):
        r"""
        Test setting left io ptr
        """
        # TODO be careful if we change Kdtree
        self.mtrc.set_io_ptr_left(globalize=True)
        self.mtrc.get_io_ptr_left()
        self.mtrc.set_io_ptr_left(globalize=False)
        self.mtrc.get_io_ptr_left()
        self.mtrc.globalize_ptrs()

    def test_set_ptr_right(self):
        """
        Test setting right io ptr
        """
        # TODO be careful if we change Kdtree
        self.mtrc.set_io_ptr_right(globalize=True)
        self.mtrc.get_io_ptr_right()
        self.mtrc.set_io_ptr_right(globalize=False)
        self.mtrc.get_io_ptr_right()
        self.mtrc.globalize_ptrs()

    def test_set_right(self):
        self.mtrc.set_right(self.right_set)
        assert self.right_set == self.right_set

    def test_set_left(self):
        self.mtrc.set_left(self.left_set)
        assert self.left_set == self.left_set

    def test_get_right(self):
        set_right = self.mtrc.get_right()
        assert set_right == self.right_set

    def test_get_left(self):
        set_left = self.mtrc.get_left()
        assert set_left == self.left_set
