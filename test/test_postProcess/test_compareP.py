# Copyright (C) 2014-2019 The BET Development Team

import numpy as np
import numpy.testing as nptest
import unittest
# import os
# import glob
import bet.sample as sample
import bet.postProcess.compareP as compP
import bet.sampling.basicSampling as bsam
# import bet.util as util
# from bet.Comm import comm, MPI

# local_path = os.path.join(os.path.dirname(bet.__file__), "/test")
local_path = ''


def unit_center_set(dim=1, num_samples=100,
                    delta=1, reg=False):
    r"""
    Make a unit hyper-rectangle sample set with positive probability
    inside an inscribed hyper-rectangle that has sidelengths delta,
    with its center at `np.array([[0.5]]*dim).
    (Useful for testing).

    :param int dim: dimension
    :param int num_samples: number of samples
    :param float delta: sidelength of region with positive probability
    :param bool reg: regular sampling (`num_samples` = per dimension)
    :rtype: :class:`bet.sample.sample_set`
    :returns: sample set object

    """
    s_set = sample.sample_set(dim)
    s_set.set_domain(np.array([[0, 1]]*dim))
    if reg:
        s = bsam.regular_sample_set(s_set, num_samples)
    else:
        s = bsam.random_sample_set('r', s_set, num_samples)
    dd = delta/2.0
    if dim > 1:
        probs = 1*(np.sum(np.logical_and(s._values <= (0.5+dd),
                                         s._values >= (0.5-dd)), axis=1)
                   >= dim)
    else:
        probs = 1*(np.logical_and(s._values <= (0.5+dd),
                                  s._values >= (0.5-dd)))
    s.set_probabilities(probs/np.sum(probs))  # uniform probabilities
    s.estimate_volume_mc()
    s.global_to_local()
    return s


def check_densities(s_set, dim=2, delta=0.1, tol=1e-4):
    # density values should be reciprocal of delta^dim
    true_den_val = 1.0/(delta**dim)
    if np.mean(np.abs(s_set._den - true_den_val)) < tol:
        return 1
    else:
        return 0


class Test_distance(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        self.int_set = sample.sample_set(dim=self.dim)
        self.num1, self.num2, self.num = 100, 100, 250
        self.left_set = unit_center_set(self.dim, self.num1, 0.5)
        self.right_set = unit_center_set(self.dim, self.num2, 0.5)
        self.domain = np.array([[0, 1]]*self.dim)
        values = np.random.rand(self.num, self.dim)
        self.int_set.set_values(values)
        self.int_set.set_domain(self.domain)

    def test_identity(self):
        r"""
        Ensure passing identical sets returns 0 distance.
        """
        for dist in ['tv', 'mink', 'norm', '2-norm', 'sqhell']:
            m = compP.metric(self.left_set, self.left_set)
            d = m.distance(dist)
            nptest.assert_equal(d, 0, 'Distance not definite.')
            m = compP.metric(self.left_set, self.left_set)
            d = m.distance(dist)
            nptest.assert_equal(d, 0, 'Distance not definite.')

    def test_aprox_symmetry(self):
        r"""
        Error up to approximation in emulation. We know the expected variance
        given a sample size to be 1/sqrt(N).
        """
        n = 1000
        m1 = compP.metric(self.left_set, self.right_set, n)
        d1 = m1.distance()
        m2 = compP.metric(self.right_set, self.left_set, n)
        d2 = m2.distance()
        nptest.assert_almost_equal(d1-d2, 0, 1, 'Distance not symmetric.')

    def test_exact_symmetry(self):
        r"""
        If two metrization objects are defined with swapped names of 
        left and right sample sets, the distance should still be identical
        """

        m1 = compP.metrization(self.int_set, self.left_set, self.right_set)
        m2 = compP.metrization(self.int_set, self.right_set, self.left_set)
        for dist in ['tv', 'mink', '2-norm', 'sqhell']:
            d1 = m1.distance(dist)
            d2 = m2.distance(dist)
            nptest.assert_almost_equal(
                d1-d2, 0, 12, 'Distance %s not symmetric.' % dist)
        import scipy.spatial.distance as ds

        # should be able to overwrite and still get correct answer.
        for dist in ['tv', 'hell', ds.cityblock]:
            m = compP.metric(self.left_set, self.right_set)
            d1 = m.distance(dist)
            m.set_right(self.left_set)
            m.set_left(self.right_set)
            d2 = m.distance(dist)
            nptest.assert_almost_equal(d1-d2, 0, 12, 'Distance not symmetric.')
            # grabbing copies like this should also work.
            ll = m.get_left().copy()
            m.set_left(m.get_right())
            m.set_right(ll)
            d2 = m.distance(dist)
            nptest.assert_almost_equal(d1-d2, 0, 12, 'Distance not symmetric.')


class Test_metrization_simple(unittest.TestCase):
    def setUp(self):
        self.dim = 3
        self.num1, self.num2, self.num = 100, 100, 500
        self.integration_set = sample.sample_set(dim=self.dim)
        self.left_set = unit_center_set(self.dim, self.num1, 0.5)
        self.right_set = unit_center_set(self.dim, self.num2, 0.5)
        values = np.ones((self.num, self.dim))
        self.integration_set.set_values(values)
        self.domain = np.tile([0, 1], [self.dim, 1])
        self.integration_set.set_domain(self.domain)
        self.left_set.set_domain(self.domain)
        self.right_set.set_domain(self.domain)
        self.mtrc = compP.metrization(sample_set_left=self.left_set,
                                      sample_set_right=self.right_set,
                                      emulated_sample_set=self.integration_set)

    def test_domain(self):
        r"""
        """
        self.mtrc.check_domain()

    def test_dim(self):
        r"""
        """
        self.mtrc.check_dim()

    def test_metric(self):
        r"""
        There are a few ways these functions can get initialized.
        Here we test the varying permutations
        """
        self.int_set = self.integration_set
        md = compP.metric(self.left_set, self.right_set)
        m10 = compP.metric(self.left_set, self.right_set, 10)
        mm = compP.metrization(self.int_set, self.left_set, self.right_set)
        mi = compP.metrization(self.int_set)

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
                              emulated_sample_set=self.integration_set)
        except sample.dim_not_matching:
            # setting wrong shapes should raise this error
            print('caught')
            pass

    def test_set_domain(self):
        r"""
        Check that improperly setting domain raises warning.
        """
        test_set = self.integration_set.copy()
        test_set.set_domain(test_set.get_domain()+0.01)
        test_metr = [compP.metrization(
            emulated_sample_set=self.integration_set),
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
        mm.get_left().set_reference_value(np.array([0.5]*self.dim))
        mm.get_right().set_reference_value(np.array([0.5]*self.dim))
        mm.get_left()._jacobians = np.ones((self.num1, self.dim, 1))
        mm.get_right()._jacobians = np.ones((self.num2, self.dim, 1))

        mm.slice([0])
        mc = mm.clip(50)
        ms = mm.merge(mc)
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
        self.mtrc.set_ptr_left(globalize=True)
        self.mtrc.get_ptr_left()
        self.mtrc.set_ptr_left(globalize=False)
        self.mtrc.get_ptr_left()
        self.mtrc.globalize_ptrs()

    def test_set_ptr_right(self):
        """
        Test setting right io ptr
        """
        # TODO be careful if we change Kdtree
        self.mtrc.set_ptr_right(globalize=True)
        self.mtrc.get_ptr_right()
        self.mtrc.set_ptr_right(globalize=False)
        self.mtrc.get_ptr_right()
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

    def test_estimate_density(self):
        r"""
        """
        self.mtrc.estimate_density()

    def test_get(self):
        r"""
        """
        mm = self.mtrc
        mm.get_int()
        mm.get_em()
        mm.get_integration_sample_set()
        mm.get_emulated()
        mm.get_emulated_sample_set()

    def test_estimate(self):
        r"""
        """
        mm = self.mtrc
        rd = mm.estimate_right_density()
        ld = mm.estimate_left_density()
        msg = "Get/set density mismatch."
        nptest.assert_array_equal(mm.get_density_left(), ld, msg)
        nptest.assert_array_equal(mm.get_density_right(), rd, msg)
        mm.estimate_density(emulated_sample_set=self.integration_set)
        mm.get_left().set_volumes(None)
        mm.get_right().set_volumes(None)
        mm.estimate_density()
        mm.get_left().set_volumes(None)
        mm.get_right().set_volumes(None)
        mm.estimate_density(emulated_sample_set=self.integration_set)
        try:  # the following should raise an error
            mm.set_int(None)
            mm.estimate_density()
        except AttributeError:
            pass
