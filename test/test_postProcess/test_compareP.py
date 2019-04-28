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
        for dist in ['tv', 'norm', '2-norm', 'hell']:
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
        n = 100
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
        for dist in ['tv', ds.cityblock]:
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


class Test_density(unittest.TestCase):
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

    def test_missing_probs(self):
        r"""
        Check that correct errors get raised
        """
        mm = compP.metrization(
            self.int_set, self.left_set.copy(), self.right_set)
        try:
            mm.get_left().set_probabilities(None)
            mm.estimate_left_density()
        except AttributeError:
            pass
        mm.set_left(self.left_set)
        # if local probs go missing, we should still be fine
        mm.get_left()._probabilities_local = None
        mm.estimate_left_density()

    def test_missing_vols(self):
        r"""
        Check that correct errors get raised
        """
        mm = compP.metrization(self.int_set, self.left_set, self.right_set)
        try:
            mm.get_left().set_volumes(None)
            mm.estimate_left_density()
        except AttributeError:
            pass

    def test_missing(self):
        r"""
        Check that correct errors get raised if sample set is None.
        Check behavior of second argument not being provided.
        """
        try:
            compP.density(None)
        except AttributeError:
            pass
        ll = self.left_set
        dd = ll._probabilities.flatten()/ll._volumes.flatten()
        compP.density(ll, None)
        nptest.assert_array_equal(ll._density, dd)

    def test_existing_density(self):
        r"""
        Test intelligent evaluation of density (when to skip).
        """
        ll = self.left_set
        ll._density = ll._probabilities.flatten()/ll._volumes.flatten()
        compP.density(ll)
        compP.density(ll, [1, 2, 3])


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
        try:
            self.mtrc._sample_set_right._dim = 15
            self.mtrc.check_dim()
        except sample.dim_not_matching:
            self.mtrc._sample_set_right._dim = self.dim
            pass
        # force inconsistent sizes
        try:
            self.mtrc.set_ptr_right()
            self.mtrc.set_ptr_left()
            self.mtrc._ptr_left = self.mtrc._ptr_left[1:]
            self.mtrc.check_dim()
        except sample.dim_not_matching:
            pass

    def test_metric(self):
        r"""
        There are a few ways these functions can get initialized.
        Here we test the varying permutations
        """
        self.int_set = self.integration_set
        compP.metric(self.left_set, self.right_set)
        compP.metric(self.left_set, self.right_set, 10)
        compP.metrization(self.int_set, self.left_set, self.right_set)
        compP.metrization(self.int_set)

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
                              emulated_sample_set=integration_set)
        except sample.dim_not_matching:
            pass
        try:
            compP.metrization(sample_set_left=self.left_set,
                              sample_set_right=None,
                              emulated_sample_set=integration_set)
        except sample.dim_not_matching:
            pass
        try:
            compP.metrization(sample_set_left=self.left_set,
                              sample_set_right=None,
                              emulated_sample_set=integration_set)
        except sample.dim_not_matching:
            pass
        # if missing domain info, should be able to infer
        self.integration_set._domain = None
        compP.metrization(sample_set_left=None,
                          sample_set_right=self.right_set,
                          emulated_sample_set=self.integration_set)

        try:  # if not enough info, raise error
            self.integration_set._domain = None
            compP.metrization(sample_set_left=None,
                              sample_set_right=None,
                              emulated_sample_set=self.integration_set)
        except AttributeError:
            pass

    def test_set_domain(self):
        r"""
        Check that improperly setting domain raises warning.
        """
        test_set = self.integration_set.copy()
        test_set.set_domain(test_set.get_domain()+0.01)
        # all the ways to initialize the class
        test_metr = [compP.metrization(self.integration_set),
                     compP.metrization(self.integration_set,
                                       sample_set_right=self.right_set),
                     compP.metrization(self.integration_set,
                                       sample_set_left=self.left_set)
                     ]
        # setting one of the missing properties
        for mm in test_metr:
            test_funs = [mm.set_right,
                         mm.set_left]
            for fun in test_funs:
                try:
                    fun(test_set)
                except sample.domain_not_matching:
                    pass

        # overwriting integration sample set
        test_metr = [
            compP.metrization(
                None, sample_set_right=self.right_set),
            compP.metrization(
                None, sample_set_left=self.left_set),
            compP.metrization(self.integration_set,
                              self.left_set, self.right_set)
        ]

        # setting one of the missing properties
        for mm in test_metr:
            try:
                mm.set_int(test_set)
            except sample.domain_not_matching:
                pass

        try:  # should catch problems on initialization too
            mm = compP.metrization(self.integration_set,
                                   self.left_set, test_set)
        except sample.domain_not_matching:
            pass
        try:  # should catch problems on initialization too
            mm = compP.metrization(self.integration_set,
                                   test_set, self.right_set)
        except sample.domain_not_matching:
            pass

    def test_passed_ptrs(self):
        r"""
        Passing incorrect pointer shape raises errors
        """
        ptr = np.ones(self.num+1)
        try:
            compP.metrization(self.integration_set,
                              self.left_set, self.right_set, ptr, None)
        except AttributeError:
            pass
        try:
            compP.metrization(self.integration_set,
                              self.left_set, self.right_set, None, ptr)
        except AttributeError:
            pass
        try:
            compP.metrization(self.integration_set,
                              self.left_set, self.right_set,
                              ptr, np.ones(self.num))
        except AttributeError:
            pass
        try:
            compP.metrization(self.integration_set,
                              self.left_set, self.right_set,
                              np.ones(self.num), ptr)
        except AttributeError:
            pass

    def test_probabilities(self):
        r"""
        Setting/getting probabilities
        """
        self.mtrc.set_left_probabilities(np.ones(self.num1))
        self.mtrc.set_right_probabilities(np.ones(self.num2))
        try:
            self.mtrc.set_left_probabilities(np.ones(self.num1+1))
        except AttributeError:
            pass
        try:
            self.mtrc.set_right_probabilities(np.ones(self.num2+1))
        except AttributeError:
            pass
        ll = self.mtrc.get_left_probabilities()
        rr = self.mtrc.get_right_probabilities()
        assert len(ll) == self.num1
        assert len(rr) == self.num2

    def test_set_volume_mc(self):
        self.mtrc.estimate_volume_mc()

    def test_copy_clip_merge_slice(self):
        r"""
        Test copying, clipping, merging, slicing
        """
        mm = self.mtrc.copy()
        mm.get_left().set_reference_value(np.array([0.5]*self.dim))
        mm.get_right().set_reference_value(np.array([0.5]*self.dim))
        mm.get_left()._jacobians = np.ones((self.num1, self.dim, 1))
        mm.get_right()._jacobians = np.ones((self.num2, self.dim, 1))
        mm.estimate_density()
        mm.slice([0])
        mc = mm.clip(50)
        mc.estimate_density()  # make sure function still works!
        ms = mm.merge(mc)
        ms = ms.clip(0)  # this should just return an identical copy
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

    def test_set_emulation(self):
        r"""
        Different ways to set emulation set.
        """
        mm = compP.metrization(None, self.left_set, None)
        integration_set = self.integration_set.copy()
        mm.set_int(integration_set)
        nptest.assert_array_equal(mm.get_int()._values,
                                  self.integration_set._values)
        mm.set_integration_sample_set(integration_set)
        nptest.assert_array_equal(mm.get_int()._values,
                                  self.integration_set._values)
        mm.set_emulated_sample_set(integration_set)
        nptest.assert_array_equal(mm.get_int()._values,
                                  self.integration_set._values)
        mm.set_em(integration_set)
        nptest.assert_array_equal(mm.get_int()._values,
                                  self.integration_set._values)
        mm.set_emulated(integration_set)
        nptest.assert_array_equal(mm.get_int()._values,
                                  self.integration_set._values)

    def test_get(self):
        r"""
        Different ways to get emulated set.
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

    def test_discretization(self):
        r"""
        Support for passing discretization objects.
        """
        dl = sample.discretization(self.left_set, self.right_set)
        dr = sample.discretization(self.right_set, self.left_set)
        mm = compP.metric(dl, dr)
        nptest.assert_array_equal(self.mtrc.get_left()._values,
                                  mm.get_left()._values)
        nptest.assert_array_equal(self.mtrc.get_right()._values,
                                  mm.get_right()._values)
        mm.set_right(dr) # assuming input sample set
        mm.set_left(dl)
        nptest.assert_array_equal(self.mtrc.get_left()._values,
                                  mm.get_left()._values)
        nptest.assert_array_equal(self.mtrc.get_right()._values,
                                  mm.get_right()._values)