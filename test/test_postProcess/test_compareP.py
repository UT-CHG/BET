# Copyright (C) 2014-2020 The BET Development Team

import numpy as np
import numpy.testing as nptest
import unittest
import bet.postProcess.compareP as compareP
import test.problem_setups as ps


class Test_voronoi(unittest.TestCase):
    def setUp(self):
        """
        Setup Voronoi sample sets.
        """
        self.set1 = ps.random_voronoi(level=2).get_input_sample_set()
        self.set2 = ps.random_voronoi(level=2).get_input_sample_set()
        self.compare_set = ps.random_voronoi(level=1)
        self.set2_init = False

    def test_identity(self):
        """
        Ensure passing identical sets returns 0 distance.
        """
        def metric(v1, v2):
            return np.max(np.abs(v1-v2))
        for dist in ['tv', 'norm', '2-norm', 'hell', metric]:
            m = compareP.compare(self.set1, self.set1)
            m.set_compare_set(self.compare_set)
            d = m.distance(functional=dist)
            nptest.assert_almost_equal(d, 0.0, err_msg="Distances should be zero")

    def test_identity_marginal(self):
        """
        Ensure passing identical sets returns 0 distance for marginals.
        """
        def metric(v1, v2):
            return np.max(np.abs(v1-v2))
        for dist in ['tv', 'norm', '2-norm', 'hell', metric]:
            m = compareP.compare(self.set1, self.set1)
            d = m.distance_marginal(i=0, functional=dist)
            nptest.assert_almost_equal(d, 0.0, err_msg="Distances should be zero")

    def test_identity_marginal_quad(self):
        """
        Ensure passing identical sets returns 0 distance for marginals using quadrature.
        """
        def metric(v1, v2):
            return np.max(np.abs(v1-v2))
        for dist in ['tv', 'norm', '2-norm', 'hell', metric]:
            m = compareP.compare(self.set1, self.set1)
            d = m.distance_marginal_quad(i=0, functional=dist)
            nptest.assert_almost_equal(d, 0.0, err_msg="Distances should be zero")

    def test_symmetry(self):
        """
        Ensure symmetry in distance metrics.
        :return:
        """
        m1 = compareP.compare(self.set1, self.set2, set2_init=self.set2_init)
        m1.set_compare_set(self.compare_set)
        m2 = compareP.compare(self.set2, self.set1, set1_init=self.set2_init)
        m2.set_compare_set(self.compare_set)

        for dist in ['tv', 'mink', '2-norm', 'sqhell']:
            d1 = m1.distance(functional=dist)
            d2 = m2.distance(functional=dist)
            nptest.assert_almost_equal(d1, d2, decimal=1, err_msg="Metric not symmetric")

    def test_symmetry_marginal(self):
        """
        Ensure symmetry in distance metrics for marginals.
        :return:
        """
        m1 = compareP.compare(self.set1, self.set2, set2_init=self.set2_init)
        m2 = compareP.compare(self.set2, self.set1, set1_init=self.set2_init)

        for dist in ['tv', 'mink', '2-norm', 'sqhell']:
            d1 = m1.distance_marginal(i=0, functional=dist)
            d2 = m2.distance_marginal(i=0, functional=dist)
            nptest.assert_almost_equal(d1, d2, err_msg="Metric not symmetric")

    def test_symmetry_marginal_quad(self):
        """
        Ensure symmetry in distance metrics for marginals using quadrature.
        :return:
        """
        m1 = compareP.compare(self.set1, self.set2, set2_init=self.set2_init)
        m2 = compareP.compare(self.set2, self.set1, set1_init=self.set2_init)

        for dist in ['tv']:
            d1 = m1.distance_marginal_quad(i=0, functional=dist, tol=1.0e-2)
            d2 = m2.distance_marginal_quad(i=0, functional=dist, tol=1.0e-2)
            nptest.assert_almost_equal(d1, d2, decimal=1, err_msg="Metric not symmetric")


class Test_kde(Test_voronoi):
    def setUp(self):
        """
        Setup kernel density estimate sample sets.
        """
        disc1, disc2 = ps.random_rv(dim=2, level=2)
        self.set1 = disc1.get_input_sample_set()
        self.set2 = disc1.get_input_sample_set()
        self.compare_set = 1000
        self.set2_init = True
