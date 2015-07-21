# Copyright (C) 2014-2015 The BET Development Team

"""
This module contains tests for :module:`bet.sensitivity.chooseQoIs`.

Most of these tests should make sure certain values are within a tolerance
rather than exact due to machine precision.
"""
import unittest
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoIs
import numpy as np
import numpy.testing as nptest
from itertools import combinations
import sys

class ChooseQoIsMethods:
    """
    Test :module:`bet.sensitivity.chooseQoIs`.
    """
    def test_chooseOptQoIs(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs`.
        """
        self.qoiIndices = range(0, self.num_qois)
        self.condnum_indices_mat = cQoIs.chooseOptQoIs(self.G, self.qoiIndices,
            self.num_qois_return, self.num_optsets_return)

        # Test the method returns the correct size array
        self.assertEqual(self.condnum_indices_mat.shape,
            (self.num_optsets_return, self.num_qois_return + 1))

        # Check that the 'global condidtion number' is greater than or equal to 1
        nptest.assert_array_less(1.0, self.condnum_indices_mat[:, 0])

        # Test the method returns the known best set of QoIs  (chosen to be
        # last Lambda_dim indices)
        nptest.assert_array_less(self.num_qois-self.Lambda_dim-1,
            self.condnum_indices_mat[0, 1:])

        # Test that none of the best chosen QoIs are the same
        self.assertEqual(len(np.unique(self.condnum_indices_mat[0, 1:])),
            len(self.condnum_indices_mat[0, 1:]))

        # Test the method for a set of QoIs rather than all possible.  Choose
        # this set so that the optimal choice is not removed.
        self.qoiIndices = np.concatenate([range(1, 3, 2),
            range(4, self.num_qois)])
        self.condnum_indices_mat = cQoIs.chooseOptQoIs(self.G, self.qoiIndices,
            self.num_qois_return, self.num_optsets_return)

        # Test the method returns the correct number of qois
        self.assertEqual(self.condnum_indices_mat.shape,
            (self.num_optsets_return, self.num_qois_return + 1))

        # Check that the 'global condidtion number' is greater than or equal
        # to 1
        nptest.assert_array_less(1.0, self.condnum_indices_mat[:, 0])

        # Test the method returns the known best set of QoIs  (chosen to be
        # last Lambda_dim indices)
        nptest.assert_array_less(self.num_qois-self.Lambda_dim-1,
            self.condnum_indices_mat[0, 1:])

        # Test that none of the best chosen QoIs are the same
        self.assertEqual(len(np.unique(self.condnum_indices_mat[0, 1:])),
            len(self.condnum_indices_mat[0, 1:]))

    def test_chooseOptQoIs_verbose(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs_verbose`.
        """
        self.qoiIndices = range(0, self.num_qois)
        [self.condnum_indices_mat, self.optsingvals] = \
            cQoIs.chooseOptQoIs_verbose(self.G, self.qoiIndices,
            self.num_qois_return, self.num_optsets_return)

        # Test that optsingvals is the right shape
        self.assertEqual(self.optsingvals.shape, ((self.num_centers,
            self.num_qois_return, self.num_optsets_return)))

    def test_find_unique_vecs(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.find_unique_vecs`.
        """
        self.qoiIndices = range(0, self.num_qois)
        unique_indices = cQoIs.find_unique_vecs(self.G, self.inner_prod_tol, 
            self.qoiIndices)

        # Test that pairwise inner products are <= inner_prod_tol
        pairs = np.array(list(combinations(list(unique_indices), 2)))
        for pair in range(pairs.shape[0]):
            curr_set = pairs[pair]
            curr_inner_prod = np.sum(self.G[:, curr_set[0], :] * self.G[:,
                curr_set[1], :]) / self.G.shape[0]
            nptest.assert_array_less(curr_inner_prod, self.inner_prod_tol)

    def test_chooseOptQoIs_large(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs_large`.
        """
        self.qoiIndices = range(0, self.num_qois)
        best_sets = cQoIs.chooseOptQoIs_large(self.G, qoiIndices=self.qoiIndices,
            inner_prod_tol=self.inner_prod_tol, cond_tol=self.cond_tol)

        for Ldim in range(self.Lambda_dim - 1):
            nptest.assert_array_less(best_sets[Ldim][:, 0], self.cond_tol)

    def test_chooseOptQoIs_large(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs_large_verbose`.
        """
        self.qoiIndices = range(0, self.num_qois)
        [best_sets, optsingvals_list] = cQoIs.chooseOptQoIs_large_verbose(self.G,
            qoiIndices=self.qoiIndices, num_optsets_return=self.num_optsets_return,
            inner_prod_tol=self.inner_prod_tol, cond_tol=self.cond_tol)
        print optsingvals_list[0].shape

        self.assertEqual(len(optsingvals_list), self.Lambda_dim - 1)
        for i in range(self.Lambda_dim - 1):
            self.assertEqual(optsingvals_list[i].shape, (self.num_centers, i + 2, self.num_optsets_return))


class test_2to20_choose2(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 2
            self.num_qois_return = 2
            self.num_optsets_return = 5
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 10
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers,
                self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 20
            coeffs = np.random.random((self.Lambda_dim,
                self.num_qois-self.Lambda_dim))
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data,
                self.centers)

            self.inner_prod_tol = 1.0
            self.cond_tol = 100.0

class test_4to20_choose4(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 4
            self.num_qois_return = 4
            self.num_optsets_return = 5
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 100
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers,
                self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 20
            coeffs = np.random.random((self.Lambda_dim,
                self.num_qois-self.Lambda_dim))
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data,
                self.centers)

            self.inner_prod_tol = 0.9
            self.cond_tol = 20.0

class test_9to15_choose9(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 9
            self.num_qois_return = 9
            self.num_optsets_return = 50
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 15
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers, self.Lambda_dim + \
                1, self.radius)
            
            self.num_qois = 15
            coeffs = np.random.random((self.Lambda_dim,
                self.num_qois - self.Lambda_dim))
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data,
                self.centers)

            self.inner_prod_tol = 0.8
            self.cond_tol = 100.0

class test_9to15_choose4(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 9
            self.num_qois_return = 4
            self.num_optsets_return = 1
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 11
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers,
                self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 15
            coeffs = np.random.random((self.Lambda_dim, self.num_qois - \
                self.Lambda_dim))
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data,
                self.centers)

            self.inner_prod_tol = 0.9
            self.cond_tol = 50.0

class test_2to28_choose2_zeros(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 2
            self.num_qois_return = 2
            self.num_optsets_return = 5
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 10
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers,
                self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 28
            coeffs = np.zeros((self.Lambda_dim, 2*self.Lambda_dim))
            coeffs = np.append(coeffs, np.random.random((self.Lambda_dim,
                self.num_qois - 3 * self.Lambda_dim)), axis=1)
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data,
                self.centers)

            self.inner_prod_tol = 0.9
            self.cond_tol = sys.float_info[0]
