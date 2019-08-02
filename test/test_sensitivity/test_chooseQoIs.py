# Copyright (C) 2014-2019 The BET Development Team

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
import bet.sample as sample


class ChooseQoIsMethods:
    """
    Test :module:`bet.sensitivity.chooseQoIs`.
    """

    def test_calculate_avg_measure(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.calculate_avg_measure`.
        """
        self.qoi_set = np.arange(0, self.input_dim)
        (self.measure, self.singvals) = cQoIs.calculate_avg_measure(
            self.input_set_centers, self.qoi_set)

        # Check that measure and singvals are the right size
        self.assertEqual(isinstance(self.measure, float), True)
        self.assertEqual(self.singvals.shape, (self.num_centers,
                                               self.input_dim))

        # Test the method returns an error when more qois are given than
        # parameters
        self.input_set_centers.set_jacobians(
            np.random.uniform(-1, 1, [10, 4, 3]))
        with self.assertRaises(ValueError):
            cQoIs.calculate_avg_measure(self.input_set_centers)

    def test_calculate_avg_skewness(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.calculate_avg_skewness`.
        """
        self.qoi_set = np.arange(0, self.input_dim)
        (self.skewness, self.skewnessgi) = cQoIs.calculate_avg_skewness(
            self.input_set_centers, self.qoi_set)

        # Check that skewness and skewnessgi are the right size
        self.assertEqual(isinstance(self.skewness, float), True)
        self.assertEqual(self.skewnessgi.shape, (self.num_centers,
                                                 self.input_dim))

        # Test the method returns an error when more qois are given than
        # parameters
        self.input_set_centers.set_jacobians(
            np.random.uniform(-1, 1, [10, 4, 3]))
        with self.assertRaises(ValueError):
            cQoIs.calculate_avg_measure(self.input_set_centers)

    def test_calculate_avg_condnum(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.calculate_avg_condnum`.
        """
        self.qoi_set = np.arange(0, self.input_dim)
        print(self.input_set_centers.get_jacobians())
        print(self.center_disc._input_sample_set.get_jacobians())
        (self.condnum, self.singvals) = cQoIs.calculate_avg_condnum(
            self.input_set_centers, self.qoi_set)

        # Check that condnum and singvals are the right size
        self.assertEqual(isinstance(self.condnum, float), True)
        self.assertEqual(self.singvals.shape, (self.num_centers,
                                               self.input_dim))

        # Test the method returns an error when more qois are given than
        # parameters
        self.input_set_centers.set_jacobians(
            np.random.uniform(-1, 1, [10, 4, 3]))
        with self.assertRaises(ValueError):
            cQoIs.calculate_avg_measure(self.input_set_centers)

    def test_chooseOptQoIs(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs`.
        """
        self.qoiIndices = np.arange(0, self.output_dim)
        self.condnum_indices_mat = cQoIs.chooseOptQoIs(self.input_set_centers,
                                                       self.qoiIndices, self.output_dim_return, self.num_optsets_return)
        self.condnum_indices_mat_vol = cQoIs.chooseOptQoIs(self.input_set_centers,
                                                           self.qoiIndices, self.output_dim_return, self.num_optsets_return,
                                                           measure=True)

        # Test the method returns the correct size array
        self.assertEqual(self.condnum_indices_mat.shape,
                         (self.num_optsets_return, self.output_dim_return + 1))

        self.assertEqual(self.condnum_indices_mat_vol.shape,
                         (self.num_optsets_return, self.output_dim_return + 1))

        # Check that the 'global condition number' is greater than or equal to
        # 1
        nptest.assert_array_less(1.0, self.condnum_indices_mat[:, 0])

        # For measure, check that it is greater than or equal to 0
        nptest.assert_array_less(0.0, self.condnum_indices_mat_vol[:, 0])

        # Test the method returns the known best set of QoIs  (chosen to be
        # last input_dim indices)
        nptest.assert_array_less(self.output_dim - self.input_dim - 1,
                                 self.condnum_indices_mat[0, 1:])

        nptest.assert_array_less(self.output_dim - self.input_dim - 1,
                                 self.condnum_indices_mat_vol[0, 1:])

        # Test that none of the best chosen QoIs are the same
        self.assertEqual(len(np.unique(self.condnum_indices_mat[0, 1:])),
                         len(self.condnum_indices_mat[0, 1:]))

        self.assertEqual(len(np.unique(self.condnum_indices_mat[0, 1:])),
                         len(self.condnum_indices_mat_vol[0, 1:]))

        ##########
        # Test the method for a set of QoIs rather than all possible.  Choose
        # this set so that the optimal choice is not removed.
        self.qoiIndices = np.concatenate([np.arange(1, 3, 2),
                                          np.arange(4, self.output_dim)])
        self.condnum_indices_mat = cQoIs.chooseOptQoIs(self.input_set_centers,
                                                       self.qoiIndices, self.output_dim_return, self.num_optsets_return)

        self.condnum_indices_mat_vol = cQoIs.chooseOptQoIs(self.input_set_centers,
                                                           self.qoiIndices, self.output_dim_return, self.num_optsets_return,
                                                           measure=True)

        # Test the method returns the correct number of qois
        self.assertEqual(self.condnum_indices_mat.shape,
                         (self.num_optsets_return, self.output_dim_return + 1))

        self.assertEqual(self.condnum_indices_mat_vol.shape,
                         (self.num_optsets_return, self.output_dim_return + 1))

        # Check that the 'global condidtion number' is greater than or equal
        # to 1
        nptest.assert_array_less(1.0, self.condnum_indices_mat[:, 0])

        nptest.assert_array_less(0.0, self.condnum_indices_mat_vol[:, 0])

        # Test the method returns the known best set of QoIs  (chosen to be
        # last input_dim indices)
        nptest.assert_array_less(self.output_dim - self.input_dim - 1,
                                 self.condnum_indices_mat[0, 1:])

        nptest.assert_array_less(self.output_dim - self.input_dim - 1,
                                 self.condnum_indices_mat_vol[0, 1:])

        # Test that none of the best chosen QoIs are the same
        self.assertEqual(len(np.unique(self.condnum_indices_mat[0, 1:])),
                         len(self.condnum_indices_mat[0, 1:]))

        self.assertEqual(len(np.unique(self.condnum_indices_mat[0, 1:])),
                         len(self.condnum_indices_mat_vol[0, 1:]))

    def test_chooseOptQoIs_verbose(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs_verbose`.
        """
        self.qoiIndices = np.arange(0, self.output_dim)
        [self.condnum_indices_mat, self.optsingvals] = \
            cQoIs.chooseOptQoIs_verbose(self.input_set_centers, self.qoiIndices,
                                        self.output_dim_return, self.num_optsets_return)

        # Test that optsingvals is the right shape
        self.assertEqual(self.optsingvals.shape, ((self.num_centers,
                                                   self.output_dim_return, self.num_optsets_return)))

    def test_find_unique_vecs(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.find_unique_vecs`.
        """
        self.qoiIndices = np.arange(0, self.output_dim)
        unique_indices = cQoIs.find_unique_vecs(self.input_set_centers,
                                                self.inner_prod_tol, self.qoiIndices)

        # Test that pairwise inner products are <= inner_prod_tol
        pairs = np.array(list(combinations(list(unique_indices), 2)))
        for pair in range(pairs.shape[0]):
            curr_set = pairs[pair]
            curr_inner_prod = np.sum(self.input_set_centers._jacobians[:,
                                                                       curr_set[0], :] * self.input_set_centers._jacobians[:,
                                                                                                                           curr_set[1], :]) / self.input_set_centers._jacobians.shape[0]
            nptest.assert_array_less(curr_inner_prod, self.inner_prod_tol)

    def test_chooseOptQoIs_large(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs_large`.
        """
        self.qoiIndices = np.arange(0, self.output_dim)
        best_sets = cQoIs.chooseOptQoIs_large(self.input_set_centers,
                                              qoiIndices=self.qoiIndices, inner_prod_tol=self.inner_prod_tol,
                                              measskew_tol=self.measskew_tol)

        if self.measskew_tol == np.inf:
            self.measskew_tol = sys.float_info[0]
        # Test that the best_sets have condition number less than the tolerance
        for Ldim in range(self.input_dim - 1):
            inds = best_sets[Ldim][:, 0] != np.inf
            nptest.assert_array_less(best_sets[Ldim][inds, 0],
                                     self.measskew_tol)

    def test_chooseOptQoIs_large_verbose(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs_large_verbose`.
        """
        self.qoiIndices = np.arange(0, self.output_dim)
        [best_sets, optsingvals_list] = cQoIs.chooseOptQoIs_large_verbose(
            self.input_set_centers, qoiIndices=self.qoiIndices,
            num_optsets_return=self.num_optsets_return,
            inner_prod_tol=self.inner_prod_tol, measskew_tol=self.measskew_tol)

        # Test that input_dim - 1 optsingval tensors are returned
        self.assertEqual(len(optsingvals_list), self.input_dim - 1)

        # Test that each tensor is the right shape
        for i in range(self.input_dim - 1):
            self.assertEqual(optsingvals_list[i].shape, (self.num_centers,
                                                         i + 2, self.num_optsets_return))


class test_2to20_choose2(ChooseQoIsMethods, unittest.TestCase):
    def setUp(self):
        self.input_dim = 2
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)
        self.output_dim_return = 2
        self.num_optsets_return = 5
        self.radius = 0.01
        np.random.seed(0)
        self.num_centers = 10
        self.centers = np.random.random((self.num_centers, self.input_dim))
        self.input_set_centers.set_values(self.centers)
        self.input_set = grad.sample_l1_ball(self.input_set_centers,
                                             self.input_dim + 1, self.radius)

        self.output_dim = 20
        self.output_set = sample.sample_set(self.output_dim)
        coeffs = np.random.random((self.input_dim,
                                   self.output_dim - self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

        self.output_set.set_values(self.input_set._values.dot(self.coeffs))
        self.my_disc = sample.discretization(self.input_set,
                                             self.output_set)
        self.center_disc = grad.calculate_gradients_rbf(
            self.my_disc, self.num_centers)
        self.input_set_centers = self.center_disc.get_input_sample_set()
        self.inner_prod_tol = 1.0
        self.measskew_tol = 100.0


class test_4to20_choose4(ChooseQoIsMethods, unittest.TestCase):
    def setUp(self):
        self.input_dim = 4
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)
        self.output_dim_return = 4
        self.num_optsets_return = 5
        self.radius = 0.01
        np.random.seed(0)
        self.num_centers = 100
        self.centers = np.random.random((self.num_centers, self.input_dim))
        self.input_set_centers.set_values(self.centers)
        self.input_set = grad.sample_l1_ball(self.input_set_centers,
                                             self.input_dim + 1, self.radius)

        self.output_dim = 20
        self.output_set = sample.sample_set(self.output_dim)
        coeffs = np.random.random((self.input_dim,
                                   self.output_dim - self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

        self.output_set.set_values(self.input_set._values.dot(self.coeffs))
        self.my_disc = sample.discretization(self.input_set,
                                             self.output_set)
        self.center_disc = grad.calculate_gradients_rbf(
            self.my_disc, self.num_centers)
        self.input_set_centers = self.center_disc.get_input_sample_set()

        self.inner_prod_tol = 0.9
        self.measskew_tol = 20.0


class test_9to15_choose9(ChooseQoIsMethods, unittest.TestCase):
    def setUp(self):
        self.input_dim = 9
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)
        self.output_dim_return = 9
        self.num_optsets_return = 50
        self.radius = 0.01
        np.random.seed(0)
        self.num_centers = 15
        self.centers = np.random.random((self.num_centers, self.input_dim))
        self.input_set_centers.set_values(self.centers)
        self.input_set = grad.sample_l1_ball(self.input_set_centers,
                                             self.input_dim + 1, self.radius)

        self.output_dim = 15
        self.output_set = sample.sample_set(self.output_dim)
        coeffs = np.random.random((self.input_dim,
                                   self.output_dim - self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

        self.output_set.set_values(self.input_set._values.dot(self.coeffs))
        self.my_disc = sample.discretization(self.input_set,
                                             self.output_set)
        self.center_disc = grad.calculate_gradients_rbf(
            self.my_disc, self.num_centers)
        self.input_set_centers = self.center_disc.get_input_sample_set()

        self.inner_prod_tol = 0.8
        self.measskew_tol = 100.0


class test_9to15_choose4(ChooseQoIsMethods, unittest.TestCase):
    def setUp(self):
        self.input_dim = 9
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)
        self.output_dim_return = 4
        self.num_optsets_return = 1
        self.radius = 0.01
        np.random.seed(0)
        self.num_centers = 11
        self.centers = np.random.random((self.num_centers, self.input_dim))
        self.input_set_centers.set_values(self.centers)
        self.input_set = grad.sample_l1_ball(self.input_set_centers,
                                             self.input_dim + 1, self.radius)

        self.output_dim = 15
        self.output_set = sample.sample_set(self.output_dim)
        coeffs = np.random.random((self.input_dim, self.output_dim -
                                   self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

        self.output_set.set_values(self.input_set._values.dot(self.coeffs))
        self.my_disc = sample.discretization(self.input_set,
                                             self.output_set)
        self.center_disc = grad.calculate_gradients_rbf(
            self.my_disc, self.num_centers)
        self.input_set_centers = self.center_disc.get_input_sample_set()

        self.inner_prod_tol = 0.9
        self.measskew_tol = 50.0


class test_2to28_choose2_zeros(ChooseQoIsMethods, unittest.TestCase):
    def setUp(self):
        self.input_dim = 2
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)
        self.output_dim_return = 2
        self.num_optsets_return = 5
        self.radius = 0.01
        np.random.seed(0)
        self.num_centers = 10
        self.centers = np.random.random((self.num_centers, self.input_dim))
        self.input_set_centers.set_values(self.centers)
        self.input_set = grad.sample_l1_ball(self.input_set_centers,
                                             self.input_dim + 1, self.radius)

        self.output_dim = 28
        self.output_set = sample.sample_set(self.output_dim)
        coeffs = np.ones((self.input_dim, 2 * self.input_dim))
        coeffs = np.append(coeffs, np.random.random((self.input_dim,
                                                     self.output_dim - 3 * self.input_dim)), axis=1)
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

        self.output_set.set_values(self.input_set._values.dot(self.coeffs))
        self.my_disc = sample.discretization(self.input_set,
                                             self.output_set)
        self.center_disc = grad.calculate_gradients_rbf(
            self.my_disc, self.num_centers)
        self.input_set_centers = self.center_disc.get_input_sample_set()

        self.inner_prod_tol = 0.9
        self.measskew_tol = np.inf
