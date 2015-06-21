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

class ChooseQoIsMethods:
    #TODO: Test parallel implementation
    """
    Test :module:`bet.sensitivity.chooseQoIs`.
    """
    def test_chooseOptQoIs(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs`.
        """
        self.qoiIndices = range(0, self.num_qois)
        self.condnum_indices_mat = cQoIs.chooseOptQoIs(self.G, self.qoiIndices, self.num_qois_return)

        # Test the method returns the correct number of qois
        self.assertEqual(self.condnum_indices_mat.shape[1], self.num_qois_return + 1)

        # Check that the 'global condidtion number' is greater than or equal to 1
        nptest.assert_array_less(1.0, self.condnum_indices_mat[:, 0])

        # Test the method returns the known best set of QoIs  (chosen to be last Lambda_dim indices)
        nptest.assert_array_less(self.num_qois-self.Lambda_dim-1, self.condnum_indices_mat[0, 1:])

        # Test that none of the chosen QoIs are the same
        self.assertEqual(len(np.unique(self.condnum_indices_mat[0, 1:])), len(self.condnum_indices_mat[0, 1:]))

        # Test the method for a set of QoIs rather than all possible.  Choose
        # this set so that the optimal choice is not removed.
        self.qoiIndices = np.concatenate([range(1, 3, 2), range(4, self.num_qois)])
        self.condnum_indices_mat = cQoIs.chooseOptQoIs(self.G, self.qoiIndices, self.num_qois_return)

        # Test the method returns the correct number of qois
        self.assertEqual(self.condnum_indices_mat.shape[1], self.num_qois_return + 1)

        # Check that the 'global condidtion number' is greater than or equal to 1
        nptest.assert_array_less(1.0, self.condnum_indices_mat[:, 0])

        # Test the method returns the known best set of QoIs  (chosen to be last Lambda_dim indices)
        nptest.assert_array_less(self.num_qois-self.Lambda_dim-1, self.condnum_indices_mat[0, 1:])

        # Test that none of the chosen QoIs are the same
        self.assertEqual(len(np.unique(self.condnum_indices_mat[0, 1:])), len(self.condnum_indices_mat[0, 1:]))

    def test_chooseOptQoIs_verbose(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs_verbose`.
        """
        self.qoiIndices = range(0, self.num_qois)
        [self.condnum_indices_mat, self.optsingvals] = cQoIs.chooseOptQoIs_verbose(self.G, self.qoiIndices, self.num_qois_return)

        # Test that optsingvals is the right shape
        self.assertEqual(self.optsingvals.shape, ((self.num_centers, self.num_qois_return)))


class test_2to20_choose2(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 2
            self.num_qois_return = 2
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 10
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers, self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 20
            coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data, self.centers)

class test_4to20_choose4(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 4
            self.num_qois_return = 4
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 10
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers, self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 20
            coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data, self.centers)

class test_9to15_choose9(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 9
            self.num_qois_return = 9
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 10
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers, self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 15
            coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data, self.centers)

class test_9to15_choose4(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 9
            self.num_qois_return = 4
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 10
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers, self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 15
            coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data, self.centers)

class test_2to28_choose2_zeros(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 2
            self.num_qois_return = 2
            self.radius = 0.01
            np.random.seed(0)
            self.num_centers = 10
            self.centers = np.random.random((self.num_centers, self.Lambda_dim))
            self.samples = grad.sample_l1_ball(self.centers, self.Lambda_dim + 1, self.radius)
            
            self.num_qois = 28
            coeffs = np.zeros((self.Lambda_dim, 2*self.Lambda_dim))
            coeffs = np.append(coeffs, np.random.random((self.Lambda_dim, self.num_qois-3*self.Lambda_dim)), axis=1)
            self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

            self.data = self.samples.dot(self.coeffs)
            self.G = grad.calculate_gradients_rbf(self.samples, self.data, self.centers)


