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
    """
    Test :module:`bet.sensitivity.chooseQoIs`.
    """
    def test_chooseOptQoIs(self):
        """
        Test :meth:`bet.sensitivity.chooseQoIs.chooseOptQoIs`.
        """
        self.indexstart = 0
        self.indexstop = self.num_qois - 1
        [self.min_condnum, self.qoiIndices] = cQoIs.chooseOptQoIs(self.G, self.indexstart, self.indexstop, self.num_qois_returned)

        # Test the method returns the correct number of qois
        self.assertEqual(len(self.qoiIndices), self.num_qois_returned)

        # Check that the 'global condidtion number' is greater than or equal to 1
        self.assertGreater(self.min_condnum, 1.0)

        # Test the method returns the known best set of QoIs  (chosen to be last Lambda_dim indices)
        nptest.assert_array_less(self.num_qois-self.Lambda_dim-1, self.qoiIndices)

        # Test that none of the chosen QoIs are the same
        self.assertEqual(len(np.unique(self.qoiIndices)), len(self.qoiIndices))
        

class test_2to20_choose2(ChooseQoIsMethods, unittest.TestCase):
        def setUp(self):
            self.Lambda_dim = 2
            self.num_qois_returned = 2
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
            self.num_qois_returned = 4
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
            self.num_qois_returned = 9
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
            self.num_qois_returned = 4
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
            self.num_qois_returned = 2
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


