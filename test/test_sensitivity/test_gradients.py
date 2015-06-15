# Copyright (C) 2014-2015 The BET Development Team

"""
This module contains tests for :module:`bet.sensitivity.gradients`.

Most of these tests should make sure certain values are within a tolerance
rather than exact due to machine precision.
"""
import unittest
import bet.sensitivity.gradients as grad
import numpy as np
import numpy.testing as nptest

class GradientsMethods:
    """
    Test all methods in :module:`bet.sensitivity.gradients`
    """
    # Test sampling methods
    def test_sample_linf_ball(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_linf_ball`.
        """
        self.samples = grad.sample_linf_ball(self.lam_domain, self.centers, self.num_close, self.radius)

        # Test the method returns the correct dimensions
        self.assertEqual(self.samples.shape, ((self.num_close+1)*self.num_centers, self.Lambda_dim))

        # Check that the samples are in lam_domain
        for Ldim in range(self.Lambda_dim):
            nptest.assert_array_less(self.lam_domain[Ldim,0], self.samples[:,Ldim])
            nptest.assert_array_less(self.samples[:,Ldim], self.lam_domain[Ldim,1])

    def test_sample_l1_ball(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_l1_ball`.
        """
        self.samples = grad.sample_l1_ball(self.centers, self.num_close, self.radius)

        # Test that the samples are within radius of center (l1 dist)
        self.tile = np.tile(self.centers, [1, self.num_close]).reshape(self.num_centers*self.num_close, self.Lambda_dim)
        nptest.assert_array_less(np.linalg.norm(self.samples[self.num_centers:]-self.tile, 1, axis=1), self.radius)

        # Test the method returns the correct dimensions
        self.assertEqual(self.samples.shape, ((self.num_close+1)*self.num_centers, self.Lambda_dim))

    # Test FD methods
    def test_pick_ffd_points(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_linf_ball`.
        """
        self.samples = grad.pick_ffd_points(self.centers, self.radius)

        # Check the distance to the corresponding center is equal to radius
        self.centerstile = np.tile(self.centers, [self.Lambda_dim, 1])
        nptest.assert_array_almost_equal(np.linalg.norm(self.centerstile-self.samples[self.num_centers:], axis=1), self.radius*np.ones(self.Lambda_dim*self.num_centers))

        # Test the method returns the correct number of samples
        self.assertEqual(self.samples.shape[0], (self.Lambda_dim+1)*self.num_centers)

    def test_pick_cfd_points(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_l1_ball`.
        """
        self.samples = grad.pick_cfd_points(self.centers, self.radius)

        # Check the distance to the corresponding center is equal to radius
        self.centerstile = np.tile(self.centers, [2*self.Lambda_dim, 1])
        nptest.assert_array_almost_equal(np.linalg.norm(self.centerstile-self.samples, axis=1), self.radius*np.ones(self.samples.shape[0]))

        # Test the method returns the correct dimension
        self.assertEqual(self.samples.shape, ((2*self.Lambda_dim)*self.num_centers, self.Lambda_dim))

    # Test RBF methods
    def test_radial_basis_function(self):
        """
        Test :meth:`bet.sensitivity.gradients.radial_basis_function`.
        """
        self.rbf = grad.radial_basis_function(self.radii_rbf, 'Multiquadric')

        # Test the method returns the correct number of function evaluations
        self.assertEqual(self.rbf.shape, (self.radii_rbf.shape))

        # Test the method returns an error when kernel is not available
        with self.assertRaises(ValueError):
            grad.radial_basis_function(self.radii_rbf, 'DNE')

    def test_radial_basis_function_dxi(self):
        """
        Test :meth:`bet.sensitivity.gradients.radial_basis_function_dxi`.
        """
        self.rbfdxi = grad.radial_basis_function_dxi(self.radii_rbfdxi, self.dxi, 'Multiquadric')

        # Test the method returns the correct number of function evaluations
        self.assertEqual(self.rbfdxi.shape, (self.radii_rbfdxi.shape))

        # Test the method returns an error when kernel is not available
        with self.assertRaises(ValueError):
            grad.radial_basis_function_dxi(self.radii_rbfdxi, self.dxi, 'DNE')

    # Test gradient approximation methods
    def test_calculate_gradients_rbf(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_rbf`.
        """
        self.samples = grad.sample_l1_ball(self.centers, self.num_close, self.radius)
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_rbf(self.samples, self.data, self.centers)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_centers, self.num_qois, self.Lambda_dim))

        # Test that each vector is normalized
        nptest.assert_array_almost_equal(np.linalg.norm(self.G, axis=2), np.ones((self.G.shape[0], self.G.shape[1])))
        


    def test_calculate_gradients_cfd(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_cfd`.
        """
        self.samples = grad.pick_cfd_points(self.centers, self.radius)
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_cfd(self.samples, self.data, self.centers, self.radius)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_centers, self.num_qois, self.Lambda_dim))

        # Test that each vector is normalized
        nptest.assert_array_almost_equal(np.linalg.norm(self.G, axis=2), np.ones((self.G.shape[0], self.G.shape[1])))

    def test_calculate_gradients_ffd(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_ffd`.
        """
        self.samples = grad.pick_ffd_points(self.centers, self.radius)
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_ffd(self.samples, self.data, self.centers, self.radius)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_centers, self.num_qois, self.Lambda_dim))

        # Test that each vector is normalized
        nptest.assert_array_almost_equal(np.linalg.norm(self.G, axis=2), np.ones((self.G.shape[0], self.G.shape[1])))

# Test cases
class test_1to20_1centers_unitsquare(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 1
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        self.lam_domain[:,0] = np.zeros(self.Lambda_dim)
        self.lam_domain[:,1] = np.ones(self.Lambda_dim)

        # Choose random centers to cluster points around
        self.num_centers = 1
        np.random.seed(0)
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.radius = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation methods
        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

class test_2to20_1centers_unitsquare(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 2
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        self.lam_domain[:,0] = np.zeros(self.Lambda_dim)
        self.lam_domain[:,1] = np.ones(self.Lambda_dim)

        # Choose random centers to cluster points around
        self.num_centers = 1
        np.random.seed(0)
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.radius = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation methods
        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

class test_4to20_100centers_randomhyperbox(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 4
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        self.lam_domain[:,0] = np.random.random(self.Lambda_dim)
        self.lam_domain[:,1] = np.random.random(self.Lambda_dim) + 2

        # Choose random centers to cluster points around
        self.num_centers = 100
        np.random.seed(0)
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.radius = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation methods
        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

class test_9to20_100centers_randomhyperbox(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 9
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        self.lam_domain[:,0] = np.random.random(self.Lambda_dim)
        self.lam_domain[:,1] = np.random.random(self.Lambda_dim) + 2

        # Choose random centers to cluster points around
        self.num_centers = 100
        np.random.seed(0)
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.radius = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation methods
        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

class test_15to37_143centers_negrandomhyperbox(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 15
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        self.lam_domain[:,0] = -1*np.random.random(self.Lambda_dim) - 2
        self.lam_domain[:,1] = -1*np.random.random(self.Lambda_dim)

        # Choose random centers to cluster points around
        self.num_centers = 143
        np.random.seed(0)
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.radius = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation methods
        self.num_qois = 37
        coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)





