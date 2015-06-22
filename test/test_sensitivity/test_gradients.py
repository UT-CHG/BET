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

#TODO Look at the accuacy tests at the end, whats going on?...

class GradientsMethods:
    """
    Test all methods in :module:`bet.sensitivity.gradients`
    """
    # Test sampling methods
    #TODO make this put sample in correct order!!!
    def test_sample_linf_ball(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_linf_ball`.
        """
        self.samples = grad.sample_linf_ball(self.centers,
            self.num_close, self.rvec, self.lam_domain)

        # Test the method returns the correct dimensions
        self.assertEqual(self.samples.shape, ((self.num_close+1) * \
            self.num_centers, self.Lambda_dim))

        # Check the method returns centers followed by the clusters around the
        # first center.
        #nptest.assert_array_less(np.linalg.norm(self.samples[self.num_centers:\
        #    self.num_centers + self.num_close, :] - self.centers[0,:], np.inf),
        #    self.rvec)

        # Check that the samples are in lam_domain
        for Ldim in range(self.Lambda_dim):
            nptest.assert_array_less(self.lam_domain[Ldim,0],
                self.samples[:,Ldim])
            nptest.assert_array_less(self.samples[:,Ldim],
                self.lam_domain[Ldim,1])

    def test_sample_l1_ball(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_l1_ball`.
        """
        self.samples = grad.sample_l1_ball(self.centers, self.num_close,
            self.rvec)

        # Test that the samples are within max(rvec) of center (l1 dist)
        self.repeat = np.repeat(self.centers, self.num_close, axis=0)
        nptest.assert_array_less(np.linalg.norm(self.samples[self.num_centers:]\
            - self.repeat, 1, axis=1), np.max(self.rvec))

        # Test the method returns the correct dimensions
        self.assertEqual(self.samples.shape, ((self.num_close+1) * \
            self.num_centers, self.Lambda_dim))

    # Test FD methods
    def test_pick_ffd_points(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_linf_ball`.
        """
        self.samples = grad.pick_ffd_points(self.centers, self.rvec)

        if not isinstance(self.rvec, np.ndarray):
            self.rvec = np.ones(self.Lambda_dim) * self.rvec

        # Check the distance to the corresponding center is equal to rvec
        self.centersrepeat = np.repeat(self.centers, self.Lambda_dim, axis=0)
        nptest.assert_array_almost_equal(np.linalg.norm(self.centersrepeat - \
            self.samples[self.num_centers:], axis=1), np.tile(self.rvec,
            self.num_centers))

        # Test the method returns the correct dimensions
        self.assertEqual(self.samples.shape, ((self.Lambda_dim+1) * \
            self.num_centers, self.Lambda_dim))

    def test_pick_cfd_points(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_l1_ball`.
        """
        self.samples = grad.pick_cfd_points(self.centers, self.rvec)

        if not isinstance(self.rvec, np.ndarray):
            self.rvec = np.ones(self.Lambda_dim) * self.rvec

        # Check the distance to the corresponding center is equal to rvec
        self.centersrepeat = np.repeat(self.centers, 2*self.Lambda_dim, axis=0)
        nptest.assert_array_almost_equal(np.linalg.norm(self.centersrepeat - \
            self.samples[self.num_centers:], axis=1), np.tile(self.rvec,
            self.num_centers * 2))

        # Test the method returns the correct dimension
        self.assertEqual(self.samples.shape, ((2*self.Lambda_dim + 1) * \
            self.num_centers, self.Lambda_dim))

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
        self.rbfdxi = grad.radial_basis_function_dxi(self.radii_rbfdxi,
            self.dxi, 'Multiquadric')

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
        self.samples = grad.sample_l1_ball(self.centers, self.num_close,
            self.rvec)
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_rbf(self.samples, self.data,
            self.centers)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_centers, self.num_qois,
            self.Lambda_dim))

        # Test that each vector is normalized or a zero vector
        normG = np.linalg.norm(self.G, axis=2)

        # If its a zero vectors, make it the unit vector in Lambda_dim
        self.G[normG==0] = 1.0/np.sqrt(self.Lambda_dim)
        nptest.assert_array_almost_equal(np.linalg.norm(self.G, axis=2),
            np.ones((self.G.shape[0], self.G.shape[1])))

    def test_calculate_gradients_ffd(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_ffd`.
        """
        self.samples = grad.pick_ffd_points(self.centers, self.rvec)
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_ffd(self.samples, self.data)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_centers, self.num_qois,
            self.Lambda_dim))

        # Test that each vector is normalized
        normG = np.linalg.norm(self.G, axis=2)

        # If its a zero vectors, make it the unit vector in Lambda_dim
        self.G[normG==0] = 1.0/np.sqrt(self.Lambda_dim)
        nptest.assert_array_almost_equal(np.linalg.norm(self.G, axis=2),
            np.ones((self.G.shape[0], self.G.shape[1])))

    def test_calculate_gradients_cfd(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_cfd`.
        """
        self.samples = grad.pick_cfd_points(self.centers, self.rvec)
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_cfd(self.samples, self.data)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_centers, self.num_qois,
            self.Lambda_dim))

        # Test that each vector is normalized
        normG = np.linalg.norm(self.G, axis=2)

        # If its a zero vectors, make it the unit vector in Lambda_dim
        self.G[normG==0] = 1.0/np.sqrt(self.Lambda_dim)
        nptest.assert_array_almost_equal(np.linalg.norm(self.G, axis=2),
            np.ones((self.G.shape[0], self.G.shape[1])))

# Test the accuracy of the gradient approximation methods
class GradientsAccuracy:
    """
    Test the accuracy of the gradient approximation method in
        :module:`bet.sensitivity.gradients`
    """
    def test_calculate_gradients_rbf_accuracy(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_rbf`.
        """
        self.G_nonlin = grad.calculate_gradients_rbf(self.samples_rbf,
            self.data_nonlin_rbf, normalize=False)

        nptest.assert_array_almost_equal(self.G_nonlin - self.G_exact, 0, decimal = 1)

    def test_calculate_gradients_ffd_accuracy(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_ffd`.
        """
        self.G_nonlin = grad.calculate_gradients_ffd(self.samples_ffd,
            self.data_nonlin_ffd, normalize=False)

        nptest.assert_array_almost_equal(self.G_nonlin - self.G_exact, 0, decimal = 1)

    def test_calculate_gradients_cfd_accuracy(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_cfd`.
        """
        self.G_nonlin = grad.calculate_gradients_cfd(self.samples_cfd,
            self.data_nonlin_cfd, normalize=False)

        nptest.assert_array_almost_equal(self.G_nonlin - self.G_exact, 0, decimal = 1)


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
            np.random.random((self.num_centers,self.Lambda_dim)) + \
            self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.rvec = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim,
            self.num_qois-self.Lambda_dim))
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
            np.random.random((self.num_centers,self.Lambda_dim)) + \
            self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.rvec = np.random.random(self.Lambda_dim)

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim,
            self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

class test_4to20_100centers_randomhyperbox(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 4
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        np.random.seed(0)
        self.lam_domain[:,0] = np.random.random(self.Lambda_dim)
        self.lam_domain[:,1] = np.random.random(self.Lambda_dim) + 2

        # Choose random centers to cluster points around
        self.num_centers = 100
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + \
            self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.rvec = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim,
            self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

class test_9to20_100centers_randomhyperbox(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 9
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        np.random.seed(0)
        self.lam_domain[:,0] = np.random.random(self.Lambda_dim)
        self.lam_domain[:,1] = np.random.random(self.Lambda_dim) + 2

        # Choose random centers to cluster points around
        self.num_centers = 100
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + \
            self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.rvec = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim,
            self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

class test_15to37_143centers_negrandomhyperbox(GradientsMethods,
        unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 15
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        np.random.seed(0)
        self.lam_domain[:,0] = -1*np.random.random(self.Lambda_dim) - 2
        self.lam_domain[:,1] = -1*np.random.random(self.Lambda_dim)

        # Choose random centers to cluster points around
        self.num_centers = 143
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + \
            self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.rvec = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.num_qois = 37
        coeffs = np.random.random((self.Lambda_dim,
            self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

class test_9to30_100centers_randomhyperbox_zeroQoIs(GradientsMethods,
        unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 9
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        np.random.seed(0)
        self.lam_domain[:,0] = np.random.random(self.Lambda_dim)
        self.lam_domain[:,1] = np.random.random(self.Lambda_dim) + 2

        # Choose random centers to cluster points around
        self.num_centers = 100
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + \
            self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.rvec = np.random.random(self.Lambda_dim)

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.Lambda_dim, self.num_close])
        self.dxi = np.random.random([self.Lambda_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.num_qois = 30
        coeffs = np.zeros((self.Lambda_dim, 2*self.Lambda_dim))
        coeffs = np.append(coeffs, np.random.random((self.Lambda_dim,
            self.num_qois-3*self.Lambda_dim)), axis=1)
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

# Test cases for the gradient approximation accuracy
class test_2to2_100centers_unitbox(GradientsAccuracy, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.Lambda_dim = 2
        self.num_qois = 2
        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        self.lam_domain[:,0] = np.zeros(self.Lambda_dim)
        self.lam_domain[:,1] = np.ones(self.Lambda_dim)

        # Choose random centers to cluster points around
        self.num_centers = 100
        np.random.seed(0)
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.Lambda_dim)) + \
            self.lam_domain[:,0]
        self.num_close = self.Lambda_dim + 1
        self.rvec = 0.01 * np.ones(self.Lambda_dim)

        self.samples_rbf = grad.sample_l1_ball(self.centers, self.num_close,
            self.rvec)
        self.samples_ffd = grad.pick_ffd_points(self.centers, self.rvec)
        self.samples_cfd = grad.pick_cfd_points(self.centers, self.rvec)

        # Define a vector valued function f : [0,1]x[0,1] -> [x^2, y^2]
        def f(x):
            f = np.zeros(x.shape)
            f[:, 0] = x[:, 0]**2
            f[:, 1] = x[:, 1]**2
            return f

        self.data_nonlin_rbf = f(self.samples_rbf)
        self.data_nonlin_ffd = f(self.samples_ffd)
        self.data_nonlin_cfd = f(self.samples_cfd)

        self.G_exact = np.zeros([self.num_centers, self.num_qois,
            self.Lambda_dim])
        self.G_exact[:, 0, 0] = 2 * self.centers[:, 0]
        self.G_exact[:, 1, 1] = 2 * self.centers[:, 1]


















