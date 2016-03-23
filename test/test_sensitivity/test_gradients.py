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
import bet.sample as sample

class GradientsMethods:
    """
    Test all methods in :module:`bet.sensitivity.gradients`
    """
    # Test sampling methods
    def test_sample_linf_ball(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_linf_ball`.
        """
        self.input_set._values = grad.sample_linf_ball(self.input_set_centers, self.num_close, self.rvec)

        # Test the method returns the correct dimensions
        self.assertEqual(self.input_set._values.shape, ((self.num_close+1) * self.num_centers, self.input_dim))

        # Check the method returns centers followed by the clusters around the
        # first center.
        self.repeat = np.repeat(self.centers, self.num_close, axis=0)
        
        nptest.assert_array_less(np.linalg.norm(self.input_set._values[self.num_centers:] - self.repeat, np.inf, axis=1), np.max(self.rvec))

        # Check that the samples are in lam_domain
        for Ldim in range(self.input_set._dim):
            nptest.assert_array_less(self.input_set._domain[Ldim, 0],
                self.input_set._values[:, Ldim])
            nptest.assert_array_less(self.input_set._values[:, Ldim],
                self.input_set._domain[Ldim, 1])


    def test_sample_l1_ball(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_l1_ball`.
        """
        self.input_set._values = grad.sample_l1_ball(self.input_set_centers, self.num_close, self.rvec)

        # Test that the samples are within max(rvec) of center (l1 dist)
        self.repeat = np.repeat(self.centers, self.num_close, axis=0)
        nptest.assert_array_less(np.linalg.norm(self.input_set._values[self.num_centers:] - self.repeat, np.inf, axis=1), np.max(self.rvec))

        # Test the method returns the correct dimensions
        self.assertEqual(self.input_set._values.shape, ((self.num_close+1) * self.num_centers, self.input_dim))

    # Test FD methods
    def test_pick_ffd_points(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_linf_ball`.
        """
        self.input_set._values = grad.pick_ffd_points(self.input_set_centers, self.rvec)

        #self.samples = grad.pick_ffd_points(self.centers, self.rvec)

        if not isinstance(self.rvec, np.ndarray):
            self.rvec = np.ones(self.input_dim) * self.rvec

        # Check the distance to the corresponding center is equal to rvec
        self.centersrepeat = np.repeat(self.centers, self.input_set._dim, axis=0)
        nptest.assert_array_almost_equal(np.linalg.norm(self.centersrepeat - self.input_set._values[self.num_centers:], axis=1), np.tile(self.rvec, self.num_centers))

        # Test the method returns the correct dimensions
        self.assertEqual(self.input_set._values.shape, ((self.input_set._dim + 1) * \
            self.num_centers, self.input_set._dim))

    def test_pick_cfd_points(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_l1_ball`.
        """
        self.input_set._values = grad.pick_cfd_points(self.input_set_centers, self.rvec)

        if not isinstance(self.rvec, np.ndarray):
            self.rvec = np.ones(self.input_dim) * self.rvec

        # Check the distance to the corresponding center is equal to rvec
        self.centersrepeat = np.repeat(self.centers, 2*self.input_set._dim, axis=0)
        nptest.assert_array_almost_equal(np.linalg.norm(self.centersrepeat - self.input_set._values[self.num_centers:], axis=1), np.tile(self.rvec, self.num_centers * 2))

        # Test the method returns the correct dimension
        self.assertEqual(self.input_set._values.shape, ((2*self.input_dim + 1) * \
            self.num_centers, self.input_set._dim))

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
        self.output_set = sample.sample_set(self.output_dim)
        self.input_set._values = grad.sample_l1_ball(self.input_set_centers, self.num_close, self.rvec)
        self.output_set._values = self.input_set._values.dot(self.coeffs)
        self.input_set._jacobians = grad.calculate_gradients_rbf(self.input_set, self.output_set, self.input_set_centers)

        # Test the method returns the correct size tensor
        self.assertEqual(self.input_set._jacobians.shape, (self.num_centers, self.output_dim,
            self.input_dim))

        # Test that each vector is normalized or a zero vector
        normG = np.linalg.norm(self.input_set._jacobians, ord=1, axis=2)

        # If its a zero vectors, make it the unit vector in input_dim
        self.input_set._jacobians[normG==0] = 1.0/self.input_dim
        nptest.assert_array_almost_equal(np.linalg.norm(self.input_set._jacobians, ord=1, axis=2), np.ones((self.input_set._jacobians.shape[0], self.input_set._jacobians.shape[1])))

    def test_calculate_gradients_ffd(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_ffd`.
        """
        self.output_set = sample.sample_set(self.output_dim)
        self.input_set._values = grad.pick_ffd_points(self.input_set_centers, self.rvec)
        self.output_set._values = self.input_set._values.dot(self.coeffs)
        self.input_set._jacobians = grad.calculate_gradients_ffd(self.input_set, self.output_set)

        # Test the method returns the correct size tensor
        self.assertEqual(self.input_set._jacobians.shape, (self.num_centers, self.output_dim,
            self.input_dim))

        # Test that each vector is normalized
        normG = np.linalg.norm(self.input_set._jacobians, ord=1, axis=2)

        # If its a zero vectors, make it the unit vector in input_dim
        self.input_set._jacobians[normG==0] = 1.0/self.input_dim
        nptest.assert_array_almost_equal(np.linalg.norm(self.input_set._jacobians, ord=1, axis=2), np.ones((self.input_set._jacobians.shape[0], self.input_set._jacobians.shape[1])))

    def test_calculate_gradients_cfd(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_cfd`.
        """
        self.output_set = sample.sample_set(self.output_dim)
        self.input_set._values = grad.pick_cfd_points(self.input_set_centers, self.rvec)
        self.output_set._values = self.input_set._values.dot(self.coeffs)
        self.input_set._jacobians = grad.calculate_gradients_cfd(self.input_set, self.output_set)

        # Test the method returns the correct size tensor
        self.assertEqual(self.input_set._jacobians.shape, (self.num_centers, self.output_dim,
            self.input_dim))

        # Test that each vector is normalized
        normG = np.linalg.norm(self.input_set._jacobians, ord=1, axis=2)

        # If its a zero vectors, make it the unit vector in input_dim
        self.input_set._jacobians[normG==0] = 1.0/self.input_set._dim
        nptest.assert_array_almost_equal(np.linalg.norm(self.input_set._jacobians, ord=1, axis=2), np.ones((self.input_set._jacobians.shape[0], self.input_set._jacobians.shape[1])))

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
        self.input_set_rbf._jacobians = grad.calculate_gradients_rbf(self.input_set_rbf, self.output_set_rbf, normalize=False)

        nptest.assert_array_almost_equal(self.input_set_rbf._jacobians - self.G_exact, 0, decimal = 2)

    def test_calculate_gradients_ffd_accuracy(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_ffd`.
        """
        self.input_set_ffd._jacobians = grad.calculate_gradients_ffd(self.input_set_ffd, self.output_set_ffd, normalize=False)

        nptest.assert_array_almost_equal(self.input_set_ffd._jacobians - self.G_exact, 0, decimal = 2)

    def test_calculate_gradients_cfd_accuracy(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_cfd`.
        """
        self.input_set_cfd._jacobians = grad.calculate_gradients_cfd(self.input_set_cfd, self.output_set_cfd, normalize=False)

        nptest.assert_array_almost_equal(self.input_set_cfd._jacobians - self.G_exact, 0, decimal = 2)


# Test cases
class test_1to20_1centers_unitsquare(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the input domain (Lambda)
        self.input_dim = 1
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)

        self.lam_domain = np.zeros((self.input_set._dim, 2))
        self.lam_domain[:,0] = np.zeros(self.input_set._dim)
        self.lam_domain[:,1] = np.ones(self.input_set._dim)

        self.input_set._domain = self.lam_domain
        self.input_set_centers._domain = self.lam_domain

        # Choose random centers in input_domian to cluster points around
        self.num_centers = 1
        self.num_close = self.input_set._dim + 1
        self.rvec = 0.1
        np.random.seed(0)
        self.centers = np.random.uniform(self.lam_domain[:, 0], self.lam_domain[:, 1] - self.lam_domain[:, 0], [self.num_centers, self.input_set._dim])
        self.input_set_centers._values = self.centers

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.input_dim, self.num_close])
        self.dxi = np.random.random([self.input_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.output_dim = 20
        coeffs = np.random.random((self.input_dim, self.output_dim-self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

class test_2to20_1centers_unitsquare(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.input_dim = 2
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)

        self.lam_domain = np.zeros((self.input_dim, 2))
        self.lam_domain[:,0] = np.zeros(self.input_dim)
        self.lam_domain[:,1] = np.ones(self.input_dim)

        self.input_set._domain = self.lam_domain
        self.input_set_centers._domain = self.lam_domain

        # Choose random centers to cluster points around
        self.num_centers = 1
        np.random.seed(0)
        self.centers = np.random.uniform(self.lam_domain[:, 0], self.lam_domain[:, 1] - self.lam_domain[:, 0], [self.num_centers, self.input_set._dim])
        self.input_set_centers._values = self.centers
        self.num_close = self.input_dim + 1
        self.rvec = np.random.random(self.input_dim)

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.input_dim, self.num_close])
        self.dxi = np.random.random([self.input_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.output_dim = 20
        coeffs = np.random.random((self.input_dim,
            self.output_dim-self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

class test_4to20_100centers_randomhyperbox(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.input_dim = 4
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)

        self.lam_domain = np.zeros((self.input_dim, 2))
        np.random.seed(0)
        self.lam_domain[:,0] = np.random.random(self.input_dim)
        self.lam_domain[:,1] = np.random.random(self.input_dim) + 2

        self.input_set._domain = self.lam_domain
        self.input_set_centers._domain = self.lam_domain

        # Choose random centers to cluster points around
        self.num_centers = 100
        self.centers = np.random.uniform(self.lam_domain[:, 0], self.lam_domain[:, 1] - self.lam_domain[:, 0], [self.num_centers, self.input_set._dim])
        self.input_set_centers._values = self.centers
        self.num_close = self.input_set._dim + 1
        self.rvec = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.input_dim, self.num_close])
        self.dxi = np.random.random([self.input_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.output_dim = 20
        coeffs = np.random.random((self.input_dim,
            self.output_dim-self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

class test_9to20_100centers_randomhyperbox(GradientsMethods, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.input_dim = 9
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)

        self.lam_domain = np.zeros((self.input_dim, 2))
        np.random.seed(0)
        self.lam_domain[:,0] = np.random.random(self.input_dim)
        self.lam_domain[:,1] = np.random.random(self.input_dim) + 2

        self.input_set._domain = self.lam_domain
        self.input_set_centers._domain = self.lam_domain

        # Choose random centers to cluster points around
        self.num_centers = 100
        self.centers = np.random.uniform(self.lam_domain[:, 0], self.lam_domain[:, 1] - self.lam_domain[:, 0], [self.num_centers, self.input_set._dim])
        self.input_set_centers._values = self.centers
        self.num_close = self.input_dim + 1
        self.rvec = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.input_dim, self.num_close])
        self.dxi = np.random.random([self.input_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.output_dim = 20
        coeffs = np.random.random((self.input_dim,
            self.output_dim-self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

class test_15to37_143centers_negrandomhyperbox(GradientsMethods,
        unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.input_dim = 15
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)

        self.lam_domain = np.zeros((self.input_dim, 2))
        np.random.seed(0)
        self.lam_domain[:,0] = -1*np.random.random(self.input_dim) - 2
        self.lam_domain[:,1] = -1*np.random.random(self.input_dim)

        self.input_set._domain = self.lam_domain
        self.input_set_centers._domain = self.lam_domain

        # Choose random centers to cluster points around
        self.num_centers = 143
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.input_dim)) + \
            self.lam_domain[:,0]
        self.input_set_centers._values = self.centers
        self.num_close = self.input_dim + 1
        self.rvec = 0.1

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.input_dim, self.num_close])
        self.dxi = np.random.random([self.input_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.output_dim = 37
        coeffs = np.random.random((self.input_dim,
            self.output_dim-self.input_dim))
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

class test_9to30_100centers_randomhyperbox_zeroQoIs(GradientsMethods,
        unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.input_dim = 9
        self.input_set = sample.sample_set(self.input_dim)
        self.input_set_centers = sample.sample_set(self.input_dim)

        self.lam_domain = np.zeros((self.input_dim, 2))
        np.random.seed(0)
        self.lam_domain[:,0] = np.random.random(self.input_dim)
        self.lam_domain[:,1] = np.random.random(self.input_dim) + 2

        self.input_set._domain = self.lam_domain
        self.input_set_centers._domain = self.lam_domain

        # Choose random centers to cluster points around
        self.num_centers = 100
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.input_dim)) + \
            self.lam_domain[:,0]
        self.input_set_centers._values = self.centers
        self.num_close = self.input_dim + 1
        self.rvec = np.random.random(self.input_dim)

        # Choose array shapes for RBF methods
        np.random.seed(0)
        self.radii_rbf = np.random.random([self.num_close, self.num_close])
        self.radii_rbfdxi = np.random.random([self.input_dim, self.num_close])
        self.dxi = np.random.random([self.input_dim, self.num_close])

        # Define example linear functions (QoIs) for gradient approximation
        # methods
        self.output_dim = 30
        coeffs = np.zeros((self.input_dim, 2 * self.input_dim))
        coeffs = np.append(coeffs, np.random.random((self.input_dim,
            self.output_dim - 3 * self.input_dim)), axis=1)
        self.coeffs = np.append(coeffs, np.eye(self.input_dim), axis=1)

# Test cases for the gradient approximation accuracy
class test_2to2_100centers_unitbox(GradientsAccuracy, unittest.TestCase):
    
    def setUp(self):
        # Define the parameter space (Lambda)
        self.input_dim = 2
        self.input_set_rbf = sample.sample_set(self.input_dim)
        self.input_set_ffd = sample.sample_set(self.input_dim)
        self.input_set_cfd = sample.sample_set(self.input_dim)

        self.input_set_centers = sample.sample_set(self.input_dim)

        self.output_dim = 2
        self.output_set_rbf = sample.sample_set(self.output_dim)
        self.output_set_ffd = sample.sample_set(self.output_dim)
        self.output_set_cfd = sample.sample_set(self.output_dim)

        self.lam_domain = np.zeros((self.input_dim, 2))
        self.lam_domain[:,0] = np.zeros(self.input_dim)
        self.lam_domain[:,1] = np.ones(self.input_dim)

        self.input_set_rbf._domain = self.lam_domain
        self.input_set_ffd._domain = self.lam_domain
        self.input_set_cfd._domain = self.lam_domain

        # Choose random centers to cluster points around
        self.num_centers = 100
        np.random.seed(0)
        self.centers = (self.lam_domain[:,1] - self.lam_domain[:,0]) * \
            np.random.random((self.num_centers,self.input_dim)) + \
            self.lam_domain[:,0]
        self.input_set_centers._values = self.centers
        self.num_close = self.input_dim + 1
        self.rvec = 0.01 * np.ones(self.input_dim)

        self.input_set_rbf._values = grad.sample_l1_ball(self.input_set_centers, self.num_close, self.rvec)
        self.input_set_ffd._values = grad.pick_ffd_points(self.input_set_centers, self.rvec)
        self.input_set_cfd._values = grad.pick_cfd_points(self.input_set_centers, self.rvec)

        # Define a vector valued function f : [0,1]x[0,1] -> [x^2, y^2]
        def f(x):
            f = np.zeros(x.shape)
            f[:, 0] = x[:, 0]**2
            f[:, 1] = x[:, 1]**2
            return f

        self.output_set_rbf._values = f(self.input_set_rbf._values)
        self.output_set_ffd._values = f(self.input_set_ffd._values)
        self.output_set_cfd._values = f(self.input_set_cfd._values)

        self.G_exact = np.zeros([self.num_centers, self.output_dim,
            self.input_dim])
        self.G_exact[:, 0, 0] = 2 * self.centers[:, 0]
        self.G_exact[:, 1, 1] = 2 * self.centers[:, 1]
