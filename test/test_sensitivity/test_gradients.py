

import unittest
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoIs
import numpy as np
import numpy.testing as nptest

class TestSamplingMethods(unittest.TestCase):
    """
    Test all sampling methods in bet.sensitivity.gradients
    """
    def setUp(self):
        self.Lambda_dim = 2
        lam_left = np.zeros(self.Lambda_dim)
        lam_right = np.ones(self.Lambda_dim)

        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        self.lam_domain[:,0] = lam_left
        self.lam_domain[:,1] = lam_right

        self.num_centers = 10
        self.centers = np.random.random((self.num_centers,self.Lambda_dim))
        
        np.random.seed(0)
        num_qois = 20
        self.num_close = 100
        self.radius = 0.1

    def test_sample_linf_ball(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_linf_ball`.
        """
        self.samples = grad.sample_linf_ball(self.lam_domain, self.centers, self.num_close, self.radius)

        # Test the method returns the correct number of samples
        self.assertEqual(self.samples.shape[0], (self.num_close+1)*self.num_centers)

        # Check that the samples are in lam_domain
        for Ldim in range(self.Lambda_dim):
            nptest.assert_array_less(self.lam_domain[Ldim,0], self.samples[:,Ldim])
            nptest.assert_array_less(self.samples[:,Ldim], self.lam_domain[Ldim,1])

    def test_sample_l1_ball(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_l1_ball`.
        """
        self.samples = grad.sample_l1_ball(self.centers, self.num_close, self.radius)

        # Test the method returns the correct number of samples
        self.assertEqual(self.samples.shape[0], (self.num_close+1)*self.num_centers)

class TestFDMethods(unittest.TestCase):
    """
    Test all finite difference methods in bet.sensitivity.gradients
    """
    def setUp(self):
        self.Lambda_dim = 2
        lam_left = np.zeros(self.Lambda_dim)
        lam_right = np.ones(self.Lambda_dim)

        self.lam_domain = np.zeros((self.Lambda_dim, 2))
        self.lam_domain[:,0] = lam_left
        self.lam_domain[:,1] = lam_right

        self.num_centers = 10
        self.centers = np.random.random((self.num_centers,self.Lambda_dim))
        
        np.random.seed(0)
        num_qois = 20
        self.radius = 0.1

    def test_pick_ffd_points(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_linf_ball`.
        """
        self.samples = grad.pick_ffd_points(self.centers, self.radius)

        # Test the method returns the correct number of samples
        self.assertEqual(self.samples.shape[0], (self.Lambda_dim+1)*self.num_centers)

    def test_pick_cfd_points(self):
        """
        Test :meth:`bet.sensitivity.gradients.sample_l1_ball`.
        """
        self.samples = grad.pick_cfd_points(self.centers, self.radius)

        # Test the method returns the correct number of samples
        self.assertEqual(self.samples.shape[0], (2*self.Lambda_dim)*self.num_centers)

class TestRBFMethods(unittest.TestCase):
    """
    Test all RBF methods in bet.sensitivity.gradients
    """
    def setUp(self):
        self.shape_radii = [20,10]
        np.random.seed(0)
        self.radii_mat = np.random.random(self.shape_radii)
        self.radii_vec = np.random.random(self.shape_radii[0])
        self.dxi = np.random.random(self.shape_radii[0])

    def test_radial_basis_function(self):
        """
        Test :meth:`bet.sensitivity.gradients.radial_basis_function`.
        """
        self.rbf = grad.radial_basis_function(self.radii_mat, 'Multiquadric')

        # Test the method returns the correct number of function evaluations
        self.assertEqual(self.rbf.shape, (self.radii_mat.shape))

        # Test the method returns an error when kernel is not available
        with self.assertRaises(ValueError):
            grad.radial_basis_function(self.radii_mat, 'DNE')

    def test_radial_basis_function_dxi(self):
        """
        Test :meth:`bet.sensitivity.gradients.radial_basis_function_dxi`.
        """
        self.rbf = grad.radial_basis_function_dxi(self.radii_vec, self.dxi, 'Multiquadric')

        # Test the method returns the correct number of function evaluations
        self.assertEqual(self.rbf.shape, (self.radii_vec.shape))

        # Test the method returns an error when kernel is not available
        with self.assertRaises(ValueError):
            grad.radial_basis_function_dxi(self.radii_vec, self.dxi, 'DNE')

class TestGradMethods(unittest.TestCase):
    """
    Test all gradient approximation methods in bet.sensitivity.gradients
    """
    def setUp(self):
        self.Lambda_dim = 2

        np.random.seed(0)
        self.num_xeval = 10
        self.num_samples = 1000
        self.radius = 0.01
        self.xeval = np.random.random((self.num_xeval,self.Lambda_dim))

        self.num_qois = 20
        coeffs = np.random.random((self.Lambda_dim, self.num_qois-self.Lambda_dim))
        self.coeffs = np.append(coeffs, np.eye(self.Lambda_dim), axis=1)

    def test_calculate_gradients_rbf(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_rbf`.
        """
        self.samples = np.random.random((self.num_samples,self.Lambda_dim))
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_rbf(self.samples, self.data, self.xeval)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_xeval, self.num_qois, self.Lambda_dim))

        # Test that each vector is normalized
        nptest.assert_array_almost_equal(np.sqrt(np.sum(self.G**2, 2)), np.ones((self.G.shape[0], self.G.shape[1])))
        


    def test_calculate_gradients_cfd(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_cfd`.
        """
        self.samples = grad.pick_cfd_points(self.xeval, self.radius)
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_cfd(self.samples, self.data, self.xeval, self.radius)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_xeval, self.num_qois, self.Lambda_dim))

        # Test that each vector is normalized
        nptest.assert_array_almost_equal(np.sqrt(np.sum(self.G**2, 2)), np.ones((self.G.shape[0], self.G.shape[1])))

    def test_calculate_gradients_ffd(self):
        """
        Test :meth:`bet.sensitivity.gradients.calculate_gradients_ffd`.
        """
        self.samples = grad.pick_ffd_points(self.xeval, self.radius)
        self.data = self.samples.dot(self.coeffs)
        self.G = grad.calculate_gradients_ffd(self.samples, self.data, self.xeval, self.radius)

        # Test the method returns the correct size tensor
        self.assertEqual(self.G.shape, (self.num_xeval, self.num_qois, self.Lambda_dim))

        # Test that each vector is normalized
        nptest.assert_array_almost_equal(np.sqrt(np.sum(self.G**2, 2)), np.ones((self.G.shape[0], self.G.shape[1])))





