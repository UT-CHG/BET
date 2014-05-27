# Lindley Graham 05/22/2014

"""
This module contains tests for :module:`bet.caculcateP.calculateP`.

    * compare using same lambda_emulated (iid and regular grid)
    * compare using different lambda_emulated

Most of these tests should make sure certain values are within a tolerance
rather than exact due to the stocastic nature of the algorithms being tested.
"""

import unittest
import bet.calculateP.calculateP as calcP
import numpy sa np
import scipy.spatial as spatial

class TestEmulateIIDLebesgue(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.emulate_iid_lebesgue`.
    """
    
    def runTest(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.

        """
        lam_left = np.array([0.0, .25, .4])
        lam_right = np.array([1.0, 4.0, .5])

        lam_domain = np.zeros((3,3))
        lam_domain[:,0] = lam_left
        lam_domain[:,1] = lam_right

        num_l_emulate = 1e6

        lambda_emulate = calcP.emulate_iid_lebesgue(lam_domain, num_l_emulate)

        # check the dimension
        np.assert_array_equal(lambda_emulate.shape, (3, num_l_emulate))

        # check that the samples are all within the correct bounds
        np.assertGreaterEqual(0.0, np.min(lambda_emulate[0, :]))
        np.assertGreaterEqual(.25, np.min(lambda_emulate[1, :]))
        np.assertGreaterEqual(.4, np.min(lambda_emulate[2, :]))
        np.assertLessEqual(1.0, np.max(lambda_emulate[0, :]))
        np.assertLessEqual(4.0, np.max(lambda_emulate[1, :]))
        np.assertLessEqual(.5, np.max(lambda_emulate[2, :]))

# test where P is uniform over the entire domain
# test where P is uniform over a hyperrectangle subdomain
# test with and without optional arguments, together and separatly
# compare using same lambda_emulated (iid and regular grid)
# compare using different lambda_emulated
# some will need to be within a tolerance and some will need to be exact
# also tolerances will need to be depend on if the lambda_emulated are the same
# and if there voronoi cells are the same, etc.
# test on a linear model and maybe a non-linear, read from file model

class TestProb(unittest.TestCase):
    """
    Tests ``prob*`` methods in :mod:`bet.calculateP.calculateP`.
    """
    @classmethod
    def setUpClass(cls):
        """
        Create inputs for ``prob*`` methods. This should run only once per
        grouping of tests from this class. But, for isolated tests it doesn't
        make any sense to do it this way. Since many of these tests are
        comparision tests it should be fine to structure things in this manner
        rather using inheritance everywhere.
        """
        # Create model associated inputs (samples, data, lam_domain)
        data_domain = np.array([[0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]])
        lam_domain = np.array([[.1, .2], [3, 4], [50, 60]])
        # iid samples
        u_samples = None #calcP.emulate_iid_lebesgue(lam_domain,
                #20**lam_domain.shape[0])
        # regular grid samples
        lam1 = np.linspace(lam_domain[0, 0], lam_domain[0, 1], 20)
        lam2 = np.linspace(lam_domain[1, 0], lam_domain[1, 1], 20)
        lam3 = np.linspace(lam_domain[2, 0], lam_domain[2, 1], 20)
        lam1, lam2, lam3 = np.meshgrid(lam1, lam2, lam3)
        r_samples = np.column_stack((lam1.ravel(), lam2.ravel(), lam3.ravel())
        # linear model
        rl_data = np.dot(r_samples, data_domain)
        ul_data = np.dot(r_samples, data_domain)
        # non-linear model
        def nonlinear_model(l_data):
            n_data = l_data
            n_data[:,0] = l_data[:,0] + l_data[:,2]
            n_data[:,1] = np.square(l_data[:,1])
            n_data[:,2] = l_data[:,0] - n_data[:,1]
            return n_data
        rn_data = nonlinear_model(rl_data)
        un_data = nonlinear_model(ul_data)
    
        # Create rho_D_M associated inputs (rho_D_M, d_distr_samples, d_tree)
        # d_distr_samples (iid or regular)
        data1 = np.linspace(data_domain[0, 0], data_domain[0, 1], 10)
        data2 = np.linspace(data_domain[1, 0], data_domain[1, 1], 10)
        data3 = np.linspace(data_domain[2, 0], data_domain[2, 1], 10)
        data1, data2, data3 = np.meshgrid(data1, data2, data3)
        r_dsamples = np.column_stack((data1.ravel(), data2.ravel(), data3.ravel())
        rd_tree = spatial.KDTree(r_dsamples)
        # rho_D_M uniform or hyperrectangle


        
        # Create lambda_emulate
        # iid
        u_lambda_emulate = None #calcP.emulate_iid_lebesgue(lam_domain,
            #30**lam_domain.shape[0])
        # regular grid
        lam1 = np.linspace(lam_domain[0, 0], lam_domain[0, 1], 40)
        lam2 = np.linspace(lam_domain[1, 0], lam_domain[1, 1], 40)
        lam3 = np.linspace(lam_domain[2, 0], lam_domain[2, 1], 40)
        lam1, lam2, lam3 = np.meshgrid(lam1, lam2, lam3)
        r_lambda_emulate = np.column_stack((lam1.ravel(), lam2.ravel(), lam3.ravel())
        


class TestVol(unittest.TestCase)
    def tearDown(self):
        """
        Get rid of rho_D_M, d_distr_samples, d_tree
        """
        self.rho_D_M = None
        self.d_distr_samples = None
        self.d_tree = None

class TestUnifVol(unittest.TestCase):
    """
    Tests ``prob_*`` methods using a uniform distribution on the entire
    lam_domain
    """
    def setUp(self):
        """
        Creates rho_D_M, d_distr_samples, d_Tree
        """
        pass

class TestModel(unittest.TestCase):
    def tearDown(self):
        """
        Get rid of  samples, data, lam_domain, lambda_emulate
        """
        self.samples = None
        self.data = None
        self.lam_domain = None

class TestLinearModel(TestModel):
    """
    Tests ``prob_*`` methods using a linear model
    """
    def setUp(self):
        """
        Creates samples, data, lam_domain, lambda_emulate
        """
        pass


