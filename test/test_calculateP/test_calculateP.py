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

class test_emulate_iid_lebesgue(unittest.TestCase):
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

# calculate P using a uniform distribution over the data space
# calculate P using a uniform distribution over a hyperrectangle subdomain for the data space
# calculate P using a linear map for the QoI map

# prob_emulated with option where lambda_emulate=None
# prob_emulated with option where d_Tree=None
# prob_emulated with both optional inputs None
# test on regular grid of samples and iid grid of samples
# make sure probabilties and lam_vol follow the MC assumption (i.e. for uniform
# over the entire space they should all be the same, the hyperrectangle case
# will be different)
# make sure lambda_emulate is correct
# make sure io_ptr and emulate_ptr are correct

# compare the P calculated by the different methods on the same samples 

# add a skip thing to the tests involving qhull so that it only runs if that
# package is installed

class test_unif_vol(unittest.TestCase):
    """
    Tests ``prob_*`` methods using a uniform distribution on the entire
    lam_domain
    """
    def setUp(self):
        """
        Creates samples, data, rho_D_M, d_distr_samples, lam_domain,
        lambda_emulate?, d_Tree?
        """
        pass
