# Steven Mattis and Lindley Graham 04/06/2015
"""
This module contains tests for :module:`bet.calculateP.calculateP`.


Most of these tests should make sure certain values are within a tolerance
rather than exact due to the stocastic nature of the algorithms being tested.
"""
import os
import unittest
import bet
import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as simpleFunP
import numpy as np
import scipy.spatial as spatial
import numpy.testing as nptest
import bet.util as util
from bet.Comm import *

data_path = os.path.dirname(bet.__file__) + "/../test/test_calculateP/datafiles"

class TestEmulateIIDLebesgue(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.emulate_iid_lebesgue`.
    """
    
    def setUp(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.

        """
        lam_left = np.array([0.0, .25, .4])
        lam_right = np.array([1.0, 4.0, .5])

        self.lam_domain = np.zeros((3,3))
        self.lam_domain[:,0] = lam_left
        self.lam_domain[:,1] = lam_right

        self.num_l_emulate = 1000000

        self.lambda_emulate = calcP.emulate_iid_lebesgue(self.lam_domain, self.num_l_emulate)
 
    def test_dimension(self):
        """
        Check the dimension.
        """
        nptest.assert_array_equal(self.lambda_emulate.shape, (int(self.num_l_emulate/size)+1,3))

    def test_bounds(self):
        """
        Check that the samples are all within the correct bounds
        """
        self.assertGreaterEqual(np.min(self.lambda_emulate[:, 0]),0.0)
        self.assertGreaterEqual(np.min(self.lambda_emulate[:,1]), 0.25)
        self.assertGreaterEqual(np.min(self.lambda_emulate[:, 2]), 0.4)
        self.assertLessEqual(np.max(self.lambda_emulate[:, 0]),1.0)
        self.assertLessEqual(np.max(self.lambda_emulate[:,1]), 4.0)
        self.assertLessEqual(np.max(self.lambda_emulate[:,2]), 0.5)

class prob:
    def test_prob_sum_to_1(self):
        """
        Test to see if the prob. sums to 1.
        """
        nptest.assert_almost_equal(np.sum(self.P),1.0)
    #@unittest.skipIf(size > 1, 'Only run in serial')
    def test_P_matches_true(self):
        """
        Test against reference probs. (Only in serial)
        """
        nptest.assert_almost_equal(self.P_ref,self.P)
    def test_vol_sum_to_1(self):
        """
        Test that volume ratios sum to 1.
        """
        nptest.assert_almost_equal(np.sum(self.lam_vol), 1.0)
    def test_prob_pos(self):
        """
        Test that all probs are non-negative.
        """
        self.assertEqual(np.sum(np.less(self.P,0)),0)

class prob_emulated:
    def test_P_sum_to_1(self):
        """
        Test that prob. sums to 1.
        """
        nptest.assert_almost_equal(np.sum(self.P_emulate),1.0)
    def test_P_matches_true(self):
        """
        Test that probabilites match reference values.
        """
        if size == 1:
            nptest.assert_almost_equal(self.P_emulate_ref,self.P_emulate)
    def test_prob_pos(self):
        """
        Test that all probabilites are non-negative.
        """
        self.assertEqual(np.sum(np.less(self.P_emulate,0)),0)

class prob_mc:
    def test_P_sum_to_1(self):
        """
        Test that probs sum to 1.
        """
        nptest.assert_almost_equal(np.sum(self.P),1.0)
    def test_P_matches_true(self):
        """
        Test the probs. match reference values.
        """
        if size==1:
            nptest.assert_almost_equal(self.P_ref,self.P)
    def test_vol_sum_to_1(self):
        """
        Test that volume ratios sum to 1.
        """
        nptest.assert_almost_equal(np.sum(self.lam_vol), 1.0)
    def test_prob_pos(self):
        """
        Test that all probs are non-negative.
        """
        self.assertEqual(np.sum(np.less(self.P,0)),0)
        
    

class TestProbMethod_3to2(unittest.TestCase):
    """
    Sets up 3 to 2 map problem.
    """
    def setUp(self):
        self.samples = np.loadtxt(data_path + "/3to2_samples.txt.gz")
        self.data = np.loadtxt(data_path + "/3to2_data.txt.gz")
        Q_ref =  np.array([0.422, 0.9385])
        (self.d_distr_prob, self.d_distr_samples, self.d_Tree) = simpleFunP.uniform_hyperrectangle(data=self.data,Q_ref=Q_ref, bin_ratio=0.2, center_pts_per_edge = 1)
        self.lam_domain= np.array([[0.0, 1.0],
                                   [0.0, 1.0],
                                   [0.0, 1.0]])
        import numpy.random as rnd
        rnd.seed(1)
        self.lambda_emulate = calcP.emulate_iid_lebesgue(lam_domain=self.lam_domain, 
                                                  num_l_emulate = 1000)


    
class Test_prob_3to2(TestProbMethod_3to2,prob):
    """
    Test :meth:`bet.calculateP.calculateP.prob` on 3 to 2 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_3to2, self).setUp()
        (self.P, self.lam_vol , _  ) = calcP.prob(samples=self.samples,
                                                  data=self.data,
                                                  rho_D_M = self.d_distr_prob,
                                                  d_distr_samples = self.d_distr_samples,
                                                  d_Tree = self.d_Tree)
        self.P_ref = np.loadtxt(data_path + "/3to2_prob.txt.gz")



class Test_prob_emulated_3to2(TestProbMethod_3to2, prob_emulated):
    """
    Test :meth:`bet.calculateP.calculateP.prob_emulated` on a 3 to 2 map.
    """
    def setUp(self):
        """
        Set up 3 to 2 map.
        """
        super(Test_prob_emulated_3to2, self).setUp()
        (self.P_emulate, self.lambda_emulate, _ , _) = calcP.prob_emulated(samples=self.samples,
                                                              data=self.data,
                                                              rho_D_M = self.d_distr_prob,
                                                              d_distr_samples = self.d_distr_samples,
                                                              lambda_emulate = self.lambda_emulate,
                                                              d_Tree = self.d_Tree)
        self.P_emulate_ref=np.loadtxt(data_path + "/3to2_prob_emulated.txt.gz")
        self.P_emulate = util.get_global_values(self.P_emulate)



class Test_prob_mc_3to2(TestProbMethod_3to2, prob_mc):
    """
    Test :meth:`bet.calculateP.calculateP.prob_mc` on a 3 to 2 map.
    """
    def setUp(self):
        """
        Set up 3 to 2 problem.
        """
        super(Test_prob_mc_3to2, self).setUp()
        (self.P, self.lam_vol , _ , _, _) = calcP.prob_mc(samples=self.samples,
                                                           data=self.data,
                                                           rho_D_M = self.d_distr_prob,
                                                           d_distr_samples = self.d_distr_samples,
                                                           lambda_emulate = self.lambda_emulate,
                                                           d_Tree = self.d_Tree)
        self.P_ref = np.loadtxt(data_path + "/3to2_prob_mc.txt.gz")

 

class TestProbMethod_3to1(unittest.TestCase):
    """
    Set up 3 to 1 map problem.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.samples = np.loadtxt(data_path + "/3to2_samples.txt.gz")
        self.data = np.loadtxt(data_path + "/3to2_data.txt.gz")[:,0]
        Q_ref =  np.array([0.422])
        (self.d_distr_prob, self.d_distr_samples, self.d_Tree) = simpleFunP.uniform_hyperrectangle(data=self.data,Q_ref=Q_ref, bin_ratio=0.2, center_pts_per_edge = 1)
        self.lam_domain= np.array([[0.0, 1.0],
                                   [0.0, 1.0],
                                   [0.0, 1.0]])
        import numpy.random as rnd
        rnd.seed(1)
        self.lambda_emulate = calcP.emulate_iid_lebesgue(lam_domain=self.lam_domain, 
                                                  num_l_emulate = 1000)
class Test_prob_3to1(TestProbMethod_3to1, prob):
    """
    Test :meth:`bet.calculateP.calculateP.prob` on a 3 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_3to1, self).setUp()
        (self.P, self.lam_vol, _ ) = calcP.prob(samples=self.samples,
                                                     data=self.data,
                                                     rho_D_M = self.d_distr_prob,
                                                     d_distr_samples = self.d_distr_samples,
                                                     d_Tree = self.d_Tree)
        self.P_ref = np.loadtxt(data_path + "/3to1_prob.txt.gz")


class Test_prob_emulated_3to1(TestProbMethod_3to1, prob_emulated):
    """
    Test :meth:`bet.calculateP.calculateP.prob_emulated` on a 3 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_emulated_3to1, self).setUp()
        (self.P_emulate, self.lambda_emulate, _ , _) = calcP.prob_emulated(samples=self.samples,
                                                              data=self.data,
                                                              rho_D_M = self.d_distr_prob,
                                                              d_distr_samples = self.d_distr_samples,
                                                              lambda_emulate = self.lambda_emulate,
                                                              d_Tree = self.d_Tree)
        self.P_emulate_ref=np.loadtxt(data_path + "/3to1_prob_emulated.txt.gz")
        self.P_emulate = util.get_global_values(self.P_emulate)



class Test_prob_mc_3to1(TestProbMethod_3to1, prob_mc):
    """
    Test :meth:`bet.calculateP.calculateP.prob_mc` on a 3 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_mc_3to1, self).setUp()
        (self.P, self.lam_vol , _ , _, _) = calcP.prob_mc(samples=self.samples,
                                                           data=self.data,
                                                           rho_D_M = self.d_distr_prob,
                                                           d_distr_samples = self.d_distr_samples,
                                                           lambda_emulate = self.lambda_emulate,
                                                           d_Tree = self.d_Tree)
        self.P_ref = np.loadtxt(data_path + "/3to1_prob_mc.txt.gz")

  
class TestProbMethod_10to4(unittest.TestCase):
    """
    Sets up 10 to 4 map problem.
    """
    def setUp(self):
        """
        Set up problem.
        """
        import numpy.random as rnd
        rnd.seed(1)
        self.lam_domain=np.zeros((10,2))
        self.lam_domain[:,0]=0.0
        self.lam_domain[:,1]=1.0
        self.num_l_emulate = 1000
        self.lambda_emulate = calcP.emulate_iid_lebesgue(self.lam_domain, self.num_l_emulate)
        self.samples =  calcP.emulate_iid_lebesgue(self.lam_domain, 100)
        self.data = np.dot(self.samples,rnd.rand(10,4))
        Q_ref =  np.mean(self.data, axis=0)
        (self.d_distr_prob, self.d_distr_samples, self.d_Tree) = simpleFunP.uniform_hyperrectangle(data=self.data,Q_ref=Q_ref, bin_ratio=0.2, center_pts_per_edge = 1)

    @unittest.skip("No reference data")
    def test_P_matches_true(self):
        pass

class Test_prob_10to4(TestProbMethod_10to4, prob):
    """
    Test :meth:`bet.calculateP.calculateP.prob` on a 10 to 4 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_10to4, self).setUp()
        (self.P, self.lam_vol , _ ) = calcP.prob(samples=self.samples,
                                                     data=self.data,
                                                     rho_D_M = self.d_distr_prob,
                                                     d_distr_samples = self.d_distr_samples,
                                                     d_Tree = self.d_Tree)

    

class Test_prob_emulated_10to4(TestProbMethod_10to4, prob_emulated):
    """
    Test :meth:`bet.calculateP.calculateP.prob_emulated` on a 10 to 4 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_emulated_10to4, self).setUp()

        (self.P_emulate, self.lambda_emulate, _ , _) = calcP.prob_emulated(samples=self.samples,
                                                              data=self.data,
                                                              rho_D_M = self.d_distr_prob,
                                                              d_distr_samples = self.d_distr_samples,
                                                              lambda_emulate = self.lambda_emulate,
                                                              d_Tree = self.d_Tree)
        self.P_emulate = util.get_global_values(self.P_emulate)




class Test_prob_mc_10to4(TestProbMethod_10to4, prob_mc):
    """
    Test :meth:`bet.calculateP.calculateP.prob_mc` on a 10 to 4 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_mc_10to4, self).setUp()
        (self.P, self.lam_vol , _ , _, _) = calcP.prob_mc(samples=self.samples,
                                                           data=self.data,
                                                           rho_D_M = self.d_distr_prob,
                                                           d_distr_samples = self.d_distr_samples,
                                                           lambda_emulate = self.lambda_emulate,
                                                           d_Tree = self.d_Tree)


class TestProbMethod_1to1(unittest.TestCase):
    """
    Sets up 1 to 1 map problem. Uses vectors instead of 2D arrays.
    """
    def setUp(self):
        """
        Set up problem.
        """
        import numpy.random as rnd
        rnd.seed(1)
        self.lam_domain=np.zeros((1,2))
        self.lam_domain[0,0]=0.0
        self.lam_domain[0,1]=1.0
        self.num_l_emulate = 1000
        self.lambda_emulate = calcP.emulate_iid_lebesgue(self.lam_domain, self.num_l_emulate)
        self.samples =  rnd.rand(100,)
        self.data = 2.0*self.samples
        Q_ref =  np.mean(self.data, axis=0)
        (self.d_distr_prob, self.d_distr_samples, self.d_Tree) = simpleFunP.uniform_hyperrectangle(data=self.data,Q_ref=Q_ref, bin_ratio=0.2, center_pts_per_edge = 1)
    @unittest.skip("No reference data")
    def test_P_matches_true(self):
        pass

class Test_prob_1to1(TestProbMethod_1to1, prob):
    """
    Test :meth:`bet.calculateP.calculateP.prob` on a 1 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_1to1, self).setUp()
        (self.P, self.lam_vol, _ ) = calcP.prob(samples=self.samples,
                                                     data=self.data,
                                                     rho_D_M = self.d_distr_prob,
                                                     d_distr_samples = self.d_distr_samples,
                                                     d_Tree = self.d_Tree)


class Test_prob_emulated_1to1(TestProbMethod_1to1, prob_emulated):
    """
    Test :meth:`bet.calculateP.calculateP.prob_emulated` on a 1 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_emulated_1to1, self).setUp()

        (self.P_emulate, self.lambda_emulate, _ , _) = calcP.prob_emulated(samples=self.samples,
                                                              data=self.data,
                                                              rho_D_M = self.d_distr_prob,
                                                              d_distr_samples = self.d_distr_samples,
                                                              lambda_emulate = self.lambda_emulate,
                                                              d_Tree = self.d_Tree)
        self.P_emulate = util.get_global_values(self.P_emulate)


class Test_prob_mc_1to1(TestProbMethod_1to1, prob_mc):
    """
    Test :meth:`bet.calculateP.calculateP.prob_mc` on a 1 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_mc_1to1, self).setUp()
        (self.P, self.lam_vol , _ , _, _) = calcP.prob_mc(samples=self.samples,
                                                           data=self.data,
                                                           rho_D_M = self.d_distr_prob,
                                                           d_distr_samples = self.d_distr_samples,
                                                           lambda_emulate = self.lambda_emulate,
                                                           d_Tree = self.d_Tree)


