# Steven Mattis 04/07/2015
"""
This module contains tests for :module:`bet.postProcess.postTools`.


Tests for correct post-processing.
"""
import unittest
import bet.calculateP.simpleFunP as simpleFunP
import bet.postProcess.postTools as postTools
import numpy as np
import scipy.spatial as spatial
import numpy.testing as nptest
import bet.util as util
from bet.Comm import *

class Test_PostTools(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.postTools`.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.lam_domain=np.array([[0.0,1.0]])
        num_samples=1000
        self.samples = np.linspace(self.lam_domain[0][0], self.lam_domain[0][1], num_samples+1)
        self.P_samples = (1.0/float(self.samples.shape[0]))*np.ones((self.samples.shape[0],))
        self.P_samples[0] = 0.0
        self.P_samples[-1] *= 2.0

        self.data = self.samples[:]
        
    def test_sort_by_rho(self):
        """
        Test :meth:`bet.postProcess.postTools.sort_by_rho`.
        """
        (P_samples, samples, _ , data) = postTools.sort_by_rho(self.P_samples, self.samples,
                                                               lam_vol=None, data=self.data)
        self.assertGreater(np.min(P_samples),0.0)
        nptest.assert_almost_equal(np.sum(P_samples),1.0)

    def test_sample_highest_prob(self):
        """
        Test :meth:`bet.postProcess.postTools.sample_highest_prob`.
        """
        (num_samples,P_samples, samples, _ , data) = postTools.sample_highest_prob(1.0,
                                                                                  self.P_samples, 
                                                                                  self.samples,
                                                                                  lam_vol=None, data=self.data, sort=True)
        nptest.assert_almost_equal(np.sum(P_samples),1.0)
        nptest.assert_equal(num_samples,1000)
        
        (num_samples,P_samples, samples, _ , data) = postTools.sample_highest_prob(0.8,
                                                                                  self.P_samples, 
                                                                                  self.samples,
                                                                                  lam_vol=None, data=self.data, sort=True)
        nptest.assert_allclose(np.sum(P_samples),0.8,0.001)
