# Copyright (C) 2014-2015 Lindley Graham and Steven Mattis

# Steven Mattis 04/07/2015
"""
This module contains tests for :module:`bet.postProcess.plotP`.


Tests for correct computation of marginals and plotting.
"""

import unittest
import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as simpleFunP
import bet.postProcess.plotP as plotP
import numpy as np
import scipy.spatial as spatial
import numpy.testing as nptest
import bet.util as util
from bet.Comm import comm
import os


class Test_calc_marg_1D(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.calculate_1D_marginal_probs`
    for a 1D parameter space.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.lam_domain=np.array([[0.0,1.0]])
        num_samples=1000
        self.samples = np.linspace(self.lam_domain[0][0], self.lam_domain[0][1], num_samples+1)
        self.P_samples = 1.0/float(comm.size)*(1.0/float(self.samples.shape[0]))*np.ones((self.samples.shape[0],))
        
    def test_1_bin(self):
        """
        Test that marginals sum to 1 and have correct shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 1)
        nptest.assert_almost_equal(marginals[0][0], 1.0)
        nptest.assert_equal(marginals[0].shape, (1,))

    def test_10_bins(self):
        """
        Test that marginals sum to 1 and have correct shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 10)
        nptest.assert_almost_equal(np.sum(marginals[0]), 1.0)
        nptest.assert_equal(marginals[0].shape, (10,))

class Test_calc_marg_2D(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.calculate_1D_marginal_probs` and  :meth:`bet.postProcess.plotP.calculate_2D_marginal_probs` for a 2D
    parameter space.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.lam_domain=np.array([[0.0,1.0],[0.0,1.0]])
        self.samples=util.meshgrid_ndim((np.linspace(self.lam_domain[0][0], self.lam_domain[0][1], 10),np.linspace(self.lam_domain[1][0], self.lam_domain[1][1], 10)))
        self.P_samples = 1.0/float(comm.size)*(1.0/float(self.samples.shape[0]))*np.ones((self.samples.shape[0],))
        
    def test_1_bin_1D(self):
        """ 
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 1)
        
        nptest.assert_almost_equal(marginals[0][0], 1.0)
        nptest.assert_almost_equal(marginals[1][0], 1.0)
        nptest.assert_equal(marginals[0].shape, (1,))
        nptest.assert_equal(marginals[1].shape, (1,))

    def test_10_bins_1D(self):
        """ 
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 10)
        nptest.assert_almost_equal(np.sum(marginals[0]), 1.0)
        nptest.assert_almost_equal(np.sum(marginals[1]), 1.0)
        nptest.assert_equal(marginals[0].shape, (10,))

    def test_1_bin_2D(self):
        """ 
        Test that 2D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 1)
        
        nptest.assert_almost_equal(marginals[(0,1)][0], 1.0)
        nptest.assert_equal(marginals[(0,1)].shape, (1,1))

    def test_10_bins_2D(self):
        """ 
        Test that 2D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 10)
        nptest.assert_almost_equal(np.sum(marginals[(0,1)]), 1.0)
        nptest.assert_equal(marginals[(0,1)].shape, (10,10))

    def test_5_10_bins_2D(self):
        """ 
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = [5,10])
        nptest.assert_almost_equal(np.sum(marginals[(0,1)]), 1.0)
        nptest.assert_equal(marginals[(0,1)].shape, (5,10))


    def test_1D_smoothing(self):
        """
        Test :meth:`bet.postProcess.plotP.smooth_marginals_1D`.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 10)
        marginals_smooth = plotP.smooth_marginals_1D(marginals, bins, sigma = 10.0)
        nptest.assert_equal(marginals_smooth[0].shape,  marginals[0].shape)
        nptest.assert_almost_equal(np.sum(marginals_smooth[0]), 1.0)

    def test_2D_smoothing(self):
        """
        Test :meth:`bet.postProcess.plotP.smooth_marginals_2D`.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 10)
        marginals_smooth = plotP.smooth_marginals_2D(marginals, bins, sigma = 10.0)
        nptest.assert_equal(marginals_smooth[(0,1)].shape,  marginals[(0,1)].shape)
        nptest.assert_almost_equal(np.sum(marginals_smooth[(0,1)]), 1.0)

    def test_plot_marginals_1D(self):
        """
        Test :meth:`bet.postProcess.plotP.plot_1D_marginal_probs`.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 10)
        try:
            plotP.plot_1D_marginal_probs(marginals, bins,self.lam_domain, filename = "file", interactive=False)
            go = True
            if os.path.exists("file_1D_0.eps"):
                os.remove("file_1D_0.eps")
            if os.path.exists("file_1D_1.eps"):
                os.remove("file_1D_1.eps")
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)

    def test_plot_marginals_2D(self):
        """
        Test :meth:`bet.postProcess.plotP.plot_2D_marginal_probs`.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.P_samples,
                                                              self.samples,
                                                              self.lam_domain,
                                                              nbins = 10)
        marginals[(0,1)][0][0]=0.0
        marginals[(0,1)][0][1]*=2.0
        try:
            plotP.plot_2D_marginal_probs(marginals, bins,self.lam_domain, filename = "file", interactive=False)
            go = True
            if os.path.exists("file_2D_0_1.eps"):
                os.remove("file_2D_0_1.eps")
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)
