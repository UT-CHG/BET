# Copyright (C) 2014-2015 The BET Development Team

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
import bet.sample as sample


class Test_calc_marg_1D(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.calculate_1D_marginal_probs`
    for a 1D parameter space.
    """
    def setUp(self):
        """
        Set up problem.
        """
        emulated_input_samples = sample.sample_set(1)
        emulated_input_samples.set_domain(np.array([[0.0, 1.0]]))

        num_samples=1000

        emulated_input_samples.set_values_local(np.linspace(emulated_input_samples.get_domain()[0][0],
                                             emulated_input_samples.get_domain()[0][1],
                                             num_samples+1))

        emulated_input_samples.set_probabilities_local(1.0/float(comm.size)*(1.0/float(\
                emulated_input_samples.get_values_local().shape[0]))\
                *np.ones((emulated_input_samples.get_values_local().shape[0],)))

        emulated_input_samples.check_num()

        self.samples = emulated_input_samples

    def test_1_bin(self):
        """
        Test that marginals sum to 1 and have correct shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins = 1)

        nptest.assert_almost_equal(marginals[0][0], 1.0)
        nptest.assert_equal(marginals[0].shape, (1,))

    def test_10_bins(self):
        """
        Test that marginals sum to 1 and have correct shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins = 10)

        nptest.assert_almost_equal(np.sum(marginals[0]), 1.0)
        nptest.assert_equal(marginals[0].shape, (10,))

class Test_calc_marg_2D(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.calculate_1D_marginal_probs` and
    :meth:`bet.postProcess.plotP.calculate_2D_marginal_probs` for a 2D
    parameter space.
    """
    def setUp(self):
        """
        Set up problem.
        """
        emulated_input_samples = sample.sample_set(2)
        emulated_input_samples.set_domain(np.array([[0.0,1.0],[0.0,1.0]]))

        emulated_input_samples.set_values_local(util.meshgrid_ndim((np.linspace(emulated_input_samples.get_domain()[0][0],
            emulated_input_samples.get_domain()[0][1], 10),
            np.linspace(emulated_input_samples.get_domain()[1][0],
                emulated_input_samples.get_domain()[1][1], 10))))

        emulated_input_samples.set_probabilities_local(1.0/float(comm.size)*\
                (1.0/float(emulated_input_samples.get_values_local().shape[0]))*\
                np.ones((emulated_input_samples.get_values_local().shape[0],)))
        emulated_input_samples.check_num()

        self.samples = emulated_input_samples

    def test_1_bin_1D(self):
        """ 
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins = 1)

        nptest.assert_almost_equal(marginals[0][0], 1.0)
        nptest.assert_almost_equal(marginals[1][0], 1.0)
        nptest.assert_equal(marginals[0].shape, (1,))
        nptest.assert_equal(marginals[1].shape, (1,))

    def test_10_bins_1D(self):
        """ 
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins = 10)

        nptest.assert_almost_equal(np.sum(marginals[0]), 1.0)
        nptest.assert_almost_equal(np.sum(marginals[1]), 1.0)
        nptest.assert_equal(marginals[0].shape, (10,))

    def test_1_bin_2D(self):
        """ 
        Test that 2D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins = 1)

        nptest.assert_almost_equal(marginals[(0,1)][0], 1.0)
        nptest.assert_equal(marginals[(0,1)].shape, (1,1))

    def test_10_bins_2D(self):
        """ 
        Test that 2D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins = 10)

        nptest.assert_almost_equal(np.sum(marginals[(0,1)]), 1.0)
        nptest.assert_equal(marginals[(0,1)].shape, (10,10))

    def test_5_10_bins_2D(self):
        """ 
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins = [5,10])

        nptest.assert_almost_equal(np.sum(marginals[(0,1)]), 1.0)
        nptest.assert_equal(marginals[(0,1)].shape, (5,10))


    def test_1D_smoothing(self):
        """
        Test :meth:`bet.postProcess.plotP.smooth_marginals_1D`.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins = 10)

        marginals_smooth = plotP.smooth_marginals_1D(marginals, bins, sigma = 10.0)

        nptest.assert_equal(marginals_smooth[0].shape,  marginals[0].shape)
        nptest.assert_almost_equal(np.sum(marginals_smooth[0]), 1.0)

    def test_2D_smoothing(self):
        """
        Test :meth:`bet.postProcess.plotP.smooth_marginals_2D`.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins = 10)

        marginals_smooth = plotP.smooth_marginals_2D(marginals, bins, sigma = 10.0)

        nptest.assert_equal(marginals_smooth[(0,1)].shape,  marginals[(0,1)].shape)
        nptest.assert_almost_equal(np.sum(marginals_smooth[(0,1)]), 1.0)

    def test_plot_marginals_1D(self):
        """
        Test :meth:`bet.postProcess.plotP.plot_1D_marginal_probs`.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins = 10)

        try:
            plotP.plot_1D_marginal_probs(marginals, bins, self.samples,
                                         filename = "file", interactive=False)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)

    def test_plot_marginals_2D(self):
        """
        Test :meth:`bet.postProcess.plotP.plot_2D_marginal_probs`.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins = 10)
        marginals[(0,1)][0][0]=0.0
        marginals[(0,1)][0][1]*=2.0
        try:
            plotP.plot_2D_marginal_probs(marginals, bins, self.samples,
                                         filename = "file", interactive=False)
            go = True
            if os.path.exists("file_2D_0_1.png") and comm.rank == 0:
                os.remove("file_2D_0_1.png")
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)

