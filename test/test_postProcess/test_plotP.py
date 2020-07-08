# Copyright (C) 2014-2020 The BET Development Team

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
import bet.calculateP.calculateR as calculateR
import bet.sampling.basicSampling as bsam


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

        num_samples = 1000

        emulated_input_samples.set_values_local(
            np.linspace(emulated_input_samples.get_domain()[0][0],
                        emulated_input_samples.get_domain()[0][1],
                        num_samples + 1))

        emulated_input_samples.set_probabilities_local(
            1.0 / float(comm.size) * (1.0 / float(
                emulated_input_samples.get_values_local().shape[0])) *
            np.ones((emulated_input_samples.get_values_local().shape[0],)))

        emulated_input_samples.check_num()

        self.samples = emulated_input_samples

    def test_1_bin(self):
        """
        Test that marginals sum to 1 and have correct shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins=1)

        nptest.assert_almost_equal(marginals[0][0], 1.0)
        nptest.assert_equal(marginals[0].shape, (1,))

    def test_10_bins(self):
        """
        Test that marginals sum to 1 and have correct shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins=10)

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
        emulated_input_samples.set_domain(np.array([[0.0, 1.0], [0.0, 1.0]]))

        emulated_input_samples.set_values_local(
            util.meshgrid_ndim((np.linspace(
                emulated_input_samples.get_domain()[0][0],
                emulated_input_samples.get_domain()[0][1],
                10),
                np.linspace(
                emulated_input_samples.get_domain()[1][0],
                emulated_input_samples.get_domain()[1][1],
                10)
            ))
        )

        emulated_input_samples.set_probabilities_local(1.0 / float(comm.size) *
                                                       (
            1.0 / float(
                emulated_input_samples.get_values_local().shape[0])
        ) *
            np.ones(
            (emulated_input_samples.get_values_local().shape[0],)
        )
        )
        emulated_input_samples.check_num()

        self.samples = emulated_input_samples

    def test_1_bin_1D(self):
        """
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins=1)

        nptest.assert_almost_equal(marginals[0][0], 1.0)
        nptest.assert_almost_equal(marginals[1][0], 1.0)
        nptest.assert_equal(marginals[0].shape, (1,))
        nptest.assert_equal(marginals[1].shape, (1,))

    def test_10_bins_1D(self):
        """
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins=10)

        nptest.assert_almost_equal(np.sum(marginals[0]), 1.0)
        nptest.assert_almost_equal(np.sum(marginals[1]), 1.0)
        nptest.assert_equal(marginals[0].shape, (10,))

    def test_1_bin_2D(self):
        """
        Test that 2D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins=1)

        nptest.assert_almost_equal(marginals[(0, 1)][0], 1.0)
        nptest.assert_equal(marginals[(0, 1)].shape, (1, 1))

    def test_10_bins_2D(self):
        """
        Test that 2D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins=10)

        nptest.assert_almost_equal(np.sum(marginals[(0, 1)]), 1.0)
        nptest.assert_equal(marginals[(0, 1)].shape, (10, 10))

    def test_5_10_bins_2D(self):
        """
        Test that 1D marginals sum to 1 and have right shape.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins=[5, 10])

        nptest.assert_almost_equal(np.sum(marginals[(0, 1)]), 1.0)
        nptest.assert_equal(marginals[(0, 1)].shape, (5, 10))

    def test_1D_smoothing(self):
        """
        Test :meth:`bet.postProcess.plotP.smooth_marginals_1D`.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins=10)

        marginals_smooth = plotP.smooth_marginals_1D(marginals, bins,
                                                     sigma=10.0)

        nptest.assert_equal(marginals_smooth[0].shape, marginals[0].shape)
        nptest.assert_almost_equal(np.sum(marginals_smooth[0]), 1.0)

    def test_2D_smoothing(self):
        """
        Test :meth:`bet.postProcess.plotP.smooth_marginals_2D`.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins=10)

        marginals_smooth = plotP.smooth_marginals_2D(marginals, bins,
                                                     sigma=10.0)

        nptest.assert_equal(marginals_smooth[(0, 1)].shape,
                            marginals[(0, 1)].shape)
        nptest.assert_almost_equal(np.sum(marginals_smooth[(0, 1)]), 1.0)

    def test_plot_marginals_1D(self):
        """
        Test :meth:`bet.postProcess.plotP.plot_1D_marginal_probs`.
        """
        (bins, marginals) = plotP.calculate_1D_marginal_probs(self.samples,
                                                              nbins=10)

        try:
            plotP.plot_1D_marginal_probs(marginals, bins, self.samples,
                                         filename="file", interactive=False)
            go = True
            if os.path.exists("file_1D_0.png") and comm.rank == 0:
                os.remove("file_1D_0.png")
            if os.path.exists("file_1D_1.png") and comm.rank == 0:
                os.remove("file_1D_1.png")
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)

    def test_plot_marginals_2D(self):
        """
        Test :meth:`bet.postProcess.plotP.plot_2D_marginal_probs`.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins=10)
        marginals[(0, 1)][0][0] = 0.0
        marginals[(0, 1)][0][1] *= 2.0
        try:
            plotP.plot_2D_marginal_probs(marginals, bins, self.samples,
                                         filename="file", interactive=False)
            plotP.plot_2D_marginal_probs(marginals, bins, self.samples, plot_surface=True,
                                         filename="file", interactive=False)
            go = True
            if os.path.exists("file_2D_0_1.png") and comm.rank == 0:
                os.remove("file_2D_0_1.png")
            if os.path.exists("file_surf_0_1.png") and comm.rank == 0:
                os.remove("file_surf_0_1.png")
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)

    def test_plot_2D_marginal_contours(self):
        """
        Test :meth:`bet.postProcess.plotP.plot_2D_marginal_contours`.
        """
        (bins, marginals) = plotP.calculate_2D_marginal_probs(self.samples,
                                                              nbins=10)
        marginals[(0, 1)][0][0] = 0.0
        marginals[(0, 1)][0][1] *= 2.0
        try:
            plotP.plot_2D_marginal_probs(marginals, bins, self.samples,
                                         filename="file", interactive=False)
            go = True
            if os.path.exists("file_2D_contours_0_1.png") and comm.rank == 0:
                os.remove("file_2D_contours_0_1.png")
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)


@unittest.skipIf(comm.size > 1, 'Only run in serial')
class Test_plot_1d_marginal_densities(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.plot_1d_marginal_densities`.
    """
    def setUp(self):
        def my_model(parameter_samples):
            Q_map = np.array([[0.506, 0.463], [0.253, 0.918], [0.685, 0.496]])
            QoI_samples = np.dot(parameter_samples, Q_map)
            return QoI_samples

        sampler = bsam.sampler(my_model)
        sampler.random_sample_set(rv=[['norm', {'loc': 2, 'scale': 3}],
                                      ['uniform', {'loc': 2, 'scale': 3}],
                                      ['beta', {'a': 2, 'b': 2}]], input_obj=3, num_samples=1000)
        sampler.compute_qoi_and_create_discretization()

        sampler2 = bsam.sampler(my_model)
        sampler2.random_sample_set(rv=[['norm', {'loc': 1, 'scale': 2}],
                                       ['uniform', {'loc': 2, 'scale': 2}],
                                       ['beta', {'a': 2, 'b': 3}]], input_obj=3, num_samples=1000)
        sampler2.compute_qoi_and_create_discretization()

        sampler.discretization.set_output_observed_set(sampler2.discretization.get_output_sample_set())
        self.disc1 = sampler.discretization
        self.disc2 = sampler2.discretization

    def test_rv(self):
        """
        Test plotting random variable probability.
        """
        calculateR.invert_to_random_variable(self.disc1, rv='beta')
        param_labels = [r'$a$', r'$b$', r'$c$']
        for i in range(3):
            plotP.plot_1d_marginal_densities(sets=(self.disc1, self.disc2), i=i,
                                sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                                title="Fitted Beta Distribution", label=param_labels[i], interactive=False)
