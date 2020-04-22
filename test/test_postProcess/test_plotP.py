# Copyright (C) 2014-2019 The BET Development Team

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
import bet.calculateP.dataConsistent as dc
import bet.sampling.basicSampling as bsam
import bet.sampling.useLUQ as useLUQ


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
            go = True
            if os.path.exists("file_2D_0_1.png") and comm.rank == 0:
                os.remove("file_2D_0_1.png")
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


class Test_plot_marginal(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.plot_marginal`.
    """
    def setUp(self):
        np.random.seed(123456)
        self.p_set = bsam.random_sample_set(rv=[['uniform', {'loc': .01, 'scale': 0.114}],
                                        ['uniform', {'loc': .05, 'scale': 1.45}]],
                                       input_obj=2, num_samples=50)

        self.o_set = bsam.random_sample_set(rv=[['beta', {'a': 2, 'b': 2, 'loc': .01, 'scale': 0.114}],
                                           ['beta', {'a': 2, 'b': 2, 'loc': .05, 'scale': 1.45}]],
                                       input_obj=2, num_samples=50)
        time_start = 2.0  # 0.5
        time_end = 6.5  # 40.0
        num_time_preds = int((time_end - time_start) * 100)
        self.times = np.linspace(time_start, time_end, num_time_preds)

        self.luq = useLUQ.useLUQ(predict_set=self.p_set, obs_set=self.o_set, lb_model=useLUQ.myModel, times=self.times)
        self.luq.setup()

        time_start_idx = 0
        time_end_idx = len(self.luq.times) - 1
        self.luq.clean_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
                            num_clean_obs=20, tol=5.0e-2, min_knots=3, max_knots=12)
        self.luq.dynamics(cluster_method='kmeans', kwargs={'n_clusters': 3, 'n_init': 10})
        self.luq.learn_qois_and_transform(num_qoi=2)
        self.disc1, self.disc2 = self.luq.make_disc()

    def test_rv(self):
        """
        Test plotting random variable probability.
        """
        dc.invert_to_random_variable(self.disc1, rv='beta')
        param_labels = [r'$a$', r'$b$']
        for i in range(2):
            plotP.plot_marginal(sets=(self.disc1, self.disc2), i=i,
                                sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                                title="Fitted Beta Distribution", label=param_labels[i], interactive=False)
