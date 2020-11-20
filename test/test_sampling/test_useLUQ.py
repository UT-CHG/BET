# Copyright (C) 2014-2020 The BET Development Team

"""
This module contains unittests for :mod:`~bet.sampling.useLUQ`
"""

import unittest
import os
import pyDOE
import numpy.testing as nptest
import numpy as np
import scipy.io as sio
import bet
import bet.sampling.basicSampling as bsam
import bet.sampling.useLUQ as useLUQ
from bet.Comm import comm
import bet.sample
from bet.sample import sample_set
from bet.sample import discretization as disc
import collections

try:
    from luq.luq import LUQ
    has_luq = True
except ImportError:
    has_luq = False


@unittest.skipIf(not has_luq, 'LUQ is not installed.')
@unittest.skipIf(comm.size > 1, 'Only run in serial')
class Test_useLUQ(unittest.TestCase):
    """
    Testing ``bet.sampling.useLUQ.useLUQ``, interfacing with a model.
    """
    def setUp(self):
        np.random.seed(123456)
        self.p_set = bsam.random_sample_set(rv=[['uniform', {'loc': .01, 'scale': 0.114}],
                                        ['uniform', {'loc': .05, 'scale': 1.45}]],
                                       input_obj=2, num_samples=20)

        self.o_set = bsam.random_sample_set(rv=[['beta', {'a': 2, 'b': 2, 'loc': .01, 'scale': 0.114}],
                                           ['beta', {'a': 2, 'b': 2, 'loc': .05, 'scale': 1.45}]],
                                       input_obj=2, num_samples=20)
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

    def test_nums(self):
        """
        Check the number of samples.
        """
        self.disc1.check_nums()
        self.disc2.check_nums()

    def test_dims(self):
        """
        Check the dimensions.
        """
        assert self.disc1.get_output_sample_set().get_dim() == 2
        assert self.disc2.get_output_sample_set().get_dim() == 2

    def test_sets(self):
        """
        Check the sets
        """
        assert self.disc1.get_input_sample_set() == self.p_set
        assert self.disc2.get_input_sample_set() == self.o_set
        assert self.disc1.get_output_observed_set() == self.disc2.get_output_sample_set()

    def test_saving(self):
        """
        Test saving.
        """
        savefile = 'test_save_useLUQ'
        self.luq.save(savefile)
        loaded = bet.util.load_object(file_name=savefile)
        disc1, disc2 = loaded.make_disc()
        assert disc1 == self.disc1
        assert disc2 == self.disc2

        if comm.rank == 0:
            os.remove(savefile + '.p')
