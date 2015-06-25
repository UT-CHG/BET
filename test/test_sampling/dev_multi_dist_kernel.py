# Copyright (C) 2014-2015 The BET Development Team
"""
This module contains unittests for :mod:`~bet.sampling.dev_multidist_kernel`
"""

import unittest, os
import numpy.testing as nptest
import numpy as np
import bet.sampling.adaptiveSampling as asam
import scipy.io as sio
from bet.Comm import *
import bet
from test_adaptiveSampling import kernel

class multi_dist_kernel(kernel):
    """
    Test :class:`bet.sampling.adaptiveSampling.multi_dist_kernel`
    """
    def setUp(self):
        """
        Set up
        """
        self.kernel = asam.multi_dist_kernel()

    def test_init(self):
        """
        Test the initalization of
        :class:`bet.sampling.adaptiveSampling.multi_dist_kernel`
        """
        assert self.kernel.radius == None
        assert self.kernel.mean == None
        assert self.kernel.current_clength == 0
        assert self.kernel.TOL== 1e-8
        assert self.kernel.increase == 2.0
        assert self.kernel.decrease == 0.5

    def test_reset(self):
        """
        Test the method
        :meth:`bet.sampling.adaptiveSampling.multi_dist_kernel.reset`
        """
        self.kernel.reset()
        assert self.kernel.radius == None
        assert self.kernel.mean == None
        assert self.kernel.current_clength == 0

    def test_delta_step(self):
        """
        Test the delta_step method of
        :class:`bet.sampling.adaptiveSampling.multi_dist_kernel`
        """
        data_old = np.vstack([self.Q_ref+3.0, self.Q_ref, self.Q_ref-3.0])
        kern_old, proposal = self.kernel.delta_step(data_old)

        # TODO: check kern_old
        #nptest.assert_array_equal(kern_old, np.zeros((self.data.shape[0],))
        assert proposal == None 
        
        data_new = np.vstack([self.Q_ref, self.Q_ref+3.0, self.Q_ref-3.0])
        kern_new, proposal = self.kernel.delta_step(data_new, kern_old)

        #TODO: check kern_new
        #nptest.assert_array_eqyal(kern_new, something)
        nptest.assert_array_equal(proposal, [0.5, 2.0, 1.0])

        # TODO
        # check self.current_clength
        # check self.radius
        # check self.mean
 
class test_multi_dist_kernel_1D(multi_dist_kernel, data_1D):
    """
    Test :class:`bet.sampling.adaptiveSampling.multi_dist_kernel` on a 1D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_multi_dist_kernel_1D, self).createData()
        super(test_multi_dist_kernel_1D, self).setUp()
      
class test_multi_dist_kernel_2D(multi_dist_kernel, data_2D):
    """
    Test :class:`bet.sampling.adaptiveSampling.multi_dist_kernel` on a 2D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_multi_dist_kernel_2D, self).createData()
        super(test_multi_dist_kernel_2D, self).setUp()
      
class test_multi_dist_kernel_3D(multi_dist_kernel, data_3D):
    """
    Test :class:`bet.sampling.adaptiveSampling.multi_dist_kernel` on a 3D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_multi_dist_kernel_3D, self).createData()
        super(test_multi_dist_kernel_3D, self).setUp()



