# Lindley Graham 04/09/2015

"""
This module contains tests for :module:`bet.calculateP.simpleFunP`

Some of these tests make sure certain values are within a tolerance rather than
exact due to the stochastic nature of the algorithms being tested. 

The ouput of all the methods being tested is of the form (rho_D_M,
d_distr_samples, d_Tree) where ``rho_D_M`` and ``d_distr_samples`` are (mdim,
M) :class:`~numpy.ndarray` and `d_Tree` is the :class:`~scipy.spatial.KDTree`
for d_distr_samples.

"""

import os, bet, unittest
import bet.calculateP.simpleFunP as simpleFunP
import numpy as np
import numpy.testing as nptest

local_path = os.path.join(os.path.dirname(bet.__file__),
'../test/test_calulateP')

class prob:
    def test_rho_D_M_sum_to_1(self):
        """
        Test that probabilities sum to 1.
        """
        nptest.assert_almost_equal(np.sum(self.rho_D_M), 1.0)
    def test_rho_D_M_pos(self):
        """
        Test that probabilities are non-negative.
        """
        assert True == np.all(self.rho_D_M >= 0.0)
    def test_dimensions(self):
        """
        Test that the dimensions of the outputs are correct.
        """
        assert self.rho_D_M.shape[0] == self.d_distr_samples.shape[1]
        assert self.data.shape[1] == self.d_distr_samples.shape[0]
        assert (self.d_tree.n, self.d_tree.m) == d_distr_samples.shape

class prob_uniform:
    def test_domain(self):
        """
        Test that the probabilities within the prescribed domain are non-zero
        and that the probabilities outside of the prescribed domain are zero.
        """
        # d_distr_samples are (mdim, M)




