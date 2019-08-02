# Copyright (C) 2014-2019 The BET Development Team

"""
This module contains unittests for :mod:`~bet.sampling.basicSampling:`
"""

import unittest
import os
import bet
import numpy.testing as nptest
import numpy as np
import scipy.io as sio
import bet.sampling.LpGeneralizedSamples as lp


def test_Lp_generalized_normal():
    """
    Tests :meth:`bet.Lp_generalized_samples.Lp_generalized_normal`

    This test only verifies the mean, but not the variance.

    """
    # 1D
    nptest.assert_allclose(np.mean(lp.Lp_generalized_normal(1, 1000), 0),
                           np.zeros((1,)), atol=1e-1)
    # 2D
    nptest.assert_allclose(np.mean(lp.Lp_generalized_normal(2, 1000), 0),
                           np.zeros((2,)), atol=1e-1)
    # 3D
    nptest.assert_allclose(np.mean(lp.Lp_generalized_normal(3, 1000), 0),
                           np.zeros((3,)), atol=1e-1)


def verify_norm_and_mean(x, r, p):
    """

    Verify that all of the samples in `x` are within the Lp ball centered at 0.
    Verify the mean of `x` is zero.

    :param x: Array containing a set of samples
    :type x: :class:`numpy.ndarry` of shape (num, dim)
    :param float r: radius of the Lp ball
    :param float p: 0 < p <= infinity, p of the Lp ball

    """
    if np.isinf(p):
        xpnorm = np.max(np.abs(x), 1)
    else:
        xpnorm = np.sum(np.abs(x)**p, 1)**(1. / p)
    assert np.all(xpnorm <= r)
    nptest.assert_allclose(np.mean(x, 0), np.zeros((x.shape[1],)), atol=1e-1)


def verify_norm(x, r, p):
    """

    Verify that all of the samples in `x` are within the Lp ball centered at 0.

    :param x: Array containing a set of samples
    :type x: :class:`numpy.ndarry` of shape (num, dim)
    :param float r: radius of the Lp ball
    :param float p: 0 < p <= infinity, p of the Lp ball

    """
    if np.isinf(p):
        xpnorm = np.max(np.abs(x), 1)
    else:
        xpnorm = np.sum(np.abs(x)**p, 1)**(1. / p)
    assert np.all(xpnorm <= r)


def test_Lp_generalized_uniform():
    """
    Tests :meth:`bet.Lp_generalized_samples.Lp_generalized_uniform`

    This test only verifies the mean, but not the variance.

    """
    # 1D
    x = lp.Lp_generalized_uniform(1, 1000)
    nptest.assert_allclose(np.mean(x, 0), np.zeros((1,)), atol=1e-1)
    assert np.all(np.logical_and(x <= 1., x >= -1))

    # 2D
    p = 1
    x = lp.Lp_generalized_uniform(2, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    p = 2
    x = lp.Lp_generalized_uniform(2, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    p = 3
    x = lp.Lp_generalized_uniform(2, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    p = np.inf
    x = lp.Lp_generalized_uniform(2, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    # 3D
    p = 1
    x = lp.Lp_generalized_uniform(3, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    p = 2
    x = lp.Lp_generalized_uniform(3, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    p = 3
    x = lp.Lp_generalized_uniform(3, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    p = np.inf
    x = lp.Lp_generalized_uniform(3, 1000, p)
    verify_norm_and_mean(x, 1.0, p)


def test_Lp_generalized_beta():
    """
    Tests :meth:`bet.Lp_generalized_samples.Lp_generalized_beta`

    This test only verifies the mean, but not the variance.

    """
    # 1D
    x = lp.Lp_generalized_beta(1, 1000)
    nptest.assert_allclose(np.mean(x, 0), np.zeros((1,)), atol=1e-1)
    assert np.all(np.logical_and(x <= 1., x >= -1))

    # 2D
    p = 1
    x = lp.Lp_generalized_beta(2, 1000, p)
    verify_norm(x, 1.0, p)

    p = 2
    x = lp.Lp_generalized_beta(2, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    p = 3
    x = lp.Lp_generalized_beta(2, 1000, p)
    verify_norm(x, 1.0, p)

    # 3D
    p = 1
    x = lp.Lp_generalized_beta(3, 1000, p)
    verify_norm(x, 1.0, p)

    p = 2
    x = lp.Lp_generalized_beta(3, 1000, p)
    verify_norm_and_mean(x, 1.0, p)

    p = 3
    x = lp.Lp_generalized_beta(3, 1000, p)
    verify_norm(x, 1.0, p)
