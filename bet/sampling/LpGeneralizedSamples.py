# Copyright (C) 2016 The BET Development Team

# Lindley Graham 05/19/2016

"""

This module provides methods to sample from Lp generalized normal, uniform, and
beta distributions on the nD ball.

Adapted from natter.LpSphericallySymmetric.py
https://github.com/fabiansinz/natter

"""

import numpy as np

def Lp_generalized_normal(dim, num, p=2, scale=1.0, loc=None):
    r"""
    
    Generate samples from an Lp generalized normal distribution.

    :param float p: :math:`0 < p \leq \infty`, p for the lp norm where
        infinity is ``numpy.inf``
    :param int dim: Dimension of the space
    :param int num: Number of samples to generate
    :param scale: Radius to scale the samples by
    :type scale: ``float``, ``int``, or :class:`numpy.ndarray` 
    :param loc: Location of the center of the samples
    :type loc: :class:`numpy.ndarray` of shape (dim,)

    """
    num = int(num)
    dim = int(dim)
    p = float(p)
    z = np.random.gamma(1./p, scale=scale, size=(num, dim))
    z = np.abs(z)**(1./p)
    samples = z * np.sign(np.random.randn(num, dim))
    if loc is not None:
        samples = samples + loc
    return samples

def Lp_generalized_uniform(dim, num, p=2, scale=1.0, loc=None):
    r"""
    
    Generate samples from an Lp generalized uniform distribution.

    :param float p: :math:`0 < p \leq \infty`, p for the lp norm where
        infinity is ``numpy.inf``
    :param int dim: Dimension of the space
    :param int num: Number of samples to generate
    :param scale: Radius to scale the samples by
    :type scale: ``float``, ``int``, or :class:`numpy.ndarray`
    :param loc: Location of the center of the samples
    :type loc: :class:`numpy.ndarray` of shape (dim,)

    """
    num = int(num)
    dim = int(dim)
    if not np.isinf(p):
        p = float(p)
        # sample from a p-generalized normal with scale 1
        samples = Lp_generalized_normal(dim, num, p)
        samples_norm = np.sum(np.abs(samples)**p, axis=1)**(1./p)
        samples = samples/np.reshape(samples_norm, (num, 1))
        r = np.random.beta(a=dim, b=1., size=(num, 1))
        samples = samples * r * scale
    else:
        samples = (np.random.random((num, dim))-.5)*2.0 * scale
    if loc is not None:
        samples = samples + loc
    return samples
    
def Lp_generalized_beta(dim, num, p=2, d=2, scale=1.0, loc=None):
    r"""
    
    Generate samples from an Lp generalized beta distribution. When p=d then
    this is simly the Lp generalized uniform distribution.

    :param float p: :math:`0 < p \leq \infty`, p for the lp norm where
        infinity is ``numpy.inf``    
    :param float d: shape parameter
    :param int dim: Dimension of the space
    :param int num: Number of samples to generate
    :param scale: Radius to scale the samples by
    :type scale: ``float``, ``int``, or :class:`numpy.ndarray`
    :param loc: Location of the center of the samples
    :type loc: :class:`numpy.ndarray` of shape (dim,)

    """
    num = int(num)
    dim = int(dim)
    p = float(p)
    # sample from a p-generalized normal with scale 1
    samples = Lp_generalized_normal(dim, num, p)
    samples_norm = np.sum(np.abs(samples)**p, axis=1)**(1./p)
    samples = samples/np.reshape(samples_norm, (num, 1))
    r = np.random.beta(a=dim/p, b=d/p, size=(num, 1))**(1./p)
    samples = samples * r * scale
    if loc is not None:
        samples = samples + loc
    return samples
