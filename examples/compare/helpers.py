# Copyright (C) 2014-2020 The BET Development Team

import bet.sample as sample
import bet.sampling.basicSampling as bsam
import numpy as np


def unit_center_set(dim=1, num_samples=100,
                    delta=1, reg=False):
    r"""
    First define a unit hyper-box sample set. Then, construct a hyper-box
    centered at ``np.array([[0.5]]*dim)`` with side-lengths ``delta``
    of this domain that has probability 1.
    (Useful for testing).

    :param int dim: dimension
    :param int num_samples: number of samples
    :param float delta: sidelength of region with positive probability
    :param bool reg: regular sampling (`num_samples` = per dimension)
    :rtype: :class:`bet.sample.sample_set`
    :returns: sample set object

    """
    s_set = sample.sample_set(dim)
    s_set.set_domain(np.array([[0, 1]] * dim))
    if reg:
        s = bsam.regular_sample_set(s_set, num_samples)
    else:
        s = bsam.random_sample_set('r', s_set, num_samples)
    dd = delta / 2.0
    if dim > 1:
        probs = 1 * (np.sum(np.logical_and(s._values <= (0.5 + dd),
                                           s._values >= (0.5 - dd)), axis=1)
                     >= dim)
    else:
        probs = 1 * (np.logical_and(s._values <= (0.5 + dd),
                                    s._values >= (0.5 - dd)))
    s.set_probabilities(probs / np.sum(probs))  # uniform probabilities
    s.set_prob_type('voronoi')
    s.estimate_volume_mc()
    s.global_to_local()
    return s


def unit_bottom_set(dim=1, num_samples=100,
                    delta=1, reg=False):
    r"""
    First define a unit hyper-box sample set. Then, construct a hyper-box
    with a corner at ``np.array([[0.0]]*dim)`` with side-lengths ``delta``
    inside of this domain that has probability 1.
    (Useful for testing).

    :param int dim: dimension
    :param int num_samples: number of samples
    :param float delta: sidelength of region with positive probability
    :param bool reg: regular sampling (`num_samples` = per dimension)
    :rtype: :class:`bet.sample.sample_set`
    :returns: sample set object

    """
    s_set = sample.sample_set(dim)
    s_set.set_domain(np.array([[0, 1]] * dim))
    if reg:
        s = bsam.regular_sample_set(s_set, num_samples)
    else:
        s = bsam.random_sample_set('r', s_set, num_samples)
    dd = delta
    if dim == 1:
        probs = 1 * (s._values <= dd)
    else:
        probs = 1 * (np.sum(s._values <= dd, axis=1) >= dim)
    s.set_probabilities(probs / np.sum(probs))  # uniform probabilities
    s.set_prob_type('voronoi')
    s.estimate_volume_mc()
    s.global_to_local()
    return s


def unit_top_set(dim=1, num_samples=100,
                 delta=1, reg=False):
    r"""
    First define a unit hyper-box sample set. Then, construct a hyper-box
    with a corner at ``np.array([[1.0]]*dim)`` with side-lengths ``delta``
    inside of this domain that has probability 1.
    (Useful for testing).

    :param int dim: dimension
    :param int num_samples: number of samples
    :param float delta: sidelength of region with positive probability
    :param bool reg: regular sampling (`num_samples` = per dimension)
    :rtype: :class:`bet.sample.sample_set`
    :returns: sample set object

    """
    s_set = sample.sample_set(dim)
    s_set.set_domain(np.array([[0, 1]] * dim))
    if reg:
        s = bsam.regular_sample_set(s_set, num_samples)
    else:
        s = bsam.random_sample_set('r', s_set, num_samples)

    dd = delta
    if dim == 1:
        probs = 1 * (s._values >= (1 - dd))
    else:
        probs = 1 * (np.sum(s._values >= (1 - dd), axis=1) >= dim)
    s.set_probabilities(probs / np.sum(probs))  # uniform probabilities
    s.set_prob_type('voronoi')
    s.estimate_volume_mc()
    s.global_to_local()
    return s
