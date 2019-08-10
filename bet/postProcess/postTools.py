# Copyright (C) 2014-2019 The BET Development Team

"""
This module provides methods for postprocessing probabilities and data.
"""
import logging
import numpy as np
import bet.sample as sample


class dim_not_matching(Exception):
    """
    Exception for when the dimension is inconsistent.
    """


class bad_object(Exception):
    """
    Exception for when the wrong type of object is used.
    """


def sort_by_rho(sample_set):
    """
    This sorts the samples within the sample_set by probability density.
    If a discretization object is given, then the QoI data is also sorted
    to maintain the correspondence.
    Any volumes present in the input space (or just the sample object)
    are also sorted.

    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base` or
        :class:`~bet.sample.discretization`
    :param indices: sorting indices
    :type indices: :class:`numpy.ndarray` of shape (num_samples,)
    :param sample_set_out: Object containing sorted samples and probabilities
    :type sample_set_out: :class:`~bet.sample.sample_set` or
        :class:`~bet.sample.discretization`

    :rtype: tuple
    :returns: (sample_set_out, indicices)

    """
    if isinstance(sample_set, sample.discretization):
        samples = sample_set._input_sample_set.get_values()
        P_samples = sample_set._input_sample_set.get_probabilities()
        lam_vol = sample_set._input_sample_set.get_volumes()
        data = sample_set._output_sample_set.get_values()
    elif isinstance(sample_set, sample.sample_set_base):
        samples = sample_set.get_values()
        P_samples = sample_set.get_probabilities()
        lam_vol = sample_set.get_volumes()
        data = None
    else:
        raise bad_object("Improper sample object")

    nnz = np.sum(P_samples > 0)
    if lam_vol is None:
        indices = np.argsort(P_samples)[::-1][0:nnz]
    else:
        indices = np.argsort(P_samples / lam_vol)[::-1][0:nnz]
    P_samples = P_samples[indices]
    samples = samples[indices, :]
    if lam_vol is not None:
        lam_vol = lam_vol[indices]
    if data is not None:
        data = data[indices, :]

    if isinstance(sample_set, sample.discretization):
        samples_out = sample.sample_set(sample_set._input_sample_set.get_dim())
        data_out = sample.sample_set(sample_set._output_sample_set.get_dim())
        sample_set_out = sample.discretization(samples_out, data_out)
        sample_set_out._input_sample_set.set_values(samples)
        sample_set_out._input_sample_set.set_probabilities(P_samples)
        sample_set_out._input_sample_set.set_volumes(lam_vol)
        sample_set_out._output_sample_set.set_values(data)
    else:
        sample_set_out = sample.sample_set(sample_set.get_dim())
        sample_set_out.set_values(samples)
        sample_set_out.set_probabilities(P_samples)
        sample_set_out.set_volumes(lam_vol)

    return (sample_set_out, indices)


def sample_prob(percentile, sample_set, sort=True, descending=False):
    """
    This calculates the highest/lowest probability samples whose probability
    sum to a given value.
    A new sample_set with the samples corresponding to these highest/lowest
    probability samples is returned along with the number of samples and
    the indices.
    This uses :meth:`~bet.postProcess.sort_by_rho`.
    The ``descending`` flag determines whether or not to calcuate the
    highest/lowest.

    :param percentile: ratio of highest probability samples to select
    :type percentile: float
    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base` or
        :class:`~bet.sample.discretization`
    :type indices: :class:`numpy.ndarray` of shape (num_samples,)
    :param indices: sorting indices
    :param bool sort: Flag whether or not to sort
    :param bool descending: Flag order of sorting
    :param sample_set_out: Object containing sorted samples and probabilities
    :type sample_set_out: :class:`~bet.sample.sample_set` or
        :class:`~bet.sample.discretization`

    :rtype: tuple
    :returns: ( num_samples, sample_set_out, data)

    """
    if isinstance(sample_set, sample.discretization):
        samples = sample_set._input_sample_set.get_values()
        P_samples = sample_set._input_sample_set.get_probabilities()
        lam_vol = sample_set._input_sample_set.get_volumes()
        data = sample_set._output_sample_set.get_values()
    elif isinstance(sample_set, sample.sample_set_base):
        samples = sample_set.get_values()
        P_samples = sample_set.get_probabilities()
        lam_vol = sample_set.get_volumes()
        data = None
    else:
        raise bad_object("Improper sample object")

    if sort:
        (sample_set, indices) = sort_by_rho(sample_set)
        if isinstance(sample_set, sample.discretization):
            samples = sample_set._input_sample_set.get_values()
            P_samples = sample_set._input_sample_set.get_probabilities()
            lam_vol = sample_set._input_sample_set.get_volumes()
            data = sample_set._output_sample_set.get_values()
        elif isinstance(sample_set, sample.sample_set_base):
            samples = sample_set.get_values()
            P_samples = sample_set.get_probabilities()
            lam_vol = sample_set.get_volumes()
            data = None
    if descending:
        P_samples = P_samples[::-1]
        samples = samples[::-1]
        if lam_vol is not None:
            lam_vol = lam_vol[::-1]
        if data is not None:
            data = data[::-1]
        indices = indices[::-1]

    P_sum = np.cumsum(P_samples)
    num_samples = np.sum(np.logical_and(0.0 < P_sum, P_sum <= percentile))
    P_samples = P_samples[0:num_samples]
    samples = samples[0:num_samples, :]
    if lam_vol is not None:
        lam_vol = lam_vol[0:num_samples]
    if data is not None:
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)
        data = data[0:num_samples, :]

    if isinstance(sample_set, sample.discretization):
        samples_out = sample.sample_set(sample_set._input_sample_set.get_dim())
        data_out = sample.sample_set(sample_set._output_sample_set.get_dim())
        sample_set_out = sample.discretization(samples_out, data_out)
        sample_set_out._input_sample_set.set_values(samples)
        sample_set_out._input_sample_set.set_probabilities(P_samples)
        sample_set_out._input_sample_set.set_volumes(lam_vol)
        sample_set_out._output_sample_set.set_values(data)
    else:
        sample_set_out = sample.sample_set(sample_set.get_dim())
        sample_set_out.set_values(samples)
        sample_set_out.set_probabilities(P_samples)
        sample_set_out.set_volumes(lam_vol)

    return (num_samples, sample_set_out,
            indices[0:num_samples])


def sample_highest_prob(top_percentile, sample_set, sort=True):
    """
    This calculates the highest probability samples whose probability sum to a
    given value.
    The number of high probability samples that sum to the value,
    a new sample_set, and the indices are returned.
    This uses :meth:`~bet.postProcess.sort_by_rho`.

    :param top_percentile: ratio of highest probability samples to select
    :type top_percentile: float
    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base`
        or :class:`~bet.sample.discretization`
    :type indices: :class:`numpy.ndarray` of shape (num_samples,)
    :param indices: sorting indices
    :param bool sort: Flag whether or not to sort
    :param sample_set_out: Object containing sorted samples and probabilities
    :type sample_set_out: :class:`~bet.sample.sample_set`
        or :class:`~bet.sample.discretization`

    :rtype: tuple
    :returns: ( num_samples, sample_set_out, indices)

    """
    return sample_prob(top_percentile, sample_set, sort)


def sample_lowest_prob(bottom_percentile, sample_set, sort=True):
    """
    This calculates the lowest probability samples whose probability sum to a
    given value.
    The number of low probability samples that sum to the value,
    a new sample_set, and the indices are returned.
    This uses :meth:`~bet.postProcess.sort_by_rho`.

    :param top_percentile: ratio of highest probability samples to select
    :type top_percentile: float
    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base`
        or :class:`~bet.sample.discretization`
    :type indices: :class:`numpy.ndarray` of shape (num_samples,)
    :param indices: sorting indices of unsorted ``P_samples``
    :param bool sort: Flag whether or not to sort
    :param sample_set_out: Object containing sorted samples and probabilities
    :type sample_set_out: :class:`~bet.sample.sample_set`
        or :class:`~bet.sample.discretization`

    :rtype: tuple
    :returns: ( num_samples, sample_set_out, indices)

    """
    return sample_prob(bottom_percentile, sample_set,
                       sort, descending=True)

