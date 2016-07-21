# Copyright (C) 2014-2016 The BET Development Team

"""
This module provides methods for postprocessing probabilities and data. 
"""
from bet.Comm import comm
import numpy as np
import scipy.io as sio
import bet.sample as sample
import logging

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
        indices = np.argsort(P_samples/lam_vol)[::-1][0:nnz]
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

    return  (num_samples, sample_set_out,
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

def save_parallel_probs_csv(P_samples, samples, P_file, lam_file,
        compress=False):
    """
    .. todo::

       Revisit when save features in sample.py are stable

    Saves probabilites and samples from parallel runs in individual ``.csv``
    files for each process.

    :param P_samples: Probabilities.
    :type P_samples: :class:`~numpy.ndarray` of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param P_file: file prefix for probabilities
    :type P_file: str
    :param lam_file: file prefix for samples
    :type lam_file: str
    :param compress: Compress file
    :type compress: bool
    
    :returns: None
    
    """
    if compress:
        suffix = '.csv.gz'
    else:
        suffix = '.csv'

    np.savetxt(P_file + str(comm.rank) + suffix, P_samples, delimiter=',')
    np.savetxt(lam_file + str(comm.rank) + suffix, samples, delimiter=',')

def collect_parallel_probs_csv(P_file, lam_file, num_files, save=False,
        compress=False):
    """
    .. todo::

       Revisit when save features in sample.py are stable

    Collects probabilities and samples saved in ``.csv`` format from parallel
    runs into single arrays.

    :param P_file: file prefix for probabilities
    :type P_file: str
    :param lam_file: file prefix for samples
    :type lam_file: str
    :param num_files: number of files
    :type num_files: int
    :param save: Save collected arrays as a ``.csv`` file.
    :type save: bool
    :param compress: Compress file
    :type compress: bool
    
    :rtype: tuple 
    :returns: (P, lam)
    
    """
    if compress:
        suffix = '.csv.gz'
    else:
        suffix = '.csv'

    P = np.loadtxt(P_file + '0' + suffix)
    lam = np.loadtxt(lam_file + '0' + suffix)
    for i in range(1, num_files):
        P = np.vstack((P, np.loadtxt(P_file + str(i) + suffix)))
        lam = np.vstack((lam, np.loadtxt(lam_file + str(i) + suffix)))

    if save:
        np.savetxt(P_file + 'all' + suffix, P)
        np.savetxt(lam_file + 'all' + suffix, lam)

    return (P, lam)

def save_parallel_probs_mat(P_samples, samples, file_prefix, compress=False):
    """
    .. todo::

       Revisit when save features in sample.py are stable

    Saves probabilites and samples from parallel runs in individual .mat files
    for each process.

    :param P_samples: Probabilities.
    :type P_samples: :class:`~numpy.ndarray` of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param file_prefix: file prefix for probabilities
    :type file_prefix: str
    
    :returns: None
    
    """
    file_dict = {"P_samples": P_samples,
               "samples": samples}
    sio.savemat(file_prefix + str(comm.rank), file_dict,
                do_compression=compress)

def collect_parallel_probs_mat(file_prefix, num_files, save=False,
       compress=False):
    """
    .. todo::

       Revisit when save features in sample.py are stable

    Collects probabilities and samples saved in .mat format from parallel runs
    into single arrays.

    :param file_prefix: file prefix 
    :type file_prefix: str
    :param num_files: number of files
    :type num_files: int
    :param save: Save collected arrays as a .mat file.
    :type save: bool
    :param compress: Compress file
    :type compress: bool
    
    :rtype: tuple 
    :returns: (P, lam)
    
    """
    file_dict = sio.io.loadmat(file_prefix + "0")
    P = file_dict["P_samples"]
    lam = file_dict["samples"]
    for i in range(1, num_files):
        file_dict = sio.io.loadmat(file_prefix + str(i))
        P = np.vstack((P, file_dict["P_samples"]))
        lam = np.vstack((lam, file_dict["samples"]))

    if save:
        file_dict = {"P_samples": P,
               "samples": lam}
        sio.savemat(file_prefix + "all", file_dict, do_compression=compress)

    return (P, lam)

def compare_yield(sort_ind, sample_quality, run_param, column_headings=None):
    """
    .. todo::

       Revisit to deprecate later.

    Compare the quality of samples where ``sample_quality`` is the measure of
    quality by which the sets of samples have been indexed and ``sort_ind`` is
    an array of the sorted indices.

    :param list sort_ind: indices that index ``sample_quality`` in sorted
        order
    :param list sample_quality: a measure of quality by which the sets of 
        samples are sorted
    :param list run_param: zipped list of :class:`~numpy.ndarray` containing
        information used to generate the sets of samples to be displayed
    :param list column_headings: Column headings to print to screen

    """
    if column_headings == None:
        column_headings = "Run parameters"
    logging.info("Sample Set No., Quality, "+ str(column_headings))
    for i in reversed(sort_ind):
        logging.info(i, sample_quality[i], np.round(run_param[i], 3))

def in_high_prob(data, rho_D, maximum, sample_nos=None):
    """
    .. todo::

       Revisit to deprecate later.

    Estimates the number of samples in high probability regions of D.

    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`np.ndarray`
    :param float maximum: maximum (or average) value of ``rho_D``
    :param list sample_nos: sample numbers to plot

    :rtype: int
    :returns: Estimate of number of samples in the high probability area.

    """
    if sample_nos is None:
        sample_nos = range(data.shape[0])
    if len(data.shape) == 1:
        rD = rho_D(data[sample_nos])
    else:
        rD = rho_D(data[sample_nos, :])
    adjusted_total_prob = int(sum(rD)/maximum)
    logging.info("Samples in box "+str(adjusted_total_prob))
    return adjusted_total_prob

def in_high_prob_multi(results_list, rho_D, maximum, sample_nos_list=None):
    """
    .. todo::

       Revisit to deprecate later.

    Estimates the number of samples in high probability regions of D for a list
    of results.

    :param list results_list: list of (results, data) tuples
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`np.ndarray`
    :param float maximum: maximum (or average) value of ``rho_D``
    :param list sample_nos_list: list of sample numbers to plot (list of lists)

    :rtype: list of int
    :returns: Estimate of number of samples in the high probability area.

    """
    adjusted_total_prob = list()
    if sample_nos_list:
        for result, sample_nos in zip(results_list, sample_nos_list):
            adjusted_total_prob.append(in_high_prob(result[1], rho_D, maximum,
                sample_nos))
    else:
        for result in results_list:
            adjusted_total_prob.append(in_high_prob(result[1], rho_D, maximum))
    return adjusted_total_prob


       
