"""
This module provides methods for plotting probabilities. 
"""
from bet.Comm import *
import numpy as np
import copy
import math
import scipy.io as sio


def sort_by_rho(P_samples, samples, lam_vol=None, data=None):
    """
    This sorts the samples by probability density. It returns the sorted values.
    If the samples are iid, no volume data is needed. It is optional to sort the QoI 
    data, but be sure to do so if using it later.

    :param P_samples: Probabilities.
    :type P_samples: :class:'~numpy.ndarray' of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:'~numpy.ndarray' of shape (num_samples, ndim)
    :param lam_vol: Volume of cell associated with sample.
    :type lam_vol: :class:'~numpy.ndarray' of shape (num_samples,)
    :param data: QoI data from running the model with the given samples.
    :type data: :class:'~numpy.ndarray' of shape (num_samples, mdim)
    :rtype: tuple
    :returns: (P_samples, samples, lam_vol, data)

    """
    if lam_vol == None:
        indices = np.argsort(P_samples)[::-1]
    else:
        indices = np.argsort(P_samples/lam_vol)[::-1]
    P_samples = P_samples[indices]
    samples = samples[indices,:]
    if lam_vol != None:
        lam_vol = lam_vol[indices]
    if data != None:
        data = data[indices]

    return (P_samples, samples, lam_vol, data)

def sample_highest_prob(top_percentile, P_samples, samples, lam_vol=None, data=None, sort=True):
    """
    This calculates the highest probability samples whose probability sum to a given value. 
    The number of high probability samples that sum to the value and the probabilities, 
    samples, volumes, and data are returned. This assumes that ``P_samples``, ``samples``, 
    ``lam_vol``, and ``data`` have all be sorted using :meth:`~bet.postProcess.sort_by_rho`.

    :param top_percentile: ratio of highest probability samples to select
    :type top_percentile: float
    :param P_samples: Probabilities.
    :type P_samples: :class:'~numpy.ndarray' of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:'~numpy.ndarray' of shape (num_samples, ndim)
    :param lam_vol: Volume of cell associated with sample.
    :type lam_vol: :class:'~numpy.ndarray' of shape (num_samples,)
    :param data: QoI data from running the model with the given samples.
    :type data: :class:'~numpy.ndarray' of shape (num_samples, mdim)
    :rtype: tuple
    :returns: ( num_samples, P_samples, samples, lam_vol, data)

    """

    if sort:
        (P_samples, samples, lam_vol, data) = sort_by_rho(P_samples, samples, lam_vol, data)

    P_sum = np.cumsum(P_samples)
    num_samples = np.sum(np.logical_and(0.0 < P_sum, P_sum <= top_percentile))
    P_samples = P_samples[0:num_samples]
    samples = samples[0:num_samples,:]
    if lam_vol != None:
        lam_vol = lam_vol[0:num_samples]
    if data != None:
        data = data[0:num_samples,:]
        
    return  (num_samples, P_samples, samples, lam_vol, data)
    

def save_parallel_probs_csv(P_samples,
                            samples,
                            P_file,
                            lam_file,
                            compress=False):
    """
    Saves probabilites and samples from parallel runs in individual .csv files for each process.

    :param P_samples: Probabilities.
    :type P_samples: :class:'~numpy.ndarray' of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:'~numpy.ndarray' of shape (num_samples, ndim)
    :param P_file: file prefix for probabilities
    :type P_file: str
    :param lam_file: file prefix for samples
    :type lam_file: str
    :param compress: Compress file
    :type compress: Boolean
    :returns: None
    """
    if compress:
        suffix = '.csv.gz'
    else:
        suffix = '.csv'

    np.savetxt(P_file + `rank` + suffix ,P_samples, delimiter = ',')
    np.savetxt(lam_file + `rank` + suffix ,samples, delimiter = ',')

def collect_parallel_probs_csv(P_file,
                               lam_file,
                               num_files,
                               save = False,
                               compress=False):
    """
    Collects probabilities and samples saved in .csv format from parallel runs into single arrays.

    :param P_file: file prefix for probabilities
    :type P_file: str
    :param lam_file: file prefix for samples
    :type lam_file: str
    :param num_files: number of files
    :type num_files: int
    :param save: Save collected arrays as a .csv file.
    :type save: Boolean
    :param compress: Compress file
    :type compress: Boolean
    :rtype: tuple 
    :returns (P, lam)
    """
    if compress:
        suffix = '.csv.gz'
    else:
        suffix = '.csv'

    P = np.loadtxt(P_file + '0' + suffix)
    lam = np.loadtxt(lam_file + '0' + suffix)
    for i in range(1,num_files):
        P = np.vstack((P, np.loadtxt(P_file + `i` + suffix)))
        lam = np.vstack((lam,np.loadtxt(lam_file + `i` + suffix)))

    if save:
        np.savetxt(P_file + 'all' + suffix, P)
        np.savetxt(lam_file + 'all' + suffix, lam)

    return (P, lam)

def save_parallel_probs_mat(P_samples,
                            samples,
                            file_prefix,
                            compress=False):
    """
    Saves probabilites and samples from parallel runs in individual .mat files for each process.

    :param P_samples: Probabilities.
    :type P_samples: :class:'~numpy.ndarray' of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:'~numpy.ndarray' of shape (num_samples, ndim)
    :param file_prefix: file prefix for probabilities
    :type file_prefix: str
    :returns: None
    """
    file_dict={"P_samples": P_samples,
               "samples": samples}
    sio.savemat(file_prefix + `rank` , file_dict, do_compression=compress)

def collect_parallel_probs_mat(file_prefix,
                               num_files,
                               save = False,
                               compress = False):
    """
    Collects probabilities and samples saved in .mat format from parallel runs into single arrays.

    :param file_prefix: file prefix 
    :type file_prefix: str
    :param num_files: number of files
    :type num_files: int
    :param save: Save collected arrays as a .mat file.
    :type save: Boolean
    :param compress: Compress file
    :type compress: Boolean
    :rtype: tuple 
    :returns (P, lam)
    """
    file_dict = sio.io.loadmat(file_prefix + "0")
    P = file_dict["P_samples"]
    lam = file_dict["samples"]
    for i in range(1,num_files):
        file_dict = sio.io.loadmat(file_prefix + `i`)
        P = np.vstack((P, file_dict["P_samples"]))
        lam = np.vstack((lam,file_dict["samples"]))

    if save:
        file_dict = {"P_samples": P,
               "samples": lam}
        sio.savemat(file_prefix + "all", file_dict, do_compression=compress)

    return (P, lam)
