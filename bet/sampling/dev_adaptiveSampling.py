# Copyright (C) 2014-2015 The BET Development Team
# Lindley Graham 4/21/15
""" 

This module contains functions for adaptive random sampling with reseeding.
We assume we are given access to a model, a parameter space, and a data space.
The model is a map from the paramter space to the data space. We desire to
build up a set of samples to solve an inverse problem thus giving us
information about the inverse mapping. Each sample consists of a parameter
coordinate, data coordinate pairing. We assume the measure of both spaces is
Lebesgue.

We employ an approach based on using multiple sample chains that restarts the
chains periodically throughout the sampling process.

"""

import numpy as np
import scipy.io as sio
import bet.sampling.adaptiveSampling as asam
import math, os
from bet.Comm import *

class sampler(asam.sampler):
    """
    This class provides methods for adaptive sampling of parameter space to
    provide samples to be used by algorithms to solve inverse problems. 
    
    chain_length
        number of batches of samples
    num_chains
        number of samples per batch (either a single int or a list of int)
    lb_model
        :class:`~bet.loadBalance.load_balance` runs the model at a given set of
        parameter samples and returns data """
    def __init__(self, num_samples, chain_length, lb_model):
        """
        Initialization
        """
        super(sampler, self).__init__(lb_model, num_samples)
        self.chain_length = chain_length
        self.num_chains_pproc = int(math.ceil(num_samples/float(chain_length*size)))
        self.num_chains = size * self.num_chains_pproc
        self.num_samples = chain_length * self.num_chains
        self.lb_model = lb_model
        self.sample_batch_no = np.repeat(range(self.num_chains), chain_length,
                0)

    def run_reseed(self, kern_list, rho_D, maximum, param_min, param_max,
            t_set, savefile, initial_sample_type="lhs", criterion='center',
            reseed=3):
        """
        Generates samples using reseeded chains and a list of different
        kernels.

        THIS IS NOT OPERATIONAL DO NOT USE.

        :param list() kern_list: List of
            :class:~`bet.sampling.adaptiveSampling.kernel` objects.
        :param rho_D: probability density on D
        :type rho_D: callable function that takes a :class:`np.array` and
            returns a :class:`numpy.ndarray`
        :param double maximum: maximum value of rho_D
        :param param_min: minimum value for each parameter dimension
        :type param_min: np.array (ndim,)
        :param param_max: maximum value for each parameter dimension
        :type param_max: np.array (ndim,)
        :param t_set: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_set: :class:~`bet.sampling.adaptiveSampling.transition_set`
        :param string savefile: filename to save samples and data
        :param string initial_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
         :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        :rtype: tuple
        :returns: ((samples, data), all_step_ratios, num_high_prob_samples,
            sorted_incidices_of_num_high_prob_samples, average_step_ratio)

        """
        results = list()
        # reseeding sampling
        results = list()
        r_step_size = list()
        results_rD = list()
        mean_ss = list()
        for kern in kern_list:
            (samples, data, step_sizes) = self.reseed_chains(
                    param_min, param_max, t_set, kern, savefile,
                    initial_sample_type, criterion, reseed)
            results.append((samples, data))
            r_step_size.append(step_sizes)
            results_rD.append(int(sum(rho_D(data)/maximum)))
            mean_ss.append(np.mean(step_sizes))
        sort_ind = np.argsort(results_rD)
        return (results, r_step_size, results_rD, sort_ind, mean_ss)

    def reseed_chains(self, param_min, param_max, t_set, kern,
            savefile, initial_sample_type="lhs", criterion='center', reseed=1):
        """
        Basic adaptive sampling algorithm.

        NOT YET IMPLEMENTED.

        :param string initial_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param param_min: minimum value for each parameter dimension
        :type param_min: np.array (ndim,)
        :param param_max: maximum value for each parameter dimension
        :type param_max: np.array (ndim,)
        :param t_set: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_set: :class:~`bet.sampling.adaptiveSampling.transition_set`
        :param function kern: functional that acts on the data used to
            determine the proposed change to the ``step_size``
        :type kernel: :class:~`bet.sampling.adaptiveSampling.kernel` object.
        :param string savefile: filename to save samples and data
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_


        :param int reseed: number of times to reseed the chains
        :rtype: tuple
        :returns: (``parameter_samples``, ``data_samples``) where
            ``parameter_samples`` is np.ndarray of shape (num_samples, ndim)
            and ``data_samples`` is np.ndarray of shape (num_samples, mdim)

        """
        pass
