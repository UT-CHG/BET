# -*- coding: utf-8 -*-
# Copyright (C) 2014-2015 The BET Development Team
# Lindley Graham 6/13/15
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
import bet.sampling.adaptiveSampling as asam
import scipy.spatial as spatial
import bet.util as util
import os
from bet.Comm import comm, MPI

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

        :param int num_samples: Total number of samples
        :param int chain_length: Number of samples per chain
        :param lb_model: runs the model at a given set of parameter samples, (N,
            ndim), and returns data (N, mdim)
        """
        super(sampler, self).__init__(num_samples, chain_length, lb_model)

    def run_reseed(self, kern_list, rho_D, maximum, param_min, param_max,
            t_set, savefile, initial_sample_type="lhs", criterion='center',
            reseed=3):
        """
        Generates samples using reseeded chains and a list of different
        kernels.

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
            (samples, data, step_sizes) = self.generalized_chains(
                    param_min, param_max, t_set, kern, savefile,
                    initial_sample_type, criterion, reseed)
            results.append((samples, data))
            r_step_size.append(step_sizes)
            results_rD.append(int(sum(rho_D(data)/maximum)))
            mean_ss.append(np.mean(step_sizes))
        sort_ind = np.argsort(results_rD)
        return (results, r_step_size, results_rD, sort_ind, mean_ss)

    def generalized_chains(self, param_min, param_max, t_set, rho_D,
            smoothIndicatorFun, savefile, initial_sample_type="lhs",
            criterion='center', reseed=1):
        r"""
        This method adaptively generates samples similar to the method
        :meth:`bet.sampling.adaptiveSampling.generalized_chains`. Adaptive
        sampling algorithm using generalized chains with limited memory. If a
        new sample is generated that is outside of the RoI then the next sample
        is generated from the last sample that was inside the RoI. After every
        ``reseed`` batches chains are restarted from ``num_chains`` best
        previous samples. The size of the hyperrectanlge from which new samples
        are drawn is based on the mean minimum pairwise distance between the
        current batch of samples.
       
        :param string initial_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param param_min: minimum value for each parameter dimension
        :type param_min: np.array (ndim,)
        :param param_max: maximum value for each parameter dimension
        :type param_max: np.array (ndim,)
        :param t_set: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_set: :class:~`bet.sampling.adaptiveSampling.transition_set`
        :param rho_D: :math:`\rho_\mathcal{D}` or :math:`\mathbf{1}_{A}` 
        :type rho_D: callable
        :param smoothIndicatorFun: A smoothed version of :math:`\mathbf{1}_A`
            where the maximum function values are on the boundary of
            :math:`\mathcal{D}` and the minimum function values are either in
            the RoI :math:`A` or in a neighborhood of :math:`\partial A`. The
            value of the function in :math:`A` must be strictly less than the
            value of the function outside of :math:`A+\Delta A`.
        :type smoothedIndicatorFun: callable
        :param string savefile: filename to save samples and data
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        :param int reseed: reseed every n batches
        :rtype: tuple
        :returns: (``parameter_samples``, ``data_samples``,
            ``all_step_ratios``) where ``parameter_samples`` is np.ndarray of
            shape (num_samples, ndim), ``data_samples`` is np.ndarray of shape
            (num_samples, mdim), and ``all_step_ratios`` is np.ndarray of shape
            (num_chains, chain_length)

        """
        if comm.size > 1:
            psavefile = os.path.join(os.path.dirname(savefile),
                    "proc{}{}".format(comm.rank, os.path.basename(savefile)))

        # Initialize Nx1 vector Step_size = something reasonable (based on size
        # of domain and transition set type)
        # Calculate domain size
        param_dist = np.sqrt(np.sum((param_max-param_min)**2))
        param_left = np.repeat([param_min], self.num_chains_pproc, 0)
        param_right = np.repeat([param_max], self.num_chains_pproc, 0)
        param_width = param_right - param_left
        # Calculate step_size
        max_ratio = t_set.max_ratio
        min_ratio = t_set.min_ratio
       
        # Initiative first batch of N samples (maybe taken from latin
        # hypercube/space-filling curve to fully explore parameter space - not
        # necessarily random). Call these Samples_old.
        (samples_old, data_old) = super(sampler, self).random_samples(
                initial_sample_type, param_min, param_max, savefile,
                self.num_chains, criterion)
        self.num_samples = self.chain_length * self.num_chains
        comm.Barrier()
        
        # now split it all up
        if comm.size > 1:
            MYsamples_old = np.empty((np.shape(samples_old)[0]/comm.size,
                np.shape(samples_old)[1])) 
            comm.Scatter([samples_old, MPI.DOUBLE], [MYsamples_old, MPI.DOUBLE])
            MYdata_old = np.empty((np.shape(data_old)[0]/comm.size,
                np.shape(data_old)[1])) 
            comm.Scatter([data_old, MPI.DOUBLE], [MYdata_old, MPI.DOUBLE])
        else:
            MYsamples_old = np.copy(samples_old)
            MYdata_old = np.copy(data_old)
        step_ratio = determine_step_ratio(param_dist, MYsamples_old)


        samples = MYsamples_old
        data = MYdata_old
        all_step_ratios = step_ratio
        #(kern_old, proposal) = kern.delta_step(MYdata_old, None)
        kern_old = rho_D(MYdata_old)
        kern_samples = kern_old
        mdat = dict()
        self.update_mdict(mdat)

        
        for batch in xrange(1, self.chain_length):
        
            # Reseed the samples
            if batch%reseed == 0:
                # This could be done more efficiently
                global_samples = util.get_global_values(np.copy(samples))
                sample_rank = smoothIndicatorFun(global_samples)
                sort_ind = np.argsort(sample_rank)[:self.num_chains]
                samples_old = global_samples[sort_ind, :]
                if comm.size > 1:
                    MYsamples_old = np.empty((np.shape(samples_old)[0]/comm.size,
                        np.shape(samples_old)[1])) 
                    comm.Scatter([samples_old, MPI.DOUBLE], [MYsamples_old, MPI.DOUBLE])
                    MYdata_old = np.empty((np.shape(data_old)[0]/comm.size,
                        np.shape(data_old)[1])) 
                    comm.Scatter([data_old, MPI.DOUBLE], [MYdata_old, MPI.DOUBLE])
                else:
                    MYsamples_old = np.copy(samples_old)
                    MYdata_old = np.copy(data_old)
                step_ratio = determine_step_ratio(param_dist, MYsamples_old)
        
            # For each of N samples_old, create N new parameter samples using
            # transition set and step_ratio. Call these samples samples_new.
            samples_new = t_set.step(step_ratio, param_width,
                    param_left, param_right, MYsamples_old)
            
            # Solve the model for the samples_new.
            data_new = self.lb_model(samples_new)
            
            # Make some decision about changing step_size(k).  There are
            # multiple ways to do this.
            # Determine step size
            #(kern_new, proposal) = kern.delta_step(data_new, kern_old)
            kern_new = rho_D(data_new)
            # Update the logical index of chains searching along the boundary
            left_roi = np.logical_and(kern_old > 0, kern_new < kern_old)
            
            step_ratio = determine_step_ratio(param_dist, samples_new)

            # Save and export concatentated arrays
            if self.chain_length < 4:
                pass
            elif (batch+1)%(self.chain_length/4) == 0:
                print "Current chain length: "+str(batch+1)+"/"+str(self.chain_length)
            samples = np.concatenate((samples, samples_new))
            data = np.concatenate((data, data_new))
            kern_samples = np.concatenate((kern_samples, kern_new))
            all_step_ratios = np.concatenate((all_step_ratios, step_ratio))
            mdat['step_ratios'] = all_step_ratios
            mdat['samples'] = samples
            mdat['data'] = data
            if comm.size > 1:
                super(sampler, self).save(mdat, psavefile)
            else:
                super(sampler, self).save(mdat, savefile)
            
            # Is the ratio greater than max?
            step_ratio[step_ratio > max_ratio] = max_ratio
            # Is the ratio less than min?
            step_ratio[step_ratio < min_ratio] = min_ratio

            # Don't update samples that have left the RoI (after finding it)
            # Don't update samples that have left the RoI (after finding it)
            MYsamples_old[np.logical_not(left_roi)] = samples_new[np.logical_not(left_roi)]
            kern_old[np.logical_not(left_roi)] = kern_new[np.logical_not(left_roi)]



        # collect everything
        MYsamples = np.copy(samples)
        MYdata = np.copy(data)
        MYall_step_ratios = np.copy(all_step_ratios)
        # ``parameter_samples`` is np.ndarray of shape (num_samples, ndim)
        samples = util.get_global_values(MYsamples,
                shape=(self.num_samples, np.shape(MYsamples)[1]))           
        # and ``data_samples`` is np.ndarray of shape (num_samples, mdim)
        data = util.get_global_values(MYdata, shape=(self.num_samples,
            np.shape(MYdata)[1]))
        # ``all_step_ratios`` is np.ndarray of shape (num_chains,
        # chain_length)
        all_step_ratios = util.get_global_values(MYall_step_ratios,
                shape=(self.num_samples,))
        all_step_ratios = np.reshape(all_step_ratios, (self.num_chains, self.chain_length))

        # save everything
        mdat['step_ratios'] = all_step_ratios
        mdat['samples'] = samples
        mdat['data'] = data
        super(sampler, self).save(mdat, savefile)

        return (samples, data, all_step_ratios)

def determine_step_ratio(param_dist, MYsamples_old, do_global=True):
    """
    Determine the mean pairwise distance between the current batch of samples.
    
    :param MYsamples_old:
    :type MYsamples_old:
    :param bool global: Flag whether or not to do this local to a processor or globally
    
    :rtype: :class:`numpy.ndarray`
    :returns: ``step_ratio``
    
    """
    
    # determine the average distance between minima
    # calculate average minimum pairwise distance between minima
    dist = spatial.distance_matrix(MYsamples_old, MYsamples_old)
    mindists = np.empty((MYsamples_old.shape[0],))
    for i in range(dist.shape[0]):
        mindists[i] = np.min(dist[i][dist[i] > 0])
    mindists_sum = np.sum(mindists)
    if do_global:
        mindists_sum = comm.allreduce(mindists, op=MPI.SUM)    
    mindists_avg = mindists_sum/(comm.size*MYsamples_old.shape[0])
    # set step ratio based on this distance
    step_ratio = mindists_avg/param_dist*np.ones(MYsamples_old.shape[0])
    return step_ratio

