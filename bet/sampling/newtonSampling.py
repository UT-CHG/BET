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
import bet.sensitivity.gradients as grad
from bet.Comm import comm, MPI
from pyDOE import lhs

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
        #TODO: update this so that chain_length and num_samples and num_chains
        # are updates so that the number of samples matches with using the l1
        # ball and lambda_dim+1 samples
        super(sampler, self).__init__(num_samples, chain_length, lb_model)

    def generalized_chains(self, param_min, param_max, t_set, rho_D,
            smoothIndicatorFun, savefile, initial_sample_type="random",
            criterion='center', radius=0.1, initial_samples=None,
            initial_data=None): 
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
        lambda_dim = len(param_max)
    
        if type(initial_samples) == 'NoneType' or type(initial_data) == 'NoneType':
            # Initiative first batch of N samples (maybe taken from latin
            # hypercube/space-filling curve to fully explore parameter space - not
            # necessarily random). Call these Samples_old.
            if initial_sample_type == "r" or initial_sample_type == "random":
                MYcenters_old = param_width*np.random.random(param_left.shape)
            elif initial_sample_type == "lhs":
                MYcenters_old = param_width*lhs(param_min.shape[-1],
                        self.num_chains_pproc, criterion)
            MYcenters_old = MYcenters_old + param_left
            MYsamples_old = grad.sample_l1_ball(MYcenters_old, lambda_dim+1,
                    radius)
            (MYsamples_old, MYdata_old) = super(sampler,
                    self).user_samples(MYsamples_old, savefile)
        else:
            # split up the old samples and data this assumes that the centers
            # are listed first and the l1 samples next
            #TODO: assumes that the number of centers is divisible by the
            # of chains per processor, might want to update to handle different
            # numbers of chains per processor
            #TODO: check with Scott that this is correct implementation
            centers = initial_samples[:self.num_chains, :]
            data_centers = initial_data[:self.num_chains, :]
            offset = self.num_chains_pproc*comm.rank
            MYcenters_old = centers[offset:offset+self.num_chains_pproc]
            MYdata_centers = data_centers[offset:offset+self.num_chains_pproc]
            offset = self.num_chains+self.num_chains_pproc*(lambda_dim+1)*comm.rank
            MYnon_centers = initial_samples[offset:offset+(lambda_dim+1)*comm.rank, :]
            MYnon_data_centers = initial_data[offset:offset+(lambda_dim+1)*comm.rank, :]
            MYsamples_old = np.concatenate((MYcenters_old, MYnon_centers))
            MYdata_old = np.concatenate((MYdata_centers, MYnon_data_centers))


        self.num_samples = self.chain_length * self.num_chains * (lambda_dim+1)
        comm.Barrier()

        samples = MYsamples_old
        data = MYdata_old
        all_step_ratios = radius*np.ones((self.num_chains_pproc,)) #step_ratio
        kern_old = rho_D(MYdata_old[self.num_chains_pproc, :])
        rank_old = smoothIndicatorFun(MYdata_old)
        kern_samples = kern_old
        mdat = dict()
        self.update_mdict(mdat)
        MYcenters_new = np.empty(MYcenters_old.shape)

        
        for batch in xrange(1, self.chain_length):
            # Determine the rank of the old samples
            rank_old = smoothIndicatorFun(MYdata_old)
            # For the centers that are not in the RoI do a newton step
            centers_in_RoI = kern_old
            samples_wC_in_RoI = None
            G = grad.calculate_gradients_rbf(MYsamples_old[samples_wC_in_RoI,:],
                    rank_old[samples_wC_in_RoI],
                    MYsamples_old[centers_in_RoI, :], normalized=False)
            normG = np.linalg.norm(G, axis=2)
            # TODO: determine if samples always need to be normalized with so
            # that radius is the same in all directions if so update
            # TODO: check this with Scott
            step_size = (0-rank_old[:self.num_chains_pproc])
            step_size = step_size/normG**2
            MYcenters_new[centers_in_RoI, :] = MYcenters_old[centers_in_RoI] + \
                    step_size*G[:, 0, :]

            # For the centers that are in the RoI sample uniformly
            not_in_RoI = np.logical_not(centers_in_RoI)
            step_ratio = determine_step_ratio(param_dist[not_in_RoI], 
                    MYcenters_old[not_in_RoI]) 
            MYcenters_new[not_in_RoI] = t_set.step(step_ratio[not_in_RoI],
                    param_width[not_in_RoI], param_left[not_in_RoI],
                    param_right[not_in_RoI], MYcenters_old[not_in_RoI])

            # Finish creating the new samples
            samples_new = grad.sample_l1_ball(MYcenters_new, lambda_dim+1,
                    radius) 

            # Solve the model for the samples_new.
            data_new = self.lb_model(samples_new)
            
            # Make some decision about changing step_size(k).  There are
            # multiple ways to do this.
            # Determine step size
            #(kern_new, proposal) = kern.delta_step(data_new, kern_old)
            kern_new = rho_D(data_new[:self.num_chains_pproc, :])
            # Update the logical index of chains searching along the boundary
            left_roi = np.logical_and(kern_old > 0, kern_new < kern_old)
            

            # Save and export concatentated arrays
            if self.chain_length < 4:
                pass
            elif (batch+1)%(self.chain_length/4) == 0:
                msg = "Current chain length: "+str(batch+1)
                msg += "/"+str(self.chain_length)
                print msg
            samples = np.concatenate((samples, samples_new))
            data = np.concatenate((data, data_new))
            kern_samples = np.concatenate((kern_samples, kern_new))
            # TODO: fix how this is done so that we're saving step size too
            all_step_ratios = np.concatenate((all_step_ratios, step_ratio))
            mdat['step_ratios'] = all_step_ratios
            mdat['samples'] = samples
            mdat['data'] = data
            if comm.size > 1:
                super(sampler, self).save(mdat, psavefile)
            else:
                super(sampler, self).save(mdat, savefile)
            
            # Don't update centers that have left the RoI (after finding it)
            MYcenters_old[np.logical_not(left_roi)] = MYcenters_new[np.logical_not(left_roi)]
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

