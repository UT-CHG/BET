# -*- coding: utf-8 -*-
# Copyright (C) 2014-2015 The BET Development Team

# -*- coding: utf-8 -*-
# Lindley Graham 3/10/2014
"""
This module contains functions for adaptive random sampling. We assume we are
given access to a model, a parameter space, and a data space. The model is a
map from the paramter space to the data space. We desire to build up a set of
samples to solve an inverse problem thus giving us information about the
inverse mapping. Each sample consists of a parameter coordinate, data
coordinate pairing. We assume the measure of both spaces is Lebesgue.

We employ an approach based on using multiple sample chains. Once a chain
enters the region of interest the chain attempts to approximate the boundary of
the region of interest using a biscetion like method.

"""

import numpy as np
import bet.sampling.adaptiveSampling as asam
import bet.util as util
from scipy.interpolate import Rbf
import scipy.optimize as optimize
import scipy.spatial as spatial
import os
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

        :param int num_samples: Total number of samples
        :param int chain_length: Number of samples per chain
        :param lb_model: runs the model at a given set of parameter samples, (N,
            ndim), and returns data (N, mdim)
        """
        super(sampler, self).__init__(num_samples, chain_length, lb_model)

    def generalized_chains(self, param_min, param_max, t_set, rho_D,
            smoothIndicatorFun, savefile, initial_sample_type="lhs",
            criterion='center'):
        r"""
        This method adaptively generates samples similar to the method
        :meth:`bet.sampling.adaptiveSampling.generalized_chains`. New samples
        are chosen based on a RBF surrogate function. However, once
        a chain enters the region of interest the chain attempts to approximate
        the boundary of the region of interest using a biscetion like method.
       
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
        :rtype: tuple
        :returns: (``parameter_samples``, ``data_samples``,
            ``all_step_ratios``) where ``parameter_samples`` is np.ndarray of
            shape (num_samples, ndim), ``data_samples`` is np.ndarray of shape
            (num_samples, mdim), and ``all_step_ratios`` is np.ndarray of shape
            (num_chains, chain_length)

        """
        if size > 1:
            psavefile = os.path.join(os.path.dirname(savefile),
                    "proc{}{}".format(rank, os.path.basename(savefile)))

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
        step_ratio = t_set.init_ratio*np.ones(self.num_chains_pproc)
       
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
            MYsamples_old = np.empty((np.shape(samples_old)[0]/size,
                np.shape(samples_old)[1])) 
            comm.Scatter([samples_old, MPI.DOUBLE], [MYsamples_old, MPI.DOUBLE])
            MYdata_old = np.empty((np.shape(data_old)[0]/size,
                np.shape(data_old)[1])) 
            comm.Scatter([data_old, MPI.DOUBLE], [MYdata_old, MPI.DOUBLE])
            step_ratio = self.determine_step_ratio(MYsamples_old)
        else:
            MYsamples_old = np.copy(samples_old)
            MYdata_old = np.copy(data_old)

        samples = MYsamples_old
        data = MYdata_old
        all_step_ratios = step_ratio
        #(kern_old, proposal) = kern.delta_step(MYdata_old, None)
        kern_old = rho_D(MYdata_old)
        kern_samples = kern_old
        mdat = dict()
        self.update_mdict(mdat)
        
        # logical index of 
        boundary_chains = np.zeros(self.num_chains_pproc, dtype='bool')
        normal_vectors = None


        MYsamples_rbf = MYsamples_old
        rbf_data = smoothIndicatorFun(MYdata_old)
        step_ratio, MYsamples_old = rbf_samples(MYsamples_rbf, rbf_data, MYsamples_old,
                param_dist)
        
        for batch in xrange(1, self.chain_length):
            # For each of N samples_old, create N new parameter samples using
            # transition set and step_ratio. Call these samples samples_new.
            samples_new = t_set.step(step_ratio, param_width,
                    param_left, param_right, MYsamples_old)

            # if batch > 1 and the old samples were in roi then project the
            # corresponding new samples on to the hyperplane perpendicular to
            # the previously calculated normal vector
            if batch > 1 and boundary_chains.any():
                # q_proj = q - dot(q-p)*normal_vector
                #dot(q-p)
                dotqp = np.einsum('...i,...i', 
                        samples_new[boundary_chains, 
                        :] - MYsamples_old[boundary_chains, :], 
                        normal_vectors)
                dotqp = np.repeat([dotqp], samples_new.shape[1], 0).transpose()
                samples_new[boundary_chains, :] = samples_new[boundary_chains, 
                        :] - dotqp*normal_vectors
            
            # Solve the model for the samples_new.
            data_new = self.lb_model(samples_new)
            
            # Make some decision about changing step_size(k).  There are
            # multiple ways to do this.
            # Determine step size
            #(kern_new, proposal) = kern.delta_step(data_new, kern_old)
            kern_new = rho_D(data_new)
            # Update the logical index of chains searching along the boundary
            boundary_chains = np.logical_or(kern_new > 0, boundary_chains)
            # Update the samples used to create the RBF surrogate
            MYsamples_rbf = np.concatenate((MYsamples_rbf,
                samples_new[np.logical_not(boundary_chains), :]))
            rbf_data = np.concatenate((rbf_data,
                smoothIndicatorFun(data_new[np.logical_not(boundary_chains), :])))
            step_ratio_rbf, minima = rbf_samples(MYsamples_rbf, rbf_data,
                    MYsamples_old[np.logical_not(boundary_chains), :],
                    param_dist)
            MYsamples_old[np.logical_not(boundary_chains), :] = minima
            step_ratio[np.logical_not(boundary_chains)] = step_ratio_rbf

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
            if size > 1:
                super(sampler, self).save(mdat, psavefile)
            else:
                super(sampler, self).save(mdat, savefile)
            
            interior = np.zeros((np.sum(boundary_chains),
                samples_new.shape[1])) 
            exterior = np.zeros((np.sum(boundary_chains),
                samples_new.shape[1]))
            only_interior = list()
            # Loop through all the chains 
            for i, chain in enumerate(boundary_chains):
                if chain:
                    # search only within the points on this chain
                    index = range(i, samples.shape[0], self.num_chains_pproc)
                    interior_ind = kern_samples[index] > 0
                    points = samples[index, :]
                    # determine the interior and exterior points on this chain
                    interior_points = points[interior_ind, :]
                    exterior_points = points[np.logical_not(interior_ind), :]
                    # find the most recent interior point
                    j = np.sum(boundary_chains[:i+1])-1
                    interior[j, :] = interior_points[-1, :]
                    # find the closest exterior point
                    if exterior_points.shape[0] > 1:
                        dist = (exterior_points - interior[j, :])**2
                        dist = np.sum(dist, axis=1)
                        exterior[j, :] = exterior_points[np.argmin(dist), :]
                    elif exterior_points.shape[0] > 0:
                        exterior[j, :] = exterior_points
                    else:
                        only_interior.append((i, j))
                else:
                    continue

            # Remove chains with only interior points from the list of boundary
            # chains (we might want these to be limited memory chains)
            if len(only_interior) > 0:
                i_ind = range(boundary_chains.shape[0])
                j_ind = range(interior.shape[0])
                only_interior = np.array(only_interior)
                boundary_chains[only_interior[:, 0]] = False
                for j in only_interior[:, 1]:
                    j_ind.remove(j)
                interior = interior[i_ind, :]
                exterior = exterior[j_ind, :]

            # calculate the normal vector
            normal_vectors = interior - exterior
            # calculate the point between the interior and exterior points
            midpoint = 0.5*(interior+exterior)
            # calculate the ratio of the propoal box (either the current step
            # ratio or the size of the hyperbox formed when the two points are
            # opposite diagonal corners of the hyperbox)
            step_ratio[boundary_chains] = normal_vectors/param_width[0:normal_vectors.shape[0], :]

            # Is the ratio greater than max?
            step_ratio[step_ratio > max_ratio] = max_ratio
            # Is the ratio less than min?
            step_ratio[step_ratio < min_ratio] = min_ratio

            # If the chain is going along the boundary set the old sample to be
            # the midpoint between the interior sample and the nearest exterior
            # sample
            nonboundary = np.logical_not(boundary_chains)
            MYsamples_old[nonboundary] = samples_new[nonboundary]
            MYsamples_old[boundary_chains] = midpoint


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

def rbf_samples(MYsamples_rbf, rbf_data, MYsamples_old, param_dist):
    r"""
    Choose points around which to sample next based on a RBF approximation
    to :math:`\rho_\mathcal{D}(Q(\lambda))`.
    
    :param MYsamples_rbf:
    :type MYsamples_rbf:
    :param MYsamples_old:
    :type MYsamples_old:
    :param double param_dist:

    :rtype: tuple
    :returns: (step_ratio, MYsamples_old)

    """
    dim = MYsamples_rbf.shape[1]
    samples_rbf = util.get_global_values(MYsamples_rbf)

    if dim == 1:
        surrogate = Rbf(samples_rbf[:, 0], rbf_data)
    elif dim == 2:
        rbfi = Rbf(samples_rbf[:, 0], samples_rbf[:, 1], 
                rbf_data)
        def surrogate(input):
            return float(rbfi(input[0], input[1]))
    elif dim == 3:
        rbfi = Rbf(samples_rbf[:, 0], samples_rbf[:, 1],
                samples_rbf[:, 2], rbf_data) 
        def surrogate(input):
            return float(rbfi(input[0], input[1], input[2]))
    elif dim == 4:
        rbfi = Rbf(samples_rbf[:, 0], samples_rbf[:, 1],
                samples_rbf[:, 2], samples_rbf[:, 3], rbf_data)
        def surrogate(input):
            return float(rbfi(input[0], input[1], input[2], input[3]))
    elif dim == 5:
        rbfi = Rbf(samples_rbf[:, 0], samples_rbf[:, 1], 
                samples_rbf[:, 2], samples_rbf[:, 3], samples_rbf[:, 4],
                rbf_data)
        def surrogate(input):
            return float(rbfi(input[0], input[1], input[2], input[3],
                    input[4]))
    elif dim == 6:
        rbfi = Rbf(samples_rbf[:, 0], samples_rbf[:, 1], 
                samples_rbf[:, 2], samples_rbf[:, 3], samples_rbf[:, 4],
                samples_rbf[:, 5], rbf_data)
        def surrogate(input):
            return float(rbfi(input[0], input[1], input[2], input[3],
                    input[4], input[5]))
    elif dim == 7:
        rbfi = Rbf(samples_rbf[:, 0], samples_rbf[:, 1], 
                samples_rbf[:, 2], samples_rbf[:, 3], samples_rbf[:, 4],
                samples_rbf[:, 5], samples_rbf[:, 6], rbf_data)
        def surrogate(input):
            return float(rbfi(input[0], input[1], input[2], input[3],
                    input[4], input[5], input[6]))
    else:
        print "surrogate creation only supported for up to 7 dimensions"
        quit()

    # evalutate minima of RBF using inital points as guesses
    for i in range(MYsamples_old.shape[0]):
        res = optimize.minimize(surrogate, MYsamples_old[i, :],
                method='Powell')
        if res.success:
            MYsamples_old[i, :] = res.x

    # determine the average distance between minima
    # calculate average minimum pairwise distance between minima
    dist = spatial.distance_matrix(MYsamples_old, MYsamples_old)
    mindists = np.empty((MYsamples_old.shape[0],))
    for i in range(dist.shape[0]):
        mindists[i] = np.min(dist[i][dist[i] > 0])
    mindists_sum = np.sum(mindists)
    mindists_sum = comm.allreduce(mindists, op=MPI.SUM)
    mindists_avg = mindists_sum/(comm.size*MYsamples_old.shape[0])
    # set step ratio based on this distance
    step_ratio = mindists_avg/param_dist*np.ones(MYsamples_old.shape[0])
    return step_ratio, MYsamples_old

