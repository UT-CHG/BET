# Copyright (C) 2014-2015 Lindley Graham and Steven Mattis

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

import bet.util as util
import numpy as np
import bet.sampling.adaptiveSampling as asam
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

    def generalized_chains(self, param_min, param_max, t_set, kern,
            savefile, initial_sample_type="lhs", criterion='center'):
        """
        This method adaptively generates samples similar to the method
        :meth:`bet.sampling.adaptiveSampling.generalized_chains`. However, once
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
        :param function kern: functional that acts on the data used to
            determine the proposed change to the ``step_size``
        :type kernel: :class:~`bet.sampling.adaptiveSampling.kernel` object.
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
        MYsamples_old = np.empty((np.shape(samples_old)[0]/size,
            np.shape(samples_old)[1])) 
        comm.Scatter([samples_old, MPI.DOUBLE], [MYsamples_old, MPI.DOUBLE])
        MYdata_old = np.empty((np.shape(data_old)[0]/size,
            np.shape(data_old)[1])) 
        comm.Scatter([data_old, MPI.DOUBLE], [MYdata_old, MPI.DOUBLE])

        samples = MYsamples_old
        data = MYdata_old
        all_step_ratios = step_ratio
        (kern_old, proposal) = kern.delta_step(MYdata_old, None)
        kern_samples = kern_old
        mdat = dict()
        self.update_mdict(mdat)
        
        # logical index of 
        boundary_chains = np.zeros(self.num_chains_pproc, dtype='bool')
        normal_vectors = None

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
            (kern_new, proposal) = kern.delta_step(data_new, kern_old)
            step_ratio = proposal*step_ratio

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
            
            # Update the logical index of chains searching along the boundary
            boundary_chains = np.logical_or(kern_new > 0, boundary_chains)
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


        if comm.size > 1:
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
            all_step_ratios = np.empty((self.num_chains, self.chain_length), dtype=np.float64)
            comm.Allgather([MYall_step_ratios, MPI.DOUBLE], [all_step_ratios, MPI.DOUBLE])
        else:
            all_step_ratios = np.reshape(all_step_ratios, (self.num_chains, self.chain_length))

        # save everything
        mdat['step_ratios'] = all_step_ratios
        mdat['samples'] = samples
        mdat['data'] = data
        super(sampler, self).save(mdat, savefile)

        return (samples, data, all_step_ratios)

"""
# keep track of boundary points
rho_D = kernel.rho_D
# Identify points inside roi
inside = rho_D(data) > 0
outside = np.logical_not(inside)
dim = data.shape[1]
# determine 2d nearest neighbors of points inside roi
tree = KDTree(data)
nearest_neighbors = tree.query(inside, 2*dim+1)

for i, neighbors in nearest_neighbors:
    pass

# for each interior point
# if a neighbor is interior do nothing
# if a neighbor is exterior and further than the minimum step size
# add 0.5*(neighbor+interior_point) to the list of new samples
# or choose two new samples randomly within the hyperplane perpendicular to the
# line formed by the exterior and interior points
# remove
"""
    
