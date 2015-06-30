# -*- coding: utf-8 -*-
# Copyright (C) 2014-2015 The BET Development Team
# Lindley Graham 6/13/15
""" 

This module contains functions for adaptive random sampling using Newton's
method.
We assume we are given access to a model, a parameter space, and a data space.
The model is a map from the paramter space to the data space. We desire to
build up a set of samples to solve an inverse problem thus giving us
information about the inverse mapping. Each sample consists of a parameter
coordinate, data coordinate pairing. We assume the measure of both spaces is
Lebesgue.

We employ an approach based on using multiple sample chains and Newton's
method. We approximate the gradient using clusters of samples to form a RBF
approximation. If a starting set of samples are provided we can use either a
FFD, CFD, or RBF approximaiton of the gradient for the first Newton step.
Otherwise, we use a RBF approximation. 

Once a chain has entered the implicitly defined RoI (region of interest) we
randomly generate new clusters of samples while not allowing the centers of the
clusters to leave the RoI.
"""
import numpy as np
import bet.sampling.adaptiveSampling as asam
import scipy.spatial as spatial
import bet.util as util
import os, math
import bet.sensitivity.gradients as grad
from bet.Comm import comm, MPI
from pyDOE import lhs
import bet.calculateP.indicatorFunctions as indF

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
            smoothIndicatorFun, savefile, initial_sample_type="random",
            criterion='center', radius_ratio=0.01, initial_samples=None,
            initial_data=None, cluster_type='rbf', TOL=1e-8, nominal_ratio=0.1): 
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

        param_domain_test = indF.hyperrectangle(param_min, param_max)

        # Initialize Nx1 vector Step_size = something reasonable (based on size
        # of domain and transition set type)
        # Calculate domain size
        param_dist = np.sqrt(np.sum((param_max-param_min)**2))
        param_left = np.repeat([param_min], self.num_chains_pproc, 0)
        param_right = np.repeat([param_max], self.num_chains_pproc, 0)
        param_width = param_right - param_left
        radius = radius_ratio*(param_max-param_min)
        lambda_dim = len(param_max)

        if cluster_type == 'rbf':
            samples_p_cluster = lambda_dim+1
        elif cluster_type == 'ffd':
            samples_p_cluster = lambda_dim
        elif cluster_type == 'cfd':
            samples_p_cluster = 2*lambda_dim
        
        # if given samples and data split them up otherwise create them
        if type(initial_samples) == type(None) or \
                type(initial_data) == type(None):
            first_cluster_type = cluster_type
            self.num_chains_pproc = int(math.ceil(self.num_samples/float(\
                self.chain_length*comm.size*(samples_p_cluster+1))))
            self.num_chains = comm.size * self.num_chains_pproc
            self.num_samples = self.chain_length * self.num_chains * \
                    (samples_p_cluster+1)
            num_samples = self.num_samples
            self.sample_batch_no = np.repeat(range(self.chain_length),
                    self.num_chains*(samples_p_cluster+1), 0)

            # Initiative first batch of N samples (maybe taken from latin
            # hypercube/space-filling curve to fully explore parameter space -
            # not necessarily random). Call these Samples_old.
            if initial_sample_type == "r" or \
                    initial_sample_type == "random":
                        MYcenters_old = param_width[:self.num_chains_pproc]*\
                            np.random.random((self.num_chains_pproc,
                            lambda_dim))
            elif initial_sample_type == "lhs":
                MYcenters_old = param_width[:self.num_chains_pproc]*lhs(param_min.shape[-1],
                        self.num_chains_pproc, criterion)
            MYcenters_old = MYcenters_old + param_left[:self.num_chains_pproc]

            if cluster_type == 'rbf':
                MYsamples_old = grad.sample_l1_ball(MYcenters_old, lambda_dim+1,
                    radius) 
            elif cluster_type == 'ffd':
                MYsamples_old = grad.pick_ffd_points(MYcenters_old, radius)
            elif cluster_type == 'cfd':
                MYsamples_old = grad.pick_cfd_points(MYcenters_old, radius)

            (MYsamples_old, MYdata_old) = super(sampler,
                    self).user_samples(MYsamples_old, savefile)
            self.num_samples = num_samples

        else:
            # split up the old samples and data this assumes that the centers
            # are listed first and the l1 samples next
            # assumes that the number of centers is divisible by the
            # of chains per processor

            self.num_chains_pproc = int(math.ceil((self.num_samples - \
                    initial_samples.shape[0])/float((self.chain_length-1)\
                    *comm.size*(samples_p_cluster+1))))
            self.num_chains = comm.size * self.num_chains_pproc
            self.num_samples = initial_samples.shape[0] + (self.chain_length-1)\
                    * self.num_chains * (samples_p_cluster+1)
            repeats = [initial_samples.shape[0]]
            repeats.extend([(self.num_chains)*(samples_p_cluster+1)]*\
                    (self.chain_length-1))
            self.sample_batch_no = np.repeat(range(self.chain_length),
                    repeats, 0)

            centers = initial_samples[:self.num_chains, :]
            data_centers = initial_data[:self.num_chains, :]
            offset = self.num_chains_pproc*comm.rank
            # determine the type of clusters (FFD, CFD, RBF)
            num_FFD_total = self.num_chains*(lambda_dim+1)
            num_CFD_total = self.num_chains*(2*lambda_dim + 1)
            num_RBF_total = self.num_chains*(lambda_dim + 2)

            if initial_samples.shape[0] == num_FFD_total:
                first_cluster_type = 'ffd'
                samples_p_cluster = lambda_dim
            elif initial_samples.shape[0] == num_CFD_total:
                first_cluster_type = 'cfd'
                samples_p_cluster = 2*lambda_dim
            elif initial_samples.shape[0] == num_RBF_total:
                first_cluster_type = 'rbf'
                samples_p_cluster = lambda_dim+1
            else:
                print "NOT A VALID CLUSTER TYPE"
                quit()
            
            MYcenters_old = centers[offset:offset+self.num_chains_pproc]
            MYdata_centers = data_centers[offset:offset+self.num_chains_pproc]
            offset = self.num_chains+self.num_chains_pproc*samples_p_cluster\
                    *comm.rank
            MYclusters = initial_samples[offset:offset+self.num_chains_pproc*\
                    samples_p_cluster*comm.rank, :]
            MYdata_clusters = initial_data[offset:offset+self.num_chains_pproc*\
                    samples_p_cluster*comm.rank, :]
            MYsamples_old = np.concatenate((MYcenters_old, MYclusters))
            MYdata_old = np.concatenate((MYdata_centers, MYdata_clusters))

        comm.Barrier()

        samples = MYsamples_old
        data = MYdata_old
        all_step_ratios = []
        kern_old = rho_D(MYdata_old[:self.num_chains_pproc, :])
        rank_old = smoothIndicatorFun(MYdata_old)
        kern_samples = kern_old
        mdat = dict()
        self.update_mdict(mdat)
        MYcenters_new = np.copy(MYcenters_old)

        
        for batch in xrange(1, self.chain_length):
            # print 'batch no.', batch
            # Determine the rank of the old samples
            rank_old = smoothIndicatorFun(MYdata_old)
            # For the centers that are not in the RoI do a newton step
            centers_in_RoI = (kern_old > 0)
            not_in_RoI = np.logical_not(centers_in_RoI)

            if not_in_RoI.any():
                # Determine indices to create np.concatenate([centers, clusters])
                # for centers not in the RoI
                #import pdb; pdb.set_trace()
                samples_woC_in_RoI = (np.arange(self.num_chains_pproc)+1)*\
                        not_in_RoI
                samples_woC_in_RoI = samples_woC_in_RoI.nonzero()[0]
                cluster_list = []
                cluster_list.append(np.copy(samples_woC_in_RoI))
                for c_num in samples_woC_in_RoI:
                    offset = self.num_chains_pproc + c_num*samples_p_cluster
                    cluster_list.append(np.arange(offset,
                        offset+samples_p_cluster))
                samples_woC_in_RoI = np.concatenate(cluster_list)
                # TODO: add in ability to reuse points
                # calculate gradient based on gradient type for the first one
                # G = grad.calculate_gradients(samples, data, num_centers,
                # rvec, normalize=False)
                if cluster_type == 'rbf' or (batch == 1 and \
                        first_cluster_type == 'rbf'):
                    G = grad.calculate_gradients_rbf(MYsamples_old\
                            [samples_woC_in_RoI, :], 
                            rank_old[samples_woC_in_RoI],
                            #RBF = 'Multiquadric',
                            #RBF = 'InverseMultiquadric',
                            #RBF = 'C4Matern',
                            normalize=False)
                elif cluster_type == 'ffd' or (batch == 1 and \
                        first_cluster_type == 'ffd'):
                    G = grad.calculate_gradients_ffd(MYsamples_old\
                            [samples_woC_in_RoI, :], 
                            rank_old[samples_woC_in_RoI],
                            normalize=False)
                elif cluster_type == 'cfd' or (batch == 1 and \
                        first_cluster_type == 'cfd'):
                    G = grad.calculate_gradients_cfd(MYsamples_old\
                            [samples_woC_in_RoI, :], 
                            rank_old[samples_woC_in_RoI],
                            normalize=False)

                # reset the samples_p_cluster to be the rbf for remaining
                # batches
                if cluster_type != first_cluster_type and batch == 1:
                    if cluster_type == 'rbf':
                        samples_p_cluster = lambda_dim+1
                    elif cluster_type == 'ffd':
                        samples_p_cluster = lambda_dim
                    elif cluster_type == 'cfd':
                        samples_p_cluster = 2*lambda_dim
                normG = np.linalg.norm(G, axis=2)
                # take a Newton step
                # calculate step size
                # TODO: we are currently doing b = a - \gamma \Grad f(a)
                # we might need to be more careful about our choice of \gamma
                step_size = (0-rank_old[not_in_RoI])
                step_size = util.fix_dimensions_vector_2darray(step_size)
                step_size = step_size/normG
                step_size = np.column_stack([step_size]*lambda_dim)
                # move centers
                MYcenters_new[not_in_RoI] = MYcenters_old[not_in_RoI]\
                        + step_size*np.round(G[:, 0, :]/normG, decimals=11)              
                # If the tolerance is too low take a random step as you
                # would using centers_in_RoI
                # determine chains with gradient < TOL 
                do_random_step = np.reshape(np.logical_and(normG <= TOL,
                    np.isnan(normG)), (normG.shape[0],))
                if np.any(do_random_step):
                    print 'Random step, too small normG, batch: ', batch
                    not_in_RoI_RS = np.zeros(not_in_RoI.shape, dtype=bool)
                    not_in_RoI_RS[do_random_step] = True
                    # do a random of size 2*radius
                    step_ratio = 2*radius_ratio*np.ones(not_in_Roi_RS.shape)
                    MYcenters_new[not_in_RoI_RS] = t_set.step(step_ratio,
                            param_width[not_in_RoI_RS],
                            param_left[not_in_RoI_RS], param_right[not_in_RoI_RS],
                            MYcenters_old[not_in_RoI_RS])

                # did the step take us outside of lambda?
                step_out = np.logical_not(param_domain_test(MYcenters_new\
                        [not_in_RoI]))

                if step_out.any():
                    msg = 'Restart {} chain(s), '.format(sum(step_out))
                    msg += 'left parameter domain, batch:{}'.format(batch)
                    print msg
                    not_in_RoI_RS = np.zeros(not_in_RoI.shape, dtype=bool)
                    not_in_RoI_RS[not_in_RoI] = step_out
                    # Restart the samples with if samples leave the
                    # parameter domain
                    MYcenters_new[not_in_RoI_RS] = param_left[not_in_RoI_RS] + \
                            param_width[not_in_RoI_RS]*np.random.random((\
                            np.sum(not_in_RoI_RS), lambda_dim))

            # For the centers that are in the RoI sample uniformly
            step_ratio = determine_step_ratio(param_dist, 
                    MYcenters_old[centers_in_RoI], nominal_ratio)
            if batch > 1 and left_roi[centers_in_RoI].any():
                step_ratio[left_roi[centers_in_RoI]] = 0.5*\
                        step_ratio[left_roi[centers_in_RoI]]
            MYcenters_new[centers_in_RoI] = t_set.step(step_ratio,
                    param_width[centers_in_RoI],
                    param_left[centers_in_RoI], param_right[centers_in_RoI],
                    MYcenters_old[centers_in_RoI])

            # Finish creating the new samples
            if cluster_type == 'rbf':
                samples_p_cluster = lambda_dim+1
                samples_new = grad.sample_l1_ball(MYcenters_new, lambda_dim+1,
                    radius) 
            elif cluster_type == 'ffd':
                samples_p_cluster = lambda_dim
                samples_new = grad.pick_ffd_points(MYcenters_new, radius)
            elif cluster_type == 'cfd':
                samples_p_cluster = 2*lambda_dim
                samples_new = grad.pick_cfd_points(MYcenters_new, radius)

            # Solve the model for the samples_new.
            data_new = self.lb_model(samples_new)
            
            # update the indicator used to determine if the center sample is in
            # the RoI or not
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
                msg = "{}/{} in RoI".format(np.sum(centers_in_RoI),
                        self.num_chains_pproc)
            samples = np.concatenate((samples, samples_new))
            data = np.concatenate((data, data_new))
            kern_samples = np.concatenate((kern_samples, kern_new))
            # join up the step ratios for the samples in/out of the RoI
            joint_step_ratios = np.empty((self.num_chains_pproc,))
            if not_in_RoI.any():
                joint_step_ratios[not_in_RoI] = step_size/param_width[not_in_RoI]
            joint_step_ratios[centers_in_RoI] = step_ratio
            all_step_ratios = np.concatenate((all_step_ratios,
                joint_step_ratios))
            mdat['step_ratios'] = all_step_ratios
            mdat['samples'] = samples
            mdat['data'] = data
            if comm.size > 1:
                super(sampler, self).save(mdat, psavefile)
            else:
                super(sampler, self).save(mdat, savefile)
            
            MYcenters_old = MYcenters_new
            MYdata_old = data_new
            MYsamples_old = samples_new
            kern_old = kern_new

        centers_in_RoI = (kern_old > 0)

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
        all_step_ratios = np.reshape(all_step_ratios,
                (self.num_chains*(samples_p_cluster+1), self.chain_length))

        # save everything
        mdat['step_ratios'] = all_step_ratios
        mdat['samples'] = samples
        mdat['data'] = data
        super(sampler, self).save(mdat, savefile)

        return (samples, data, all_step_ratios)


def determine_step_ratio(param_dist, MYsamples_old, nominal_ratio=0.50):
    """
    Estimate the radius of the RoI using the current samples. This could be
    improved by using all of the samplings in the RoI. This will begin to
    break when sets are not simply connected.
    
    :param MYsamples_old:
    :type MYsamples_old:
    
    :rtype: :class:`numpy.ndarray`
    :returns: ``step_ratio``
    
    """
    all_samples = util.get_global_values(MYsamples_old) 
    dist = spatial.distance_matrix(all_samples, all_samples)

    if dist.shape == (0,) or dist.shape == (1,) or dist.shape == (0, 0):
        step_ratio = nominal_ratio*np.ones(MYsamples_old.shape[0])
    else:
        # set step ratio based on this distance
        step_ratio = nominal_ratio*np.max(dist)/param_dist*\
                np.ones(MYsamples_old.shape[0])
    #return nominal_ratio*np.zeros(MYsamples_old.shape[0])
    return step_ratio

