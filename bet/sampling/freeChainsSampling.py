# -*- coding: utf-8 -*-
# Lindley Graham 3/10/2014
"""
This module contains functions for adaptive random sampling. We assume we are
given access to a model, a parameter space, and a data space. The model is a
map from the paramter space to the data space. We desire to build up a set of
samples to solve an inverse problem thus giving us information about the
inverse mapping. Each sample consists of a parameter coordinate, data
coordinate pairing. We assume the measure of both spaces is Lebesgue.

We employ an approach based on using multiple sample chains.
"""

import numpy as np
import scipy.io as sio
import bet.sampling.basicSampling as bsam
import bet.sampling.adaptiveSampling as asam
import math

import bet.vis.Comm as comm
import bet.vis.MPI as MPI
size = comm.Get_size()
rank = comm.Get_rank()

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
        super(sampler, self).__init__(num_samples, chain_length, lb_model)
        self.chain_length = chain_length
        self.num_chains_pproc = int(math.ceil(num_samples/float(chain_length*size)))
        self.num_chains = size * self.num_chains_pproc
        self.num_samples = chain_length * self.num_chains
        self.lb_model = lb_model
        self.sample_batch_no = np.repeat(range(self.num_chains), chain_length,
                0)

    def generalized_chains(self, param_min, param_max, t_kernel, kern,
            savefile, initial_sample_type="lhs", criterion='center'):
        """
        Basic adaptive sampling algorithm using generalized chains.
       
        :param string initial_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param param_min: minimum value for each parameter dimension
        :type param_min: np.array (ndim,)
        :param param_max: maximum value for each parameter dimension
        :type param_max: np.array (ndim,)
        :param t_kernel: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_kernel: :class:~`t_kernel`
        :param function kern: functional that acts on the data used to
            determine the proposed change to the ``step_size``
        :param string savefile: filename to save samples and data
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        :rtype: tuple
        :returns: (``parameter_samples``, ``data_samples``, ``all_step_ratios``) where
            ``parameter_samples`` is np.ndarray of shape (num_samples, ndim),
            ``data_samples`` is np.ndarray of shape (num_samples, mdim), and 
            ``all_step_ratios`` is np.ndarray of shape (num_chains,
            chain_length)

        """

        if size > 1:
            savefile = os.path.join(os.path.dirname(savefile),
                    "proc{}{}".format(rank, os.path.basename(savefile)))

        # Initialize Nx1 vector Step_size = something reasonable (based on size
        # of domain and transition set type)
        # Calculate domain size
        param_left = np.repeat([param_min], self.num_chains_pproc, 0)
        param_right = np.repeat([param_max], self.num_chains_pproc, 0)
        param_width = param_right - param_left
        # Calculate step_size
        max_ratio = t_kernel.max_ratio
        min_ratio = t_kernel.min_ratio
        step_ratio = t_kernel.init_ratio*np.ones(self.num_chains_pproc)
       
        # Initiative first batch of N samples (maybe taken from latin
        # hypercube/space-filling curve to fully explore parameter space - not
        # necessarily random). Call these Samples_old.
        (samples_old, data_old) = super(sampler, self).random_samples(
                initial_sample_type, param_min, param_max, savefile,
                self.num_chains, criterion)
        comm.Barrier()

        
        # now split it all up
        MYsamples_old = np.empty((np.shape(samples_old)[0]/size, np.shape(samples_old)[1]))
        comm.Scatter([samples_old, MPI.DOUBLE], [MYsamples_old, MPI.DOUBLE])
        MYdata_old = np.empty((np.shape(data_old)[0]/size, np.shape(data_old)[1]))
        comm.Scatter([data_old, MPI.DOUBLE], [MYdata_old,
            MPI.DOUBLE])

        samples = MYsamples_old
        data = MYdata_old
        all_step_ratios = step_ratio
        (kern_old, proposal) = kern.delta_step(data_old, None)
        mdat = dict()
        self.update_mdict(mdat)

        for batch in xrange(1, self.chain_length):
            # For each of N samples_old, create N new parameter samples using
            # transition set and step_ratio. Call these samples samples_new.
            samples_new = t_kernel.step(step_ratio, param_width,
                    param_left, param_right, MYsamples_old)
            
            # Solve the model for the samples_new.
            data_new = self.lb_model(samples_new)
            
            # Make some decision about changing step_size(k).  There are
            # multiple ways to do this.
            # Determine step size
            (kern_old, proposal) = kern.delta_step(data_new, kern_old)
            step_ratio = proposal*step_ratio
            # Is the ratio greater than max?
            step_ratio[step_ratio > max_ratio] = max_ratio
            # Is the ratio less than min?
            step_ratio[step_ratio < min_ratio] = min_ratio

            # Save and export concatentated arrays
            if self.chain_length < 4:
                pass
            elif (batch+1)%(self.chain_length/4) == 0:
                print "Current chain length: "+str(batch+1)+"/"+str(self.chain_length)
            samples = np.concatenate((samples, samples_new))
            data = np.concatenate((data, data_new))
            all_step_ratios = np.concatenate((all_step_ratios, step_ratio))
            mdat['step_ratios'] = all_step_ratios
            mdat['samples'] = samples
            mdat['data'] = data
            super(sampler, self).save(mdat, savefile)

            # samples_old = samples_new
            MYsamples_old = samples_new

        # collect everything
        MYsamples = samples
        MYdata = data
        MYall_step_ratios = all_step_ratios
        # ``parameter_samples`` is np.ndarray of shape (num_samples, ndim)
        samples = np.empty((num_samples, np.shape(MYsamples)[1], dtype=np.float64)
        # and ``data_samples`` is np.ndarray of shape (num_samples, mdim)
        data = np.empty((num_samples, np.shape(MYdata)[1]), dtype=np.float64)
        all_step_ratios = np.empty((self.num_chains, self.chain_length), dtype=np.float64)
        # now allgather
        comm.Allgather([MYsamples, MPI.DOUBLE], [samples, MPI.DOUBLE])
        comm.Allgather([MYdata, MPI.DOUBLE], [data, MPI.DOUBLE])
        comm.Allgather([MYall_step_ratios, MPI.DOUBLE], [all_step_ratios, MPI.DOUBLE])

        # save everything
        mdat['step_ratios'] = all_step_ratios
        mdat['samples'] = samples
        mdat['data'] = data
        super(sampler, self).save(mdat, savefile)

        return (samples, data, all_step_ratios)

