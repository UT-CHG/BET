# Copyright (C) 2014-2015 The BET Development Team

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
import bet.sensitivity.gradients as grad
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
        super(sampler, self).__init__(num_samples, chain_length, lb_model)
        self.chain_length = chain_length
        self.num_chains_pproc = int(math.ceil(num_samples/float(chain_length*size)))
        self.num_chains = size * self.num_chains_pproc
        self.num_samples = chain_length * self.num_chains
        self.lb_model = lb_model
        self.sample_batch_no = np.repeat(range(self.num_chains), chain_length,
                0)

    def run_newton(self, samples_curr, data_curr, num_centers, radius, Q_ref, lam_domain=None):
        """
        Generates samples using a modified Newtons Method.
        """
        Q_min = Q_ref - 0.1
        Q_max = Q_ref + 0.1
        Lambda_dim = samples_curr.shape[1]
        num_samples = samples_curr.shape[0]
        centers_curr = samples_curr[:num_centers,:]

        samples = samples_curr
        data = data_curr

        for step in range(self.chain_length):

            if (data_curr[data_curr[data_curr<Q_max]>Q_min]).size:
                print 'Found one in region.'

            G = grad.calculate_gradients_rbf(samples_curr, data_curr, centers_curr, normalized=False)
            normG = np.linalg.norm(G, axis=2)

            step_size = (Q_ref-data_curr[:num_centers])/normG**2

            centers_curr = centers_curr + step_size*G[:,0,:]
     
            samples_curr = grad.sample_l1_ball(centers_curr, Lambda_dim+1, radius)

            data_curr = self.lb_model(samples_curr)

            samples = np.append(samples, samples_curr, axis=0)
            data = np.append(data, data_curr, axis=0)

        return (samples, data)





