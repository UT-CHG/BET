# Copyright (C) 2014-2015 The BET Development Team

# Lindley Graham 4/15/2014
"""
This module contains functions for sampling. We assume we are given access to a
model, a parameter space, and a data space. The model is a map from the
paramter space to the data space. We desire to build up a set of samples to
sovle an inverse problem this guving use information about the inverse mapping.
Each sample consists for a paramter coordinate, data coordinate pairing. We
assume the measure on both spaces in Lebesgue.
"""

import numpy as np
import scipy.io as sio
from pyDOE import lhs
import bet.util as util
from bet.Comm import *

def loadmat(save_file, model=None):
    """
    Loads data from ``save_file`` into a
    :class:`~bet.basicSampling.sampler` object.

    :param string save_file: file name
    :param model: runs the model at a given set of parameter samples and
        returns data 
    :rtype: tuple
    :returns: (sampler, samples, data)

    """
    # load the data from a *.mat file
    mdat = sio.loadmat(save_file)
    # load the samples
    if mdat.has_key('samples'):
        samples = mdat['samples']
        num_samples = samples.shape[0]
    else:
        samples = None
        num_samples = None
    # load the data
    if mdat.has_key('data'):
        data = mdat['data']
    else:
        data = None
    loaded_sampler = sampler(model, num_samples)    
    return (loaded_sampler, samples, data)

class sampler(object):
    """
    This class provides methods for adaptive sampling of parameter space to
    provide samples to be used by algorithms to solve inverse problems. 

    num_samples
        total number of samples OR list of number of samples per dimension such
        that total number of samples is prob(num_samples)
    lb_model
        :class:`~bet.loadBalance.load_balance` runs the model at a given set of
        parameter samples and returns data 
    """
    def __init__(self, lb_model, num_samples=None):
        """
        Initialization
        
        :param lb_model: Interface to physics-based model takes an input of
            shape (N, ndim) and returns an output of shape (N, mdim)
        :param int num_samples: N, number of samples (optional)
        """
        self.num_samples = num_samples
        self.lb_model = lb_model

    def save(self, mdict, save_file):
        """
        Save matrices to a ``*.mat`` file for use by ``MATLAB BET`` code and
        :meth:`~bet.sampling.loadmat`

        :param dict mdict: dictonary of sampling data and sampler parameters
        :param string save_file: file name

        """
        sio.savemat(save_file, mdict, do_compression=True)

    def update_mdict(self, mdict):
        """
        Set up references for ``mdict``

        :param dict mdict: dictonary of sampler parameters

        """
        mdict['num_samples'] = self.num_samples

    def random_samples(self, sample_type, param_min, param_max,
            savefile, num_samples=None, criterion='center', parallel=False):
        """
        Sampling algorithm with three basic options

            * ``random`` (or ``r``) generates ``num_samples`` samples in
                ``lam_domain`` assuming a Lebesgue measure.
            * ``lhs`` generates a latin hyper cube of samples.

        Note: This function is designed only for generalized rectangles and
        assumes a Lebesgue measure on the parameter space.
       
        :param string sample_type: type sampling random (or r),
            latin hypercube(lhs), regular grid (rg), or space-filling
            curve(TBD) 
        :param param_min: minimum value for each parameter dimension
        :type param_min: :class:`numpy.ndarray` (ndim,)
        :param param_max: maximum value for each parameter dimension
        :type param_max: :class:`numpy.ndarray` (ndim,)
        :param string savefile: filename to save samples and data
        :param int num_samples: N, number of samples (optional)
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        :param bool parallel: Flag for parallel implementation. Uses
            lowercase ``mpi4py`` methods if ``samples.shape[0]`` is not
            divisible by ``size``. Default value is ``False``. 
        :rtype: tuple
        :returns: (``parameter_samples``, ``data_samples``) where
            ``parameter_samples`` is np.ndarray of shape (num_samples, ndim)
            and ``data_samples`` is np.ndarray of shape (num_samples, mdim)

        """
        # Create N samples
        if num_samples == None:
            num_samples = self.num_samples
        param_left = np.repeat([param_min], num_samples, 0)
        param_right = np.repeat([param_max], num_samples, 0)
        samples = (param_right-param_left)
         
        if sample_type == "lhs":
            samples = samples * lhs(param_min.shape[-1],
                    num_samples, criterion)
        elif sample_type == "random" or "r":
            samples = samples * np.random.random(param_left.shape) 
        samples = samples + param_left
        return self.user_samples(samples, savefile, parallel)

    def user_samples(self, samples, savefile, parallel=False):
        """
        Samples the model at ``samples`` and saves the results.

        Note: There are many ways to generate samples on a regular grid in
        Numpy and other Python packages. Instead of reimplementing them here we
        provide sampler that utilizes user specified samples.

        :param samples: samples to evaluate the model at
        :type samples: :class:`~numpy.ndarray` of shape (num_smaples, ndim)
        :param string savefile: filename to save samples and data
        :param bool parallel: Flag for parallel implementation. Uses
            lowercase ``mpi4py`` methods if ``samples.shape[0]`` is not
            divisible by ``size``. Default value is ``False``. 
        :rtype: tuple
        :returns: (``parameter_samples``, ``data_samples``) where
            ``parameter_samples`` is np.ndarray of shape (num_samples, ndim)
            and ``data_samples`` is np.ndarray of shape (num_samples, mdim)

        """
        
        # Update the number of samples
        self.num_samples = samples.shape[0]

        # Solve the model at the samples
        if not(parallel) or size == 1:
            data = self.lb_model(samples)
        elif parallel:
            my_len = self.num_samples/size
            if rank != size-1:
                my_index = range(0+rank*my_len, (rank+1)*my_len)
            else:
                my_index = range(0+rank*my_len, self.num_samples)
            if len(samples.shape) == 1:
                my_samples = samples[my_index]
            else:
                my_samples = samples[my_index, :]
            my_data = self.lb_model(my_samples)
            data = util.get_global_values(my_data)
            samples = util.get_global_values(my_samples)
        
        # if data or samples are of shape (num_samples,) expand dimensions
        if len(samples.shape) == 1:
            samples = np.expand_dims(samples, axis=1)
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)


        mdat = dict()
        self.update_mdict(mdat)
        mdat['samples'] = samples
        mdat['data'] = data

        if rank == 0:
            self.save(mdat, savefile)
        
        return (samples, data)


