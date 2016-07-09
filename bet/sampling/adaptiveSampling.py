# Copyright (C) 2014-2016 The BET Development Team

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
import bet.util as util
import math, os, glob, logging
from bet.Comm import comm 
import bet.sample as sample

def loadmat(save_file, lb_model=None):
    """
    Loads data from ``save_file`` into a
    :class:`~bet.sampling.adaptiveSampling.sampler` object.
    
    :param string save_file: file name
    :param lb_model: runs the model at a given set of parameter samples, (N,
        ndim), and returns data (N, mdim)
    
    :rtype: tuple
    :returns: (sampler, discretization)
    
    """
    # load the data from a *.mat file
    mdat = sio.loadmat(save_file)
    # load the discretization
    discretization = sample.load_discretization(save_file)
    num_samples = np.squeeze(mdat['num_samples'])
    # recreate the sampler
    new_sampler = sampler(num_samples,
            np.squeeze(mdat['chain_length']), lb_model)
    return (new_sampler, discretization)

class sampler(bsam.sampler):
    """
    This class provides methods for adaptive sampling of parameter space to
    provide samples to be used by algorithms to solve inverse problems. 
    
    """
    def __init__(self, num_samples, chain_length, lb_model):
        """
        
        Initialization

        :param int num_samples: total number of samples
        :param int chain_length: number of batches of samples
        :param callable lb_model: runs the model at a given set of parameter
            samples, (N, ndim), and returns data (N, mdim)
        
        """
        super(sampler, self).__init__(lb_model, num_samples)
        #: number of batches of samples
        self.chain_length = chain_length
        #: number of samples per processor per batch (either a single int or a
        #:  list of int) 
        self.num_chains_pproc = int(math.ceil(num_samples/\
                    float(chain_length*comm.size)))
        #: number of samples per batch (either a single int or a list of int)
        self.num_chains = comm.size * self.num_chains_pproc
        #: Total number of samples
        self.num_samples = chain_length * self.num_chains
        #: runs the model at a given set of parameter samples, (N,
        #:    ndim), and returns data (N, mdim)
        self.lb_model = lb_model
        #: batch number for this particular chain 
        self.sample_batch_no = np.repeat(range(self.num_chains), chain_length,
                0)

    def update_mdict(self, mdict):
        """
        Set up references for ``mdict``

        :param dict mdict: dictonary of sampler parameters

        """
        super(sampler, self).update_mdict(mdict)
        mdict['chain_length'] = self.chain_length
        mdict['num_chains'] = self.num_chains
        mdict['sample_batch_no'] = self.sample_batch_no
        
    def run_gen(self, kern_list, rho_D, maximum, input_domain,
            t_set, savefile, initial_sample_type="lhs", criterion='center'):
        """
        Generates samples using generalized chains and a list of different
        kernels.

        :param list kern_list: List of
            :class:~`bet.sampling.adaptiveSampling.kernel` objects.
        :param rho_D: probability density on D
        :type rho_D: callable function that takes a :class:`numpy.ndarray` and
            returns a :class:`numpy.ndarray`
        :param float maximum: maximum value of rho_D
        :param input_domain: min, max value for each input dimension
        :type input_domain: :class:`numpy.ndarray` (ndim, 2)
        :param t_set: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_set: :class:`bet.sampling.adaptiveSampling.transition_set`
        :param string savefile: filename to save samples and data
        :param string initial_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        
        :rtype: tuple
        :returns: (discretization, , num_high_prob_samples,
            sorted_incidices_of_num_high_prob_samples, average_step_ratio)
        
        """
        # generalized chains
        results = list()
        r_step_size = list()
        results_rD = list()
        mean_ss = list()
        for kern in kern_list:
            (discretization, step_sizes) = self.generalized_chains(
                    input_domain, t_set, kern, savefile,
                    initial_sample_type, criterion)
            results.append(discretization)
            r_step_size.append(step_sizes)
            results_rD.append(int(sum(rho_D(discretization._output_sample_set.\
                    get_values())/maximum)))
            mean_ss.append(np.mean(step_sizes))
        sort_ind = np.argsort(results_rD)
        return (results, r_step_size, results_rD, sort_ind, mean_ss)

    def run_tk(self, init_ratio, min_ratio, max_ratio, rho_D, maximum,
            input_domain, kernel, savefile,
            initial_sample_type="lhs", criterion='center'):
        """
        Generates samples using generalized chains and
        :class:`~bet.sampling.transition_set` created using
        the `init_ratio`, `min_ratio`, and `max_ratio` parameters.
    
        :param list init_ratio: Initial step size ratio compared to the
            parameter domain.
        :param list min_ratio: Minimum step size compared to the initial step
            size.
        :param list max_ratio: Maximum step size compared to the maximum step
            size.
        :param rho_D: probability density on D
        :type rho_D: callable function that takes a :class:`numpy.ndarray` and
            returns a :class:`numpy.ndarray`
        :param float maximum: maximum value of rho_D
        :param input_domain: min, max value for each input dimension
        :type input_domain: :class:`numpy.ndarray` (ndim, 2)
        :param kernel: functional that acts on the data used to
            determine the proposed change to the ``step_size``
        :type kernel: :class:`bet.sampling.adaptiveSampling.kernel` object.
        :param string savefile: filename to save samples and data
        :param string initial_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        
        :rtype: tuple
        :returns: (discretization, , num_high_prob_samples,
            sorted_incidices_of_num_high_prob_samples, average_step_ratio)
        
        """
        results = list()
        r_step_size = list()
        results_rD = list()
        mean_ss = list()
        for i, j, k  in zip(init_ratio, min_ratio, max_ratio):
            ts = transition_set(i, j, k)
            (discretization, step_sizes) = self.generalized_chains(
                   input_domain, ts, kernel, savefile,
                    initial_sample_type, criterion)
            results.append(discretization)
            r_step_size.append(step_sizes)
            results_rD.append(int(sum(rho_D(discretization._output_sample_set.\
                    get_values())/maximum)))
            mean_ss.append(np.mean(step_sizes))
        sort_ind = np.argsort(results_rD)
        return (results, r_step_size, results_rD, sort_ind, mean_ss)

    def run_inc_dec(self, increase, decrease, tolerance, rho_D, maximum,
            input_domain, t_set, savefile,
            initial_sample_type="lhs", criterion='center'):
        """
        Generates samples using generalized chains and
        :class:`~bet.sampling.adaptiveSampling.rhoD_kernel` created using
        the `increase`, `decrease`, and `tolerance` parameters.

        :param list increase: the multiple to increase the step size by
        :param list decrease: the multiple to decrease the step size by
        :param list tolerance: a tolerance used to determine if two
            different values are close
        :param rho_D: probability density on D
        :type rho_D: callable function that takes a :class:`numpy.ndarray` and
            returns a :class:`numpy.ndarray`
        :param float maximum: maximum value of rho_D
        :param input_domain: min, max value for each input dimension
        :type input_domain: :class:`numpy.ndarray` (ndim, 2)
        :param t_set: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_set: :class:`bet.sampling.adaptiveSampling.transition_set`
        :param string savefile: filename to save samples and data
        :param string initial_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        
        :rtype: tuple
        :returns: (discretization, , num_high_prob_samples,
            sorted_incidices_of_num_high_prob_samples, average_step_ratio)
        
        """
        kern_list = list()
        for i, j, z in zip(increase, decrease, tolerance):
            kern_list.append(rhoD_kernel(maximum, rho_D, i, j, z)) 
        return self.run_gen(kern_list, rho_D, maximum, input_domain,
                t_set, savefile, initial_sample_type, criterion)

    def generalized_chains(self, input_obj, t_set, kern,
            savefile, initial_sample_type="random", criterion='center',
            hot_start=0): 
        """
        Basic adaptive sampling algorithm using generalized chains.

        .. todo::

            Test HOTSTART from parallel files using different num proc

        :param string initial_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param input_obj: Either a :class:`bet.sample.sample_set` object for an
            input space, an array of min and max bounds for the input values
            with ``min = input_domain[:, 0]`` and ``max = input_domain[:, 1]``,
            or the dimension of an input space
        :type input_obj: :class:`~bet.sample.sample_set`,
            :class:`numpy.ndarray` of shape (ndim, 2), or :class: `int`
        :param t_set: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_set: :class:`bet.sampling.adaptiveSampling.transition_set`
        :param kern: functional that acts on the data used to
            determine the proposed change to the ``step_size``
        :type kernel: :class:~`bet.sampling.adaptiveSampling.kernel` object.
        :param string savefile: filename to save samples and data
        :param int hot_start: Flag whether or not hot start the sampling
            chains from a previous set of chains. Note that ``num_chains`` must
            be the same, but ``num_chains_pproc`` need not be the same. 0 -
            cold start, 1 - hot start from uncompleted run, 2 - hot
            start from finished run
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        
        :rtype: tuple
        :returns: (``discretization``, ``all_step_ratios``) where
            ``discretization`` is a :class:`~bet.sample.discretization` object
            containing ``num_samples``  and  ``all_step_ratios`` is np.ndarray
            of shape ``(num_chains, chain_length)``
        
        """

        # Calculate step_size
        max_ratio = t_set.max_ratio
        min_ratio = t_set.min_ratio

        if not hot_start:
            logging.info("COLD START")
            step_ratio = t_set.init_ratio*np.ones(self.num_chains_pproc)
           
            # Initiative first batch of N samples (maybe taken from latin
            # hypercube/space-filling curve to fully explore parameter space -
            # not necessarily random). Call these Samples_old.
            disc_old = super(sampler, self).create_random_discretization(
                    initial_sample_type, input_obj, savefile,
                    self.num_chains, criterion, globalize=False)
            self.num_samples = self.chain_length * self.num_chains
            comm.Barrier()
            
            # populate local values 
            #disc_old._input_sample_set.global_to_local()
            #disc_old._output_sample_set.global_to_local()
            input_old = disc_old._input_sample_set.copy()
            
            disc = disc_old.copy()
            all_step_ratios = step_ratio 

            (kern_old, proposal) = kern.delta_step(disc_old.\
                    _output_sample_set.get_values_local(), None)

            start_ind = 1
        if hot_start:
            # LOAD FILES
            if hot_start == 1: # HOT START FROM PARTIAL RUN
                if comm.rank == 0:
                    logging.info("HOT START from partial run")
                # Find and open save files
                save_dir = os.path.dirname(savefile)
                base_name = os.path.dirname(savefile)
                mdat_files = glob.glob(os.path.join(save_dir,
                        "proc*_{}".format(base_name)))
                if len(mdat_files) == 0:
                    logging.info("HOT START using serial file")
                    mdat = sio.loadmat(savefile)
                    disc = sample.load_discretization(savefile)
                    kern_old = np.squeeze(mdat['kern_old'])
                    all_step_ratios = np.squeeze(mdat['step_ratios'])
                    chain_length = disc.check_nums()/self.num_chains
                    if all_step_ratios.shape == (self.num_chains,
                                                        chain_length):
                        msg = "Serial file, from completed"
                        msg += " run updating hot_start"
                        hot_start = 2
                    # reshape if parallel
                    if comm.size > 1:
                        temp_input = np.reshape(disc._input_sample_set.\
                                get_values(), (self.num_chains,
                                    chain_length, -1), 'F')
                        temp_output = np.reshape(disc._output_sample_set.\
                                get_values(), (self.num_chains,
                                    chain_length, -1), 'F')
                        all_step_ratios = np.reshape(all_step_ratios,
                                 (self.num_chains, -1), 'F')
                elif hot_start == 1 and len(mdat_files) == comm.size:
                    logging.info("HOT START using parallel files (same nproc)")
                    # if the number of processors is the same then set mdat to
                    # be the one with the matching processor number (doesn't
                    # really matter)
                    mdat = sio.loadmat(mdat_files[comm.rank])
                    disc = sample.load_discretization(mdat_files[comm.rank])
                    kern_old = np.squeeze(mdat['kern_old'])
                    all_step_ratios = np.squeeze(mdat['step_ratios'])
                elif hot_start == 1 and len(mdat_files) != comm.size:
                    logging.info("HOT START using parallel files (diff nproc)")
                    # Determine how many processors the previous data used
                    # otherwise gather the data from mdat and then scatter
                    # among the processors and update mdat
                    mdat_files_local = comm.scatter(mdat_files)
                    mdat_local = [sio.loadmat(m) for m in mdat_files_local]
                    disc_local = [sample.load_discretization(m) for m in\
                            mdat_files_local]
                    mdat_list = comm.allgather(mdat_local)
                    disc_list = comm.allgather(disc_local)
                    mdat_global = []
                    disc_global = []
                    # instead of a list of lists, create a list of mdat
                    for mlist, dlist in zip(mdat_list, disc_list): 
                        mdat_global.extend(mlist)
                        disc_global.extend(dlist)
                    # get num_proc and num_chains_pproc for previous run
                    old_num_proc = max((len(mdat_list), 1))
                    old_num_chains_pproc = self.num_chains/old_num_proc
                    # get batch size and/or number of dimensions
                    chain_length = disc_global[0].check_nums()/\
                            old_num_chains_pproc
                    disc = disc_global[0].copy()
                    # create lists of local data
                    temp_input = []
                    temp_output = []
                    all_step_ratios = []
                    kern_old = []
                    # RESHAPE old_num_chains_pproc, chain_length(or batch), dim
                    for mdat, disc_local in zip(mdat_global, disc_local):
                        temp_input.append(np.reshape(disc_local.\
                                _input_sample_set.get_values_local(),
                                (old_num_chains_pproc, chain_length, -1), 'F'))
                        temp_output.append(np.reshape(disc_local.\
                                _output_sample_set.get_values_local(),
                                (old_num_chains_pproc, chain_length, -1), 'F'))
                        all_step_ratios.append(np.reshape(mdat['step_ratios'],
                            (old_num_chains_pproc, chain_length, -1), 'F'))
                        kern_old.append(np.reshape(mdat['kern_old'],
                            (old_num_chains_pproc,), 'F'))
                    # turn into arrays
                    temp_input = np.concatenate(temp_input)
                    temp_output = np.concatenate(temp_output)
                    all_step_ratios = np.concatenate(all_step_ratios)
                    kern_old = np.concatenate(kern_old)
            if hot_start == 2: # HOT START FROM COMPLETED RUN:
                if comm.rank == 0:
                    logging.info("HOT START from completed run")
                mdat = sio.loadmat(savefile)
                disc = sample.load_discretization(savefile)
                kern_old = np.squeeze(mdat['kern_old'])
                all_step_ratios = np.squeeze(mdat['step_ratios'])
                chain_length = disc.check_nums()/self.num_chains
                # reshape if parallel
                if comm.size > 1:
                    temp_input = np.reshape(disc._input_sample_set.\
                                get_values(), (self.num_chains, chain_length,
                                    -1), 'F')
                    temp_output = np.reshape(disc._output_sample_set.\
                                get_values(), (self.num_chains, chain_length,
                                    -1), 'F')
                    all_step_ratios = np.reshape(all_step_ratios,
                            (self.num_chains, chain_length), 'F')
            # SPLIT DATA IF NECESSARY
            if comm.size > 1 and (hot_start == 2 or (hot_start == 1 and \
                    len(mdat_files) != comm.size)):
                # Use split to split along num_chains and set *._values_local
                disc._input_sample_set.set_values_local(np.reshape(np.split(\
                        temp_input, comm.size, 0)[comm.rank],
                        (self.num_chains_pproc*chain_length, -1), 'F'))
                disc._output_sample_set.set_values_local(np.reshape(np.split(\
                        temp_output, comm.size, 0)[comm.rank],
                        (self.num_chains_pproc*chain_length, -1), 'F'))
                all_step_ratios = np.reshape(np.split(all_step_ratios,
                    comm.size, 0)[comm.rank],
                    (self.num_chains_pproc*chain_length,), 'F')
                kern_old = np.reshape(np.split(kern_old, comm.size,
                    0)[comm.rank], (self.num_chains_pproc,), 'F')
            else:
                all_step_ratios = np.reshape(all_step_ratios, (-1,), 'F')
            # MAKE SURE ARRAYS ARE LOCALIZED FROM HERE ON OUT WILL ONLY
            # OPERATE ON _local_values
            # Set mdat, step_ratio, input_old, start_ind appropriately
            step_ratio = all_step_ratios[-self.num_chains_pproc:]
            input_old = sample.sample_set(disc._input_sample_set.get_dim())
            input_old.set_domain(disc._input_sample_set.get_domain())
            input_old.set_values_local(disc._input_sample_set.\
                    get_values_local()[-self.num_chains_pproc:, :])

            # Determine how many batches have been run
            start_ind = disc._input_sample_set.get_values_local().\
                    shape[0]/self.num_chains_pproc
        
        mdat = dict()
        self.update_mdict(mdat)
        input_old.update_bounds_local()

        for batch in xrange(start_ind, self.chain_length):
            # For each of N samples_old, create N new parameter samples using
            # transition set and step_ratio. Call these samples input_new.
            input_new = t_set.step(step_ratio, input_old)
        
            # Solve the model for the input_new.
            output_new_values = self.lb_model(input_new.get_values_local())
            
            # Make some decision about changing step_size(k).  There are
            # multiple ways to do this.
            # Determine step size
            (kern_old, proposal) = kern.delta_step(output_new_values, kern_old)
            step_ratio = proposal*step_ratio
            # Is the ratio greater than max?
            step_ratio[step_ratio > max_ratio] = max_ratio
            # Is the ratio less than min?
            step_ratio[step_ratio < min_ratio] = min_ratio

            # Save and export concatentated arrays
            if self.chain_length < 4:
                pass
            elif comm.rank == 0 and (batch+1)%(self.chain_length/4) == 0:
                logging.info("Current chain length: "+\
                            str(batch+1)+"/"+str(self.chain_length))
            disc._input_sample_set.append_values_local(input_new.\
                    get_values_local())
            disc._output_sample_set.append_values_local(output_new_values)
            all_step_ratios = np.concatenate((all_step_ratios, step_ratio))
            mdat['step_ratios'] = all_step_ratios
            mdat['kern_old'] = kern_old
            
            super(sampler, self).save(mdat, savefile, disc, globalize=False)
            input_old = input_new

        # collect everything
        disc._input_sample_set.update_bounds_local() 
        #disc._input_sample_set.local_to_global()
        #disc._output_sample_set.local_to_global()

        MYall_step_ratios = np.copy(all_step_ratios) 
        # ``all_step_ratios`` is np.ndarray of shape (num_chains,
        # chain_length)
        all_step_ratios = util.get_global_values(MYall_step_ratios,
                shape=(self.num_samples,))
        all_step_ratios = np.reshape(all_step_ratios, (self.num_chains,
            self.chain_length), 'F')

        # save everything
        mdat['step_ratios'] = all_step_ratios
        mdat['kern_old'] = util.get_global_values(kern_old,
                shape=(self.num_chains,))
        super(sampler, self).save(mdat, savefile, disc, globalize=True)

        return (disc, all_step_ratios)
        
def kernels(Q_ref, rho_D, maximum):
    """
    Generates a list of kernstic objects.
    
    :param Q_ref: reference parameter value
    :type Q_ref: :class:`numpy.ndarray`
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`numpy.ndarray` and
        returns a :class:`numpy.ndarray`
    :param float maximum: maximum value of rho_D
    
    :rtype: list
    :returns: [maxima_mean_kernel, rhoD_kernel, maxima_kernel]
    
    """
    kern_list = list()
    kern_list.append(maxima_mean_kernel(np.array([Q_ref]), rho_D))
    kern_list.append(rhoD_kernel(maximum, rho_D))
    kern_list.append(maxima_kernel(np.array([Q_ref]), rho_D))
    return kern_list

class transition_set(object):
    """
    Basic class that is used to create a step to move from samples_old to
    input_new based. This class generates steps for a random walk using a
    very basic algorithm. Future classes will inherit from this one with
    different implementations of the
    :meth:~`polysim.run_framework.apdative_sampling.step` method.
    This basic transition set is designed without a preferential direction.
    
    """

    def __init__(self, init_ratio, min_ratio, max_ratio):
        """
        Initialization

        :param float init_ratio: initial step ratio
        :param float min_ratio: minimum step_ratio
        :param float max_ratio: maximum step_ratio

        """
        #: float, initial step ratio
        self.init_ratio = init_ratio
        #: float,  minimum step_ratio
        self.min_ratio = min_ratio
        #: float, maximum step_ratio
        self.max_ratio = max_ratio
    
    def step(self, step_ratio, input_old): 
        """
        Generate ``num_samples`` new steps using ``step_ratio`` and
        ``input_width`` to calculate the ``step size``. Each step will have a
        random direction.
        
        :param step_ratio: define maximum step_size = ``step_ratio*input_width``
        :type step_ratio: :class:`numpy.ndarray` of shape (num_samples,)
        :param input_old: Input from the previous step.
        :type input_old: :class:`~numpy.ndarray` of shape (num_samples,
            ndim)
        
        :rtype: :class:`numpy.ndarray` of shape (num_samples, ndim)
        :returns: input_new
        
        """
        # calculate maximum step size
        step_size = np.repeat([step_ratio], input_old.get_dim(),
                0).transpose()*input_old._width_local
        # check to see if step will take you out of parameter space
        # calculate maximum proposed step
        my_right = input_old.get_values_local() + 0.5*step_size
        my_left = input_old.get_values_local() - 0.5*step_size
        # Is the new sample greaters than the right limit?
        far_right = my_right >= input_old._right_local
        far_left = my_left <= input_old._left_local
        # If the input could leave the domain then truncate the box defining
        # the step_size
        my_right[far_right] = input_old._right_local[far_right]
        my_left[far_left] = input_old._left_local[far_left]
        my_width = my_right-my_left
        #input_center = (input_right+input_left)/2.0
        input_new_values = my_width * np.random.random(input_old.shape_local())
        input_new_values = input_new_values + my_left
        input_new = input_old.copy()
        input_new.set_values_local(input_new_values)
        return input_new

class kernel(object):
    """
    Parent class for kernels to determine change in step size. This class
    provides a method for determining the proposed change in step size. Since
    this is simply a skeleton parent class it does not change the step size at
    all.
    
    """

    def __init__(self, tolerance=1E-08, increase=1.0, decrease=1.0):
        """
        Initialization

        :param float tolerance: Tolerance for comparing two values
        :param float increase: The multiple to increase the step size by
        :param float decrease: The multiple to decrease the step size by

        """
        #: float, Tolerance for comparing two values
        self.TOL = tolerance
        #: float,  The multiple to increase the step size by
        self.increase = increase
        #: float, The multiple to decrease the step size by
        self.decrease = decrease

    def delta_step(self, output_new, kern_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param output_new: QoI for a given batch of samples 
        :type output_new: :class:`numpy.ndarray` of shape (num_chains, mdim)
        :param kern_old: kernel evaluated at previous step
        
        :rtype: typle
        :returns: (kern_new, proposal)
        
        """
        return (kern_old, np.ones((output_new.shape[0],)))

class rhoD_kernel(kernel):
    """
    We assume we know the distribution rho_D on the QoI and that the goal is to
    determine inverse regions of high probability accurately (in terms of
    getting the measure correct). This class provides a method for determining
    the proposed change in step size as follows. We check if the QoI at each of
    the input_new(k) are closer or farther away from a region of high
    probability in D than the QoI at samples_old(k).  For example, if they are
    closer, then we can reduce the step_size(k) by 1/2.
    Note: This only works well with smooth rho_D.

    """

    def __init__(self, maximum, rho_D, tolerance=1E-08, increase=2.0, 
            decrease=0.5):
        """
        Initialization

        :param float maximum: maximum value of rho_D
        :param function rho_D: probability density on D
        :param float tolerance: Tolerance for comparing two values
        :param float increase: The multiple to increase the step size by
        :param float decrease: The multiple to decrease the step size by

        """
        #: float, maximum value of rho_D
        self.MAX = maximum
        #: callable, function, probability density on D
        self.rho_D = rho_D
        #: bool, flag sort order
        self.sort_ascending = False
        super(rhoD_kernel, self).__init__(tolerance, increase, decrease)

    def delta_step(self, output_new, kern_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param output_new: QoI for a given batch of samples 
        :type output_new: :class:`numpy.ndarray` of shape (num_chains, mdim)
        :param kern_old: kernel evaluated at previous step
        
        :rtype: tuple
        :returns: (kern_new, proposal)
        
        """
        # Evaluate kernel for new data.
        kern_new = self.rho_D(output_new)
        
        if kern_old is None:
            return (kern_new, None)
        else:
            kern_diff = (kern_new-kern_old)/self.MAX
            # Compare to kernel for old data.
            # Is the kernel NOT close?
            kern_close = np.logical_not(np.isclose(kern_diff, 0,
                atol=self.TOL))
            kern_max = np.isclose(kern_new, self.MAX, atol=self.TOL)
            # Is the kernel greater/lesser?
            kern_greater = np.logical_and(kern_diff > 0, kern_close)
            kern_greater = np.logical_or(kern_greater, kern_max)
            kern_lesser = np.logical_and(kern_diff < 0, kern_close)

            # Determine step size
            proposal = np.ones(kern_new.shape)
            proposal[kern_greater] = self.decrease
            proposal[kern_lesser] = self.increase
            return (kern_new, proposal.transpose())


class maxima_kernel(kernel):
    """
    We assume we know the maxima of the distribution rho_D on the QoI and that
    the goal is to determine inverse regions of high probability accurately (in
    terms of getting the measure correct). This class provides a method for
    determining the proposed change in step size as follows. We check if the
    QoI at each of the input_new(k) are closer or farther away from a region
    of high probability in D than the QoI at samples_old(k). For example, if
    they are closer, then we can reduce the step_size(k) by 1/2.

    """

    def __init__(self, maxima, rho_D, tolerance=1E-08, increase=2.0, 
            decrease=0.5):
        """
        Initialization
        
        :param maxima: locations of the maxima of rho_D on D 
        :type maxima: :class:`numpy.ndarray` of chape (num_maxima, mdim)
        :param rho_D: probability density on D
        :type rho_D: callable function that takes a :class:`numpy.ndarray` and
            returns a class:`numpy.ndarray`
        :param float tolerance: Tolerance for comparing two values
        :param float increase: The multiple to increase the step size by
        :param float decrease: The multiple to decrease the step size by

        """
        #: locations of the maxima of rho_D on D
        self.MAXIMA = maxima
        #: int, number of maxima
        self.num_maxima = maxima.shape[0]
        #: list of maximum values of rho_D
        self.rho_max = rho_D(maxima)
        super(maxima_kernel, self).__init__(tolerance, increase, decrease)
        #: bool, flag sort order
        self.sort_ascending = True

    def delta_step(self, output_new, kern_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param output_new: QoI for a given batch of samples 
        :type output_new: :class:`numpy.ndarray` of shape (num_chains, mdim)
        :param kern_old: kernel evaluated at previous step
        
        :rtype: tuple
        :returns: (kern_new, proposal)
        
        """
        # Evaluate kernel for new data.
        kern_new = np.zeros((output_new.shape[0]))

        for i in xrange(output_new.shape[0]):
            # calculate distance from each of the maxima
            vec_from_maxima = np.repeat([output_new[i, :]], self.num_maxima, 0)
            vec_from_maxima = vec_from_maxima - self.MAXIMA
            # weight distances by 1/rho_D(maxima)
            dist_from_maxima = np.linalg.norm(vec_from_maxima, 2,
                1)/self.rho_max
            # set kern_new to be the minimum of weighted distances from maxima
            kern_new[i] = np.min(dist_from_maxima)
        
        if kern_old is None:
            return (kern_new, None)
        else:
            kern_diff = (kern_new-kern_old)
            # Compare to kernel for old data.
            # Is the kernel NOT close?
            kern_close = np.logical_not(np.isclose(kern_diff, 0,
                atol=self.TOL))
            # Is the kernel greater/lesser?
            kern_greater = np.logical_and(kern_diff > 0, kern_close)
            kern_lesser = np.logical_and(kern_diff < 0, kern_close)
            # Determine step size
            proposal = np.ones(kern_new.shape)
            # if further than kern_old then increase
            proposal[kern_greater] = self.increase
            # if closer than kern_old then decrease
            proposal[kern_lesser] = self.decrease
        return (kern_new, proposal)


class maxima_mean_kernel(maxima_kernel):
    """
    We assume we know the maxima of the distribution rho_D on the QoI and that
    the goal is to determine inverse regions of high probability accurately (in
    terms of getting the measure correct). This class provides a method for
    determining the proposed change in step size as follows. We check if the
    QoI at each of the input_new(k) are closer or farther away from a region
    of high probability in D than the QoI at samples_old(k). For example, if
    they are closer, then we can reduce the step_size(k) by 1/2.
    
    """

    def __init__(self, maxima, rho_D, tolerance=1E-08, increase=2.0, 
            decrease=0.5):
        """
        Initialization
        
        :param maxima: locations of the maxima of rho_D on D 
        :type maxima: :class:`numpy.ndarray` of chape (num_maxima, mdim)
        :param rho_D: probability density on D
        :type rho_D: callable function that takes a :class:`numpy.ndarray` and
            returns a class:`numpy.ndarray`
        :param float tolerance: Tolerance for comparing two values
        :param float increase: The multiple to increase the step size by
        :param float decrease: The multiple to decrease the step size by

        """
        #: approximate radius
        self.radius = None
        #: approximate mean
        self.mean = None
        #: current number of estimates for approx. mean, radius
        self.current_clength = 0
        super(maxima_mean_kernel, self).__init__(maxima, rho_D, tolerance,
                increase, decrease)

    def reset(self):
        """
        Resets the the batch number and the estimates of the mean and maximum
        distance from the mean.
        """
        self.radius = None
        self.mean = None
        self.current_clength = 0

    def delta_step(self, output_new, kern_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param output_new: QoI for a given batch of samples 
        :type output_new: :class:`numpy.ndarray` of shape (num_chains, mdim)
        :param kern_old: kernel evaluated at previous step
        
        :rtype: tuple
        :returns: (kern_new, proposal)
        
        """
        # Evaluate kernel for new data.
        kern_new = np.zeros((output_new.shape[0]))
        self.current_clength = self.current_clength + 1

        for i in xrange(output_new.shape[0]):
            # calculate distance from each of the maxima
            vec_from_maxima = np.repeat([output_new[i, :]], self.num_maxima, 0)
            vec_from_maxima = vec_from_maxima - self.MAXIMA
            # weight distances by 1/rho_D(maxima)
            dist_from_maxima = np.linalg.norm(vec_from_maxima, 2,
                1)/self.rho_max
            # set kern_new to be the minimum of weighted distances from maxima
            kern_new[i] = np.min(dist_from_maxima)
        if kern_old is None:
            # calculate the mean
            self.mean = np.mean(output_new, 0)
            # calculate the distance from the mean
            vec_from_mean = output_new - np.repeat([self.mean],
                    output_new.shape[0], 0)
            # estimate the radius of D
            self.radius = np.max(np.linalg.norm(vec_from_mean, 2, 1))
            return (kern_new, None)
        else:
            # update the estimate of the mean
            self.mean = (self.current_clength-1)*self.mean + np.mean(output_new,
                    0) 
            self.mean = self.mean / self.current_clength
            # calculate the distance from the mean
            vec_from_mean = output_new - np.repeat([self.mean],
                    output_new.shape[0], 0)
            # esitmate the radius of D
            self.radius = max(np.max(np.linalg.norm(vec_from_mean, 2, 1)),
                    self.radius)
            # calculate the relative change in distance
            kern_diff = (kern_new-kern_old)
            # normalize by the radius of D (IF POSSIBLE)
            kern_diff = kern_diff #/ self.radius
            # Compare to kernel for old data.
            # Is the kernel NOT close?
            kern_close = np.logical_not(np.isclose(kern_diff, 0,
                atol=self.TOL))
            # Is the kernel greater/lesser?
            kern_greater = np.logical_and(kern_diff > 0, kern_close)
            kern_lesser = np.logical_and(kern_diff < 0, kern_close)
            # Determine step size
            proposal = np.ones(kern_new.shape)
            # if further than kern_old then increase
            proposal[kern_greater] = self.increase
            # if closer than kern_old then decrease
            proposal[kern_lesser] = self.decrease
        return (kern_new, proposal)
