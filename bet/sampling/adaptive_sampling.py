# -*- coding: utf-8 -*-
# Lindley Graham 3/10/2014
"""
This modules contains functions for adaptive random sampling. We assume we are
given access to a model, a parameter space, and a data space. The model is a
map from the paramter space to the data space. We desire to build up a set of
samples to solve an inverse problem thus giving us information about the
inverse mapping. The each sample consists of a parameter coordinate, data
coordinate pairing. We assume we are given a measure on both spaces.

We employ an approach based on using multiple sample chains in a MCMC type
approach.
"""

import numpy as np
import scipy.io as sio
from polysim.pyADCIRC.basic import pickleable
from pyDOE import lhs
import matplotlib.pyplot as plt

def in_box(data, rho_D, maximum, sample_nos=None):
    """
    Estimates the number of samples in high probability regions of D.

    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`np.ndarray`
    :param list sample_nos: sample numbers to plot

    """
    if sample_nos==None:
        sample_nos = range(data.shape[0])
    rD = rho_D(data[sample_nos,:])
    print "Samples in box "+str(int(sum(rD)/maximum))

def in_box_many(results_list, rho_D, maximum, sample_nos_list=None):
    """
    Estimates the number of samples in high probability regions of D for a list of results.

    :param list results_list: list of (results, data) tuples
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`np.ndarray`
    :param list sample_nos_list: list of sample numbers to plot (list of lists)

    """
    if sample_nos_list:
        for result, sample_nos in zip(results_list, sample_nos_list):
            in_box(result[1], rho_D, maximum, sample_nos)
    else:
        for result in results_list:
            in_box(result[1], rho_D, maximum)

def loadmat(save_file, model = None):
    """
    Loads data from ``save_file`` into a
    :class:`~polysim.run_framework.adaptive_samplers.adaptiveSamples` object.

    :param string save_file: file name
    :param model: runs the model at a given set of parameter samples and returns data
    :rtype: tuple
    :returns: (sampler, samples, data)

    """
    # load the data from a *.mat file
    mdat = sio.loadmat(save_file)
    # load the samples
    if mdat.has_key('samples'):
        samples = mdat['samples']
    else:
        samples = None
    # load the data
    if mdat.has_key('data'):
        data = mdat['data']
    else:
        data = None
    # recreate the sampler
    print "I HAZ BUG AND WILL FAIL"
    sampler = adaptiveSamples(mdat['num_batches'], mdat['samples_per_batch'], model)
    
    return (sampler, samples, data)

class adaptiveSamples(pickleable):
    """
    This class provides methods for adaptive sampling of parameter space to
    provide samples to be used by algorithms to solve inverse problems. 

    num_batches
        number of batches of samples
    samples_per_batch
        number of samples per batch (either a single int or a list of int)
    metric
        metric on the data space, a function d(x,y) where d: M x M --> Real
    model
        runs the model at a given set of parameter samples and returns data
    """
    def __init__(self, num_batches, samples_per_batch, model):
        """
        Initialization
        """
        self.num_batches = num_batches
        self.samples_per_batch = samples_per_batch
        self.model = model
        self.sample_batch_no = np.repeat(range(samples_per_batch), num_batches,
                0)

    def save(self, mdict, save_file):
        """
        Save matrices to a ``*.mat`` file for use by ``MATLAB BET`` code and
        :meth:`~polysim.run_framework.adaptive_sampling.loadmat`

        :param dict() mdict: dictonary of sampling data and sampler parameters
        :param string save_file: file name

        """
        sio.savemat(save_file, mdict, do_compression=True)

    def update_mdict(self, mdict):
        """
        Set up references for ``mdict``

        :param dict() mdict: dictonary of sampler parameters

        """
        mdict['num_batches'] = self.num_batches
        mdict['samples_per_batch'] = self.samples_per_batch
        mdict['sample_batch_no'] = self.sample_batch_no
        
    def show_param_2D(self, samples, data, rho_D = None, p_true=None,
            sample_nos=None, save=True, show=True):
        """
        Plot samples in parameter space and colors them either by rho_D or by
        sample batch number.

        :param samples: Samples to plot
        :type samples: :class:`np.ndarray`
        :param data: Data associated with ``samples``
        :type data: :class:`np.ndarray`
        :param list sample_nos: sample numbers to plot
        :param rho_D: probability density on D
        :type rho_D: callable function that takes a :class:`np.array` and returns a
            :class:`np.ndarray`
        :param p_true: true parameter value
        :type p_true: :class:`np.ndarray`

        """
        if sample_nos==None:
            sample_nos = range(samples.shape[1])
        if rho_D!=None:
            rD = rho_D(data[sample_nos,:])
        else:
            rD = self.sample_batch_no[sample_nos]
        plt.scatter(samples[0,sample_nos],samples[1,sample_nos],c=rD,
                cmap=plt.cm.Oranges_r)
        plt.colorbar()
        if p_true != None:
            plt.scatter(p_true[0], p_true[1], c='b')
        if save:
            plt.autoscale(tight=True)
            plt.xlabel(r'$\lambda_1$')
            plt.ylabel(r'$\lambda_2$')
            plt.savefig('param_samples_cs.eps',
                    bbox_inches='tight', transparent=True, pad_inches=0)
        if show:
            plt.show()
        else:
            plt.close()

    def show_data_2D(self, data, rho_D=None, Q_true=None, sample_nos=None,
            save=True, show=True):
        """
        Plot samples in data space and colors them either by rho_D or by
        sample batch number.

        :param data: Data associated with ``samples``
        :type data: :class:`np.ndarray`
        :param list sample_nos: sample numbers to plot
        :param rho_D: probability density on D
        :type rho_D: callable function that takes a :class:`np.array` and returns a
            :class:`np.ndarray`
        :param Q_true: true parameter value
        :type Q_true: :class:`np.ndarray`

        """   
        if sample_nos==None:
            sample_nos = range(data.shape[0])
        if rho_D!=None:
            rD = rho_D(data[sample_nos,:])
        else:
            rD = self.sample_batch_no[sample_nos]
        plt.scatter(data[sample_nos,0], data[sample_nos,1],c=rD,
                cmap=plt.cm.Oranges_r)
        plt.colorbar()
        if Q_true != None:
            plt.scatter(Q_true[0], Q_true[1], c='b')
        if save:
            plt.autoscale(tight=True)
            plt.xlabel(r'$q_1$')
            plt.ylabel(r'$q_6$')
            plt.savefig('data_samples_cs.eps',
                    bbox_inches='tight', transparent=True, pad_inches=0)
        if show:
            plt.show()
        else:
            plt.close()

    def generalized_chains(self, inital_sample_type, param_min, param_max,
            t_kernel, heuristic, savefile, criterion='center'):
        """
        Basic adaptive sampling algorithm.
       
        :param string inital_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param param_min: minimum value for each parameter dimension
        :type param_min: np.array (ndim,)
        :param param_max: maximum value for each parameter dimension
        :type param_max: np.array (ndim,)
        :param t_kernel: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_kernel: :class:~`t_kernel`
        :param function heuristic: functional that acts on the data used to
            determine the proposed change to the ``step_size``
        :param string savefile: filename to save samples and data
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        :rtype: tuple
        :returns: (``parameter_samples``, ``data_samples``) where
            ``parameter_samples`` is np.ndarray of shape (ndim, num_samples)
            and ``data_samples`` is np.ndarray of shape (num_samples, mdim)

        """
        # Initialize Nx1 vector Step_size = something reasonable (based on size
        # of domain and transition kernel type)
        # Calculate domain size
        param_width = param_max - param_min
        # Calculate step_size
        max_ratio = t_kernel.max_ratio
        min_ratio = t_kernel.min_ratio
        step_ratio = t_kernel.init_ratio

        # Initiative first batch of N samples (maybe taken from latin
        # hypercube/space-filling curve to fully explore parameter space - not
        # necessarily random). Call these Samples_old.
        param_left = np.repeat([param_min], self.samples_per_batch,
                0).transpose()
        param_right = np.repeat([param_max], self.samples_per_batch,
                0).transpose()
        param_center = (param_right+param_left)/2.0
        samples_old = (param_right-param_left)
         
        if inital_sample_type == "lhs":
            samples_old = samples_old * lhs(param_min.shape[0],
                    self.samples_per_batch, criterion).transpose()
        elif inital_sample_type == "random" or "r":
            samples_old = samples_old * np.random.random(param_left.shape) 
        samples_old = samples_old + param_left
        samples = samples_old

        # Why don't we solve the problem at initial samples?
        data_old = self.model(samples_old)
        data = data_old
        (heur_old, proposal) = heuristic.delta_step(data_old, None)

        mdat = dict()
        self.update_mdict(mdat)
         
        for batch in xrange(1, self.num_batches):
            # For each of N samples_old, create N new parameter samples using
            # transition kernel and step_ratio. Call these samples samples_new.
            step, step_size = t_kernel.step(step_ratio, param_width,
                    self.samples_per_batch)
            # check to see if step will take you out of parameter space
            # if heading out of bounds choose step with same length with
            # direction heading towards the center of the domain
            # calulcate vector to center
            vec_to_center = samples_old - param_center
            # normalize the vec_to_center
            norm = np.linalg.norm(vec_to_center , 2, 0)
            #vec_to_center = vec_to_center/np.repeat([norm], 2, 0)
            # mutliply by step_size
            vec_to_center = samples_old - step_size*vec_to_center/4.0
            # calculate propose step
            samples_new = samples_old + step
            # Is the new sample greater than the right limit?
            far_right = samples_new >= param_right
            far_left = samples_new <= param_left
            out_of_bounds = np.logical_or(far_right, far_left)
            samples_new[out_of_bounds] = vec_to_center[out_of_bounds]

            # Solve the model for the samples_new.
            data_new = self.model(samples_new)
            
            # Make some decision about changing step_size(k).  There are
            # multiple ways to do this.
            # Determine step size
            (heur_old, proposal) = heuristic.delta_step(data_new, heur_old)
            step_ratio = proposal*step_ratio
            # Is the ratio greater than max?
            step_ratio[step_ratio > max_ratio] = max_ratio
            # Is the ratio less than min?
            step_ratio[step_ratio < min_ratio] = min_ratio

            # Save and export concatentated arrays
            if self.num_batches < 4:
                pass
            elif (batch+1)%(self.num_batches/4) == 0:
                print str(batch+1)+"th batch of "+str(self.num_batches)+" batches"
            samples = np.concatenate((samples, samples_new), axis=1)
            data = np.concatenate((data, data_new))
            mdat['samples'] = samples
            mdat['data'] = data
            self.save(mdat, savefile)

            # samples_old = samples_new
            samples_old = samples_new
        return (samples, data)

    def reseed_chains(self, inital_sample_type, param_min, param_max,
            t_kernel, heuristic, savefile, criterion='center', reseed=1):
        """
        Basic adaptive sampling algorithm.
       
        :param string inital_sample_type: type of initial sample random (or r),
            latin hypercube(lhs), or space-filling curve(TBD)
        :param param_min: minimum value for each parameter dimension
        :type param_min: np.array (ndim,)
        :param param_max: maximum value for each parameter dimension
        :type param_max: np.array (ndim,)
        :param t_kernel: method for creating new parameter steps using
            given a step size based on the paramter domain size
        :type t_kernel: :class:~`t_kernel`
        :param function heuristic: functional that acts on the data used to
            determine the proposed change to the ``step_size``
        :param string savefile: filename to save samples and data
        :param string criterion: latin hypercube criterion see 
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        :param int reseed: number of times to reseed the chains
        :rtype: tuple
        :returns: (``parameter_samples``, ``data_samples``) where
            ``parameter_samples`` is np.ndarray of shape (ndim, num_samples)
            and ``data_samples`` is np.ndarray of shape (num_samples, mdim)

        """
        # Initialize Nx1 vector Step_size = something reasonable (based on size
        # of domain and transition kernel type)
        # Calculate domain size
        param_width = param_max - param_min
        # Calculate step_size
        max_ratio = t_kernel.max_ratio
        min_ratio = t_kernel.min_ratio
        step_ratio = t_kernel.init_ratio

        # Initiative first batch of N samples (maybe taken from latin
        # hypercube/space-filling curve to fully explore parameter space - not
        # necessarily random). Call these Samples_old.
        param_left = np.repeat([param_min], self.samples_per_batch,
                0).transpose()
        param_right = np.repeat([param_max], self.samples_per_batch,
                0).transpose()
        param_center = (param_right+param_left)/2.0
        samples_old = (param_right-param_left)
         
        if inital_sample_type == "lhs":
            samples_old = samples_old * lhs(param_min.shape[0],
                    self.samples_per_batch, criterion).transpose()
        elif inital_sample_type == "random" or "r":
            samples_old = samples_old * np.random.random(param_left.shape) 
        samples_old = samples_old + param_left
        samples = samples_old

        # Why don't we solve the problem at initial samples?
        data_old = self.model(samples_old)
        data = data_old
        (heur_old, proposal) = heuristic.delta_step(data_old, None)

        mdat = dict()
        self.update_mdict(mdat)

        for batch in xrange(1, self.num_batches):
            # For each of N samples_old, create N new parameter samples using
            # transition kernel and step_ratio. Call these samples samples_new.
            step, step_size = t_kernel.step(step_ratio, param_width,
                    self.samples_per_batch)
            # check to see if step will take you out of parameter space
            # if heading out of bounds choose step with same length with
            # direction heading towards the center of the domain
            # calulcate vector to center
            vec_to_center = samples_old - param_center
            # normalize the vec_to_center
            norm = np.linalg.norm(vec_to_center , 2, 0)
            #vec_to_center = vec_to_center/np.repeat([norm], 2, 0)
            # mutliply by step_size
            vec_to_center = samples_old - step_size*vec_to_center/4.0
            # calculate propose step
            samples_new = samples_old + step
            # Is the new sample greater than the right limit?
            far_right = samples_new >= param_right
            far_left = samples_new <= param_left
            out_of_bounds = np.logical_or(far_right, far_left)
            # check to see if leaves the domain
            # if leaves the domain
            # truncate the box so that it stays in the domain
            # step = (right-left)*(1+.5*random_vec) + left
            # if to the left
            # calcuate right based on step_size
            # right = samples_old + param_width*step_size
            # step = (right-param_left)*np.random.random() + param_left
            # if to right
            # calculate left based on step_size
            # left = samples_old - param_width*step_size
            # step = (leff-param_right)*np.random.random() + left
            samples_new[out_of_bounds] = vec_to_center[out_of_bounds]

            # Solve the model for the samples_new.
            data_new = self.model(samples_new)
            
            # Make some decision about changing step_size(k).  There are
            # multiple ways to do this.
            # Determine step size
            (heur_old, proposal) = heuristic.delta_step(data_new, heur_old)
            step_ratio = proposal*step_ratio
            # Is the ratio greater than max?
            step_ratio[step_ratio > max_ratio] = max_ratio
            # Is the ratio less than min?
            step_ratio[step_ratio < min_ratio] = min_ratio

            # Save and export concatentated arrays
            if self.num_batches < 4:
                pass
            elif (batch+1)%(self.num_batches/4) == 0:
                print str(batch+1)+"th batch of "+str(self.num_batches)+" batches"
            samples = np.concatenate((samples, samples_new), axis=1)
            data = np.concatenate((data, data_new))
            mdat['samples'] = samples
            mdat['data'] = data
            self.save(mdat, savefile)

            # samples_old = samples_new
            if self.num_batches < reseed or batch+1 == self.num_batches:
                samples_old = samples_new
            elif (batch+1)%(self.num_batches/reseed) == 0:
                # reseed the chains!
                print "Reseeding at batch "+str(batch+1)+"/"+str(self.num_batches)
                # this could be made faster  by just storing the heuristic as
                # we go instead of recalculating it which is more accurate
                (heur_reseed, prop_r) = heuristic.delta_step(data, None)
                # we might want to add in something to make sure we have a
                # space filling coverage after the reseeding
                sort_ind = np.argsort(heur_reseed)
                if heuristic.sort_ascending:
                    sort_ind = sort_ind[0:self.samples_per_batch]
                else:
                    max_ind = range(len(sort_ind)-1,
                            len(sort_ind)-self.samples_per_batch-1, -1)
                    sort_ind = sort_ind[max_ind]
                samples_old = samples[:,sort_ind]
                heur_old = heur_reseed[sort_ind]
            else:
                samples_old = samples_new
        return (samples, data)


class transition_kernel(pickleable):
    """
    Basic class that is used to create a step to move from samples_old to
    samples_new based. This class generates steps for a random walk using a
    very basic algorithm. Future classes will inherit from this one with
    different implementations of the
    :meth:~`polysim.run_framework.apdative_sampling.step` method.

    This basic transition kernel is designed without a preferential direction.

    init_ratio
        Initial step size ratio compared to the parameter domain.
    min_ratio
        Minimum step size compared to the inital step size.
    max_ratio
        Maximum step size compared to the maximum step size.
    """

    def __init__(self, init_ratio, min_ratio, max_ratio):
        """
        Initialization
        """
        self.init_ratio = init_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def step(self, step_ratio, param_width, samples_per_batch):
        """
        Generate ``num_samples`` new steps using ``step_ratio`` and
        ``param_width`` to calculate the ``step size``. Each step will have a
        random direction.

        :param step_ratio: define maximum step_size = ``step_ratio*param_width``
        :type step_ratio: :class:`np.array` of shape (num_samples,)
        :param param_width: width of the parameter domain
        :type param_width: np.array (ndim,)
        :rtype: :class:`np.array` of shape (ndim, num_samples)
        :returns: step

        """
        # calculate maximum step size
        step_size = step_ratio*np.repeat([param_width], samples_per_batch,
                0).transpose()
        # randomize the direction
        random_vec = 2.0*np.random.random(step_size.shape)-1
        step = step_size*random_vec
        return (step, step_size)

class heuristic(pickleable):
    """
    Parent class for heuristics to determine change in step size. This class
    provides a method for determining the proposed change in step size. Since
    this is simply a skeleton parent class it does not change the step size at
    all.
    
    tolerance
        a tolerance used to determine if two different values are close
    increase
        the multiple to increase the step size by
    decrease
        the multiple to decrease the step size by
    """

    def __init__(self, tolerance=1E-08, increase=1.0, decrease=1.0):
        """
        Initialization
        """
        self.TOL = tolerance
        self.increase = increase
        self.decrease = decrease

    def delta_step(self, data_new, heur_old=None):
        """
        This method determines the proposed change in step size. 

        :param data_new: QoI for a given batch of samples 
        :type data_new: :class:`np.array` of shape (samples_per_batch, mdim)
        :param heur_old: heuristic evaluated at previous step
        :rtype: typle
        :returns: (heur_new, proposal)

        """
        return (None, np.ones((data_new.shape[0],)))

class rhoD_heuristic(heuristic):
    """
    We assume we know the distribution rho_D on the QoI and that the goal is to
    determine inverse regions of high probability accurately (in terms of
    getting the measure correct). This class provides a method for determining
    the proposed change in step size as follows. We check if the QoI at each of
    the samples_new(k) are closer or farther away from a region of high
    probability in D than the QoI at samples_old(k).  For example, if they are
    closer, then we can reduce the step_size(k) by 1/2.

    Note: This only works well with smooth rho_D.

    maximum
        maximum value of rho_D on D
    rho_D
        probability density on D
    tolerance 
        a tolerance used to determine if two different values are close
    increase
        the multiple to increase the step size by
    decrease
        the multiple to decrease the step size by

    """

    def __init__(self, maximum, rho_D, tolerance=1E-08, increase=2.0, 
            decrease=0.5):
        """
        Initialization
        """
        self.MAX = maximum
        self.rho_D = rho_D
        self.sort_ascending = False
        super(rhoD_heuristic, self).__init__(tolerance, increase, decrease)

    def delta_step(self, data_new, heur_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param data_new: QoI for a given batch of samples 
        :type data_new: :class:`np.array` of shape (samples_per_batch, mdim)
        :param heur_old: heuristic evaluated at previous step
        :rtype: tuple
        :returns: (heur_new, proposal)

        """
        # Evaluate heuristic for new data.
        heur_new = self.rho_D(data_new)

        if heur_old == None:
            return (heur_new, None)
        else:
            heur_diff = (heur_new-heur_old)/self.MAX
            # Compare to heuristic for old data.
            # Is the heuristic NOT close?
            heur_close = np.logical_not(np.isclose(heur_diff, 0,
                atol=self.TOL))
            heur_max = np.isclose(heur_new, self.MAX, atol=self.TOL)
            # Is the heuristic greater/lesser?
            heur_greater = np.logical_and(heur_diff > 0, heur_close)
            heur_greater = np.logical_or(heur_greater, heur_max)
            heur_lesser = np.logical_and(heur_diff < 0, heur_close)

            # Determine step size
            proposal = np.ones(heur_new.shape)
            proposal[heur_greater] = self.decrease
            proposal[heur_lesser] = self.increase
            return (heur_new, proposal.transpose())


class maxima_heuristic(pickleable):
    """
    We assume we know the maxima of the distribution rho_D on the QoI and that
    the goal is to determine inverse regions of high probability accurately (in
    terms of getting the measure correct). This class provides a method for
    determining the proposed change in step size as follows. We check if the
    QoI at each of the samples_new(k) are closer or farther away from a region
    of high probability in D than the QoI at samples_old(k). For example, if
    they are closer, then we can reduce the step_size(k) by 1/2.

    maxima
        locations of the maxima of rho_D on D
        np.array of shape (num_maxima, mdim)
    rho_max
        rho_D(maxima), list of maximum values of rho_D
    tolerance 
        a tolerance used to determine if two different values are close
    increase
        the multiple to increase the step size by
    decrease
        the multiple to decrease the step size by

    """

    def __init__(self, maxima, rho_D, tolerance=1E-08, increase=2.0, 
            decrease=0.5):
        """
        Initialization
        """
        self.MAXIMA = maxima
        self.num_maxima = maxima.shape[0]
        self.rho_max = rho_D(maxima)
        self.TOL = tolerance
        self.increase = increase
        self.decrease = decrease
        self.sort_ascending = True

    def delta_step(self, data_new, heur_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param data_new: QoI for a given batch of samples 
        :type data_new: :class:`np.array` of shape (samples_per_batch, mdim)
        :param heur_old: heuristic evaluated at previous step
        :rtype: tuple
        :returns: (heur_new, proposal)

        """
        # Evaluate heuristic for new data.
        heur_new = np.zeros((data_new.shape[0]))

        for i in xrange(data_new.shape[0]):
            # calculate distance from each of the maxima
            vec_from_maxima = np.repeat([data_new[i,:]], self.num_maxima, 0)
            vec_from_maxima = vec_from_maxima - self.MAXIMA
            # weight distances by 1/rho_D(maxima)
            dist_from_maxima = np.linalg.norm(vec_from_maxima, 2,
                1)/self.rho_max
            # set heur_new to be the minimum of weighted distances from maxima
            heur_new[i] = np.min(dist_from_maxima)

        if heur_old == None:
            return (heur_new, None)
        else:
            heur_diff = (heur_new-heur_old)
            # Compare to heuristic for old data.
            # Is the heuristic NOT close?
            heur_close = np.logical_not(np.isclose(heur_diff, 0,
                atol=self.TOL))
            # Is the heuristic greater/lesser?
            heur_greater = np.logical_and(heur_diff > 0, heur_close)
            heur_lesser = np.logical_and(heur_diff < 0, heur_close)
            # Determine step size
            proposal = np.ones(heur_new.shape)
            # if further than heur_old then increase
            proposal[heur_greater] = self.increase
            # if closer than heur_old then decrease
            proposal[heur_lesser] = self.decrease
        return (heur_new, proposal)


class maxima_mean_heuristic(pickleable):
    """
    We assume we know the maxima of the distribution rho_D on the QoI and that
    the goal is to determine inverse regions of high probability accurately (in
    terms of getting the measure correct). This class provides a method for
    determining the proposed change in step size as follows. We check if the
    QoI at each of the samples_new(k) are closer or farther away from a region
    of high probability in D than the QoI at samples_old(k). For example, if
    they are closer, then we can reduce the step_size(k) by 1/2.

    maxima
        locations of the maxima of rho_D on D
        np.array of shape (num_maxima, mdim)
    rho_max
        rho_D(maxima), list of maximum values of rho_D
    tolerance 
        a tolerance used to determine if two different values are close
    increase
        the multiple to increase the step size by
    decrease
        the multiple to decrease the step size by

    """

    def __init__(self, maxima, rho_D, tolerance=1E-08, increase=2.0, 
            decrease=0.5):
        """
        Initialization
        """
        self.MAXIMA = maxima
        self.num_maxima = maxima.shape[0]
        self.rho_max = rho_D(maxima)
        self.TOL = tolerance
        self.increase = increase
        self.decrease = decrease
        self.radius = None
        self.mean = None
        self.batch_num = 0
        self.sort_ascending = True

    def reset(self):
        """
        Resets the the batch number and the estimates of the mean and maximum
        distance from the mean.
        """
        self.radius = None
        self.mean = None
        self.batch_num = 0

    def delta_step(self, data_new, heur_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param data_new: QoI for a given batch of samples 
        :type data_new: :class:`np.array` of shape (samples_per_batch, mdim)
        :param heur_old: heuristic evaluated at previous step
        :rtype: tuple
        :returns: (heur_new, proposal)

        """
        # Evaluate heuristic for new data.
        heur_new = np.zeros((data_new.shape[0]))
        self.batch_num = self.batch_num + 1

        for i in xrange(data_new.shape[0]):
            # calculate distance from each of the maxima
            vec_from_maxima = np.repeat([data_new[i,:]], self.num_maxima, 0)
            vec_from_maxima = vec_from_maxima - self.MAXIMA
            # weight distances by 1/rho_D(maxima)
            dist_from_maxima = np.linalg.norm(vec_from_maxima, 2,
                1)/self.rho_max
            # set heur_new to be the minimum of weighted distances from maxima
            heur_new[i] = np.min(dist_from_maxima)

        if heur_old == None:
            # calculate the mean
            self.mean = np.mean(data_new, 0)
            # calculate the distance from the mean
            vec_from_mean = data_new - np.repeat([self.mean],
                    data_new.shape[0], 0)
            # estimate the radius of D
            self.radius = np.max(np.linalg.norm(vec_from_mean, 2, 1))
            return (heur_new, None)
        else:
            # update the estimate of the mean
            self.mean = (self.batch_num-1)*self.mean + np.mean(data_new, 0)
            self.mean = self.mean / self.batch_num
            # calculate the distance from the mean
            vec_from_mean = data_new - np.repeat([self.mean],
                    data_new.shape[0], 0)
            # esitmate the radius of D
            self.radius = max(np.max(np.linalg.norm(vec_from_mean, 2, 1)),
                    self.radius)
            # calculate the relative change in distance
            heur_diff = (heur_new-heur_old)
            # normalize by the radius of D (IF POSSIBLE)
            heur_diff = heur_diff #/ self.radius
            # Compare to heuristic for old data.
            # Is the heuristic NOT close?
            heur_close = np.logical_not(np.isclose(heur_diff, 0,
                atol=self.TOL))
            # Is the heuristic greater/lesser?
            heur_greater = np.logical_and(heur_diff > 0, heur_close)
            heur_lesser = np.logical_and(heur_diff < 0, heur_close)
            # Determine step size
            proposal = np.ones(heur_new.shape)
            # if further than heur_old then increase
            proposal[heur_greater] = self.increase
            # if closer than heur_old then decrease
            proposal[heur_lesser] = self.decrease
        return (heur_new, proposal)


class multi_dist_heuristic(pickleable):
    """
    The goal is to make a sampling that is robust to different types of
    distributions on QoI, i.e., we do not know a priori where the regions of
    high probability are in D. This class provides a method for determining the
    proposed step size as follows. We keep track of the change of the QoI
    values from one sample to the next compared to the total range of QoI
    values explored so far. If a big relative change is detected, then you know
    that you have come across a region with larger derivatives and you should
    place more samples around there to resolve the induced regions of
    generalized contours, i.e., reduce the step size. If the change in QoI
    values is relatively small, you are in a region where there is little
    sensitivity, so take larger step sizes.

    radius
        current estimate of the radius of D (1/2 the diameter of D)
    mean
        current estimate of the mean QoI
    batch_num
        current batch number
    TOL 
        a tolerance used to determine if two different values are close
    increase
        the multiple to increase the step size by
    decrease
        the multiple to decrease the step size by

    """

    def __init__(self, tolerance=1E-08, increase=2.0, 
            decrease=0.5):
        """
        Initialization
        """
        self.radius = None
        self.mean = None
        self.batch_num = 0
        self.TOL = tolerance
        self.increase = increase
        self.decrease = decrease

    def reset(self):
        """
        Resets the the batch number and the estimates of the mean and maximum
        distance from the mean.
        """
        self.radius = None
        self.mean = None
        self.batch_num = 0

    def delta_step(self, data_new, heur_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param data_new: QoI for a given batch of samples 
        :type data_new: :class:`np.array` of shape (samples_per_batch, mdim)
        :param heur_old: QoI evaluated at previous step
        :rtype: tuple
        :returns: (heur_new, proposal)

        """
        # Evaluate heuristic for new data.
        heur_new = data_new
        self.batch_num = self.batch_num + 1

        if heur_old == None:
            proposal = None
            # calculate the mean
            self.mean = np.mean(data_new, 0)
            # calculate the distance from the mean
            vec_from_mean = heur_new - np.repeat([self.mean],
                    heur_new.shape[0], 0)
            # estimate the radius of D
            self.radius = np.max(np.linalg.norm(vec_from_mean, 2, 1)) 
        else:
            # update the estimate of the mean
            self.mean = (self.batch_num-1)*self.mean + np.mean(data_new, 0)
            self.mean = self.mean / self.batch_num
            # calculate the distance from the mean
            vec_from_mean = heur_new - np.repeat([self.mean],
                    heur_new.shape[0], 0)
            # esitmate the radius of D
            self.radius = max(np.max(np.linalg.norm(vec_from_mean, 2, 1)),
                    self.radius)
            # calculate the relative change in QoI
            heur_diff = (heur_new-heur_old)
            # normalize by the radius of D
            heur_diff = np.linalg.norm(vec_from_mean, 2, 1)#/self.radius
            # Compare to heuristic for old data.
            # Is the heuristic NOT close?
            heur_close = np.logical_not(np.isclose(heur_diff, 0,
                atol=self.TOL))
            # Is the heuristic greater/lesser?
            heur_greater = np.logical_and(heur_diff > 0, heur_close)
            heur_lesser = np.logical_and(heur_diff < 0, heur_close)
            # Determine step size
            proposal = np.ones(heur_diff.shape)
            proposal[heur_greater] = self.decrease
            proposal[heur_lesser] = self.increase
        return (heur_new, proposal)




