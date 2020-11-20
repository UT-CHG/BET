# Copyright (C) 2014-2020 The BET Development Team

"""
This module contains functions for sampling. We assume we are given access to a
model, a parameter space, and a data space. The model is a map from the
parameter space to the data space. We desire to build up a set of samples to
solve an inverse problem thus giving information about the inverse mapping.
Each sample consists for a parameter coordinate, data coordinate pairing. We
assume the measure on both spaces is Lebesgue.
"""

import collections.abc
import os
import warnings
import logging
import glob
import numpy as np
import scipy.io as sio
import scipy.stats as stats
from pyDOE import lhs
from bet.Comm import comm
import bet.sample as sample
import bet.sample
import bet.util as util


class bad_object(Exception):
    """
    Exception for when the wrong type of object is used.
    """


def sample_from_updated(input_set, num_samples, globalize=True):
    """
    Create a new sample set from resampling from the updated probability measure of another sample set.

    :param input_set: Sample set or discretization containing updated probability measure from which to sample.
    :type input_set: :class:`~bet.sample.sample_set` or :class:`~bet.sample.discretization`
    :param num_samples: Number of new samples to create.
    :type num_samples: int
    :param globalize: Whether or not to globalize objects.
    :type bool
    :return: Sample set containing new samples
    :rtype: :class:`~bet.sample.sample_set`

    """
    if isinstance(input_set, bet.sample.discretization):
        input_set = input_set.get_input_sample_set()
    elif not isinstance(input_set, bet.sample.sample_set):
        raise bad_object("input_set is of the wrong type.")

    new_set = sample.sample_set(dim=input_set.get_dim())
    if input_set.get_prob_type() == 'rv':
        return random_sample_set(input_set.get_prob_parameters(), new_set, num_samples, globalize)
    elif input_set.get_prob_type() == 'kde':
        param_marginals, cluster_weights = input_set.get_prob_parameters()
        v_outer = []
        for i, w in enumerate(cluster_weights):
            v_inner = []
            num_samples_clust = round(w*num_samples)
            num_samples_local = int((num_samples_clust / comm.size) +
                                    (comm.rank < num_samples_clust % comm.size))
            for j in range(input_set.get_dim()):
                v_inner.append(param_marginals[j][i].resample(num_samples_local))
            v_outer.append(np.vstack(v_inner))
        vals_local = np.hstack(v_outer)
        new_set.set_values_local(vals_local)
        new_set.set_prob_type_init('kde')
        new_set.set_prob_parameters_init((param_marginals, cluster_weights))
        if globalize:
            new_set.local_to_global()
        return new_set
    elif input_set.get_prob_type() == 'gmm':
        means, covariances, cluster_weights = input_set.get_prob_parameters()
        v_outer = []
        for i, w in enumerate(cluster_weights):
            num_samples_clust = round(w * num_samples)
            num_samples_local = int((num_samples_clust / comm.size) +
                                    (comm.rank < num_samples_clust % comm.size))
            v_outer.append(stats.multivariate_normal.rvs(mean=means[i], cov=covariances[i], size=num_samples_local))
        vals_local = np.vstack(v_outer)
        new_set.set_values_local(vals_local)
        new_set.set_prob_type_init('gmm')
        new_set.set_prob_parameters_init((means, covariances, cluster_weights))
        if globalize:
            new_set.local_to_global()
        return new_set
    else:
        raise bad_object("The updated probability measure is undefined or not allowed for this method.")


def random_sample_set(rv, input_obj, num_samples, globalize=True):
    """
    Create a sample set by sampling random variates from continuous distributions
    from :class:`scipy.stats.rv_continuous`. See https://docs.scipy.org/doc/scipy/reference/stats.html.

    `rv` can take multiple types of formats depending on type of distribution.

    A string is used for the same distribution with default parameters in each dimension.
    ex. rv = 'uniform' or rv = 'beta'

    A list or tuple of length 2 is used for the same distribution with user-defined parameters in each dimension as a
    dictionary.
    ex. rv = ['uniform', {'loc':-2, 'scale':5}] or rv = ['beta', {'a': 2, 'b':5, 'loc':-2, 'scale':5}]

    A list of length dim which entries of lists or tuples of length 2 is used for different distributions with
    user-defined parameters in each dimension as a
    dictionary.
    ex. rv = [['uniform', {'loc':-2, 'scale':5}],
              ['beta', {'a': 2, 'b':5, 'loc':-2, 'scale':5}]]

    :param rv: Type and parameters for continuous random variables.
    :type rv: str, list, or tuple
    :param input_obj: :class:`~bet.sample.sample_set` object containing the dimension to sample from, or the dimension.
    :type input_obj: :class:`~bet.sample.sample_set` or int or :class:`numpy.ndarray`
    :param num_samples: Number of samples
    :type num_samples: int
    :param globalize: Whether or not to globalize vectors.
    :type globalize: bool

    """
    # for backward compatibility
    if rv == "r" or rv == "random":
        rv = "uniform"
    elif rv == 'lhs':
        return lhs_sample_set(input_obj, num_samples, criterion='center', globalize=globalize)
    # check to see what the input object is
    if isinstance(input_obj, sample.sample_set):
        input_sample_set = input_obj
    elif isinstance(input_obj, int):
        input_sample_set = sample.sample_set(input_obj)
    elif isinstance(input_obj, np.ndarray):
        input_sample_set = sample.sample_set(input_obj.shape[0])
        input_sample_set.set_domain(input_obj)
    else:
        raise sample.wrong_input("input_obj is of wrong type.")

    dim = input_sample_set.get_dim()

    if type(rv) is str:
        if input_sample_set.get_domain() is None:
            rv = [[rv, {}]] * dim
        else:
            domain = input_sample_set.get_domain()
            rv_type = rv
            rv = []
            for i in range(dim):
                rv.append([rv_type, {'loc': domain[i, 0], 'scale': domain[i, 1]-domain[i, 0]}])
    elif type(rv) in (list, tuple):
        if len(rv) == 2 and type(rv[0]) is str and type(rv[1]) is dict:
            rv = [rv] * dim
        elif len(rv) != dim:
            raise sample.dim_not_matching("rv has fewer entries than the dimension.")
    else:
        raise sample.wrong_input("rv must be a string, list, or tuple.")

    # define local number of samples
    num_samples_local = int((num_samples / comm.size) +
                            (comm.rank < num_samples % comm.size))

    input_values_local = np.empty((num_samples_local, dim))
    domain = np.empty((dim, 2))

    for i in range(dim):
        rv_continuous = getattr(stats, rv[i][0])
        args = rv[i][1]
        input_values_local[:, i] = rv_continuous.rvs(size=num_samples_local, **args)
        domain[i, :] = rv_continuous.interval(1, **args)
    input_sample_set.set_values_local(input_values_local)
    input_sample_set.set_domain(domain)
    input_sample_set.set_prob_type_init("rv")
    input_sample_set.set_prob_parameters_init(rv)
    input_sample_set.check_num_local()
    input_sample_set.check_num()

    comm.barrier()

    if globalize:
        input_sample_set.local_to_global()
    else:
        input_sample_set._values = None
    return input_sample_set


def lhs_sample_set(input_obj, num_samples, criterion, globalize=True):
    """
    Sampling algorithm for generating samples from a Latin hypercube
    in the domain present with ``input_obj`` (a default unit hypercube
    is used if no domain has been specified)

    :param input_obj: :class:`~bet.sample.sample_set` object containing
        the dimension or domain to sample from, the domain to sample from, or
        the dimension
    :type input_obj: :class:`~bet.sample.sample_set` or :class:`numpy.ndarray`
        of shape (dim, 2) or ``int``
    :param num_samples: number of samples
    :type num_samples: int
    :param criterion: latin hypercube criterion see
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`
    :type criterion: str
    :param globalize: Whether or not to globalize local variables.
    :type globalize: bool
    :rtype: :class:`~bet.sample.sample_set`
    :returns: :class:`~bet.sample.sample_set`

    """
    # check to see what the input object is
    if isinstance(input_obj, sample.sample_set):
        input_sample_set = input_obj
    elif isinstance(input_obj, int):
        input_sample_set = sample.sample_set(input_obj)
    elif isinstance(input_obj, np.ndarray):
        input_sample_set = sample.sample_set(input_obj.shape[0])
        input_sample_set.set_domain(input_obj)

    dim = input_sample_set.get_dim()
    if input_sample_set.get_domain() is None:
        # create the domain
        input_domain = np.array([[0., 1.]] * dim)
        input_sample_set.set_domain(input_domain)
        logging.warning("Setting domain to hypercube.")

    # update the bounds based on the number of samples
    input_sample_set.update_bounds(num_samples)
    input_values = np.copy(input_sample_set._width)
    input_values = input_values * lhs(dim, num_samples, criterion)
    input_values = input_values + input_sample_set._left
    input_sample_set.set_values_local(np.array_split(input_values, comm.size)[comm.rank])

    comm.barrier()
    if globalize:
        input_sample_set.local_to_global()
    else:
        input_sample_set._values = None
    input_sample_set.set_prob_type_init("lhs")
    input_sample_set.set_prob_parameters_init(criterion)

    return input_sample_set


def regular_sample_set(input_obj, num_samples_per_dim=1):
    """
    Sampling algorithm for generating a regular grid of samples taken
    on the domain present with ``input_obj`` (a default unit hypercube
    is used if no domain has been specified)

    :param input_obj: :class:`~bet.sample.sample_set` object containing
        the dimension or domain to sample from, the domain to sample from, or
        the dimension
    :type input_obj: :class:`~bet.sample.sample_set` or :class:`numpy.ndarray`
        of shape (dim, 2) or ``int``
    :param num_samples_per_dim: number of samples per dimension
    :type num_samples_per_dim: :class:`~numpy.ndarray` of dimension
        ``(input_sample_set._dim,)``

    :rtype: :class:`~bet.sample.sample_set`
    :returns: :class:`~bet.sample.sample_set` object which contains
        input ``num_samples``

    """
    # check to see what the input object is
    if isinstance(input_obj, sample.sample_set):
        input_sample_set = input_obj.copy()
    elif isinstance(input_obj, int):
        input_sample_set = sample.sample_set(input_obj)
    elif isinstance(input_obj, np.ndarray):
        input_sample_set = sample.sample_set(input_obj.shape[0])
        input_sample_set.set_domain(input_obj)
    else:
        raise bad_object("Improper sample object")

    # Create N samples
    dim = input_sample_set.get_dim()

    if not isinstance(num_samples_per_dim, collections.abc.Iterable):
        num_samples_per_dim = num_samples_per_dim * np.ones((dim,))
    if np.any(np.less_equal(num_samples_per_dim, 0)):
        warnings.warn('Warning: num_samples_per_dim must be greater than 0')

    num_samples = int(np.product(num_samples_per_dim))

    if input_sample_set.get_domain() is None:
        # create the domain
        input_domain = np.array([[0., 1.]] * dim)
        input_sample_set.set_domain(input_domain)
    else:
        input_domain = input_sample_set.get_domain()
    # update the bounds based on the number of samples
    input_values = np.zeros((num_samples, dim))

    vec_samples_dimension = np.empty((dim), dtype=object)
    for i in range(dim):
        bin_width = (input_domain[i, 1] - input_domain[i, 0]) / \
            np.float(num_samples_per_dim[i])
        vec_samples_dimension[i] = list(np.linspace(
            input_domain[i, 0] - 0.5 * bin_width,
            input_domain[i, 1] + 0.5 * bin_width,
            int(num_samples_per_dim[i]) + 2))[1:int(num_samples_per_dim[i] + 1)]

    arrays_samples_dimension = np.meshgrid(
        *[vec_samples_dimension[i] for i in np.arange(0, dim)],
        indexing='ij')

    for i in range(dim):
        input_values[:, i:i + 1] = np.vstack(arrays_samples_dimension[i]
                                             .flat[:])

    input_sample_set.set_values(input_values)
    input_sample_set.global_to_local()
    input_sample_set.set_prob_type_init("grid")
    input_sample_set.set_prob_parameters_init(num_samples_per_dim)

    return input_sample_set


class sampler(object):
    """
    This class provides methods for sampling of parameter space to
    provide samples to be used by algorithms to solve inverse problems.

    """
    def __init__(self, lb_model,
                 error_estimates=False, jacobians=False):
        """
        Initialization

        :param lb_model: Interface to physics-based model takes an input of
            shape (N, ndim) and returns an output of shape (N, mdim)
        :type lb_model: callable function
        :param bool error_estimates: Whether or not the model returns error estimates
        :param bool jacobians: Whether or not the model returns Jacobians
        """
        self.lb_model = lb_model
        self.error_estimates = error_estimates
        self.jacobians = jacobians
        self.input_sample_set = None
        self.discretization = None

    def local_to_global(self):
        """
        Globalize local variables.
        """
        if self.input_sample_set is not None:
            self.input_sample_set.local_to_global()
        if self.discretization is not None:
            self.discretization.local_to_global()

    def random_sample_set(self, rv, input_obj, num_samples, globalize=True):
        """
        Create a sample set by sampling random variates from continuous distributions
        from :class:`scipy.stats.rv_continuous`. See https://docs.scipy.org/doc/scipy/reference/stats.html.

        `rv` can take multiple types of formats depending on type of distribution.

        A string is used for the same distribution with default parameters in each dimension.
        ex. rv = 'uniform' or rv = 'beta'

        A list or tuple of length 2 is used for the same distribution with user-defined parameters in each dimension as a
        dictionary.
        ex. rv = ['uniform', {'loc':-2, 'scale':5}] or rv = ['beta', {'a': 2, 'b':5, 'loc':-2, 'scale':5}]

        A list of length dim which entries of lists or tuples of length 2 is used for different distributions with
        user-defined parameters in each dimension as a
        dictionary.
        ex. rv = [['uniform', {'loc':-2, 'scale':5}],
                  ['beta', {'a': 2, 'b':5, 'loc':-2, 'scale':5}]]

        :param rv: Type and parameters for continuous random variables.
        :type rv: str, list, or tuple
        :param input_obj: :class:`~bet.sample.sample_set` object containing the dimension to sample from, or the dimension.
        :type input_obj: :class:`~bet.sample.sample_set` or int
        :param num_samples: Number of samples
        :type num_samples: int
        :param globalize: Whether or not to globalize vectors.
        :type globalize: bool
        :return:
        """
        self.input_sample_set = random_sample_set(rv, input_obj, num_samples, globalize=globalize)
        return self.input_sample_set

    def regular_sample_set(self, input_obj, num_samples_per_dim=1):
        """
        Sampling algorithm for generating a regular grid of samples taken
        on the domain present with ``input_obj`` (a default unit hypercube
        is used if no domain has been specified)

        :param input_obj: :class:`~bet.sample.sample_set` object containing
            the dimension or domain to sample from, the domain to sample from, or
            the dimension
        :type input_obj: :class:`~bet.sample.sample_set` or :class:`numpy.ndarray`
            of shape (dim, 2) or ``int``
        :param num_samples_per_dim: number of samples per dimension
        :type num_samples_per_dim: :class:`~numpy.ndarray` of dimension
            ``(input_sample_set._dim,)``

        :rtype: :class:`~bet.sample.sample_set`
        :returns: :class:`~bet.sample.sample_set` object which contains
            input ``num_samples``

        """
        self.input_sample_set = regular_sample_set(input_obj, num_samples_per_dim)
        return self.input_sample_set

    def lhs_sample_set(self, input_obj, num_samples, criterion, globalize=True):
        """
        Sampling algorithm for generating samples from a Latin hypercube
        in the domain present with ``input_obj`` (a default unit hypercube
        is used if no domain has been specified)

        :param input_obj: :class:`~bet.sample.sample_set` object containing
            the dimension or domain to sample from, the domain to sample from, or
            the dimension
        :type input_obj: :class:`~bet.sample.sample_set` or :class:`numpy.ndarray`
            of shape (dim, 2) or ``int``
        :param num_samples: number of samples
        :type num_samples: int
        :param criterion: latin hypercube criterion see
                `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`
        :type criterion: str
        :param globalize: Whether or not to globalize local variables.
        :type globalize: bool
        :rtype: :class:`~bet.sample.sample_set`
        :returns: :class:`~bet.sample.sample_set`

        """
        self.input_sample_set = lhs_sample_set(input_obj, num_samples, criterion, globalize)
        return self.input_sample_set

    def compute_QoI_and_create_discretization(self, input_sample_set=None,
                                              savefile=None, globalize=True):
        """
        Dummy function for `compute_qoi_and_create_discretization`.
        """
        logging.warning("This will be removed in a later version. Use compute_qoi_and_create_discretization instead.")
        return self.compute_qoi_and_create_discretization(input_sample_set, savefile, globalize)

    def compute_qoi_and_create_discretization(self, input_sample_set=None,
                                              savefile=None, globalize=True):
        """
        Samples the model at ``input_sample_set`` and saves the results.

        Note: There are many ways to generate samples on a regular grid in
        Numpy and other Python packages. Instead of reimplementing them here we
        provide sampler that utilizes user specified samples.

        :param input_sample_set: samples to evaluate the model at
        :type input_sample_set: :class:`~bet.sample.sample_set` with
            num_samples
        :param string savefile: filename to save samples and data
        :param bool globalize: Makes local variables global.

        :rtype: :class:`~bet.sample.discretization`
        :returns: :class:`~bet.sample.discretization` object which contains
            input and output of length ``num_samples``

        """

        if input_sample_set is not None:
            self.input_sample_set = input_sample_set

        # Solve the model at the samples
        if self.input_sample_set._values_local is None:
            self.input_sample_set.global_to_local()

        local_output = self.lb_model(
            self.input_sample_set.get_values_local())

        if isinstance(local_output, np.ndarray):
            local_output_values = local_output
        elif isinstance(local_output, tuple):
            if len(local_output) == 1:
                local_output_values = local_output[0]
            elif len(local_output) == 2 and self.error_estimates:
                (local_output_values, local_output_ee) = local_output
            elif len(local_output) == 2 and self.jacobians:
                (local_output_values, local_output_jac) = local_output
            elif len(local_output) == 3:
                (local_output_values, local_output_ee, local_output_jac) = \
                    local_output
        else:
            raise bad_object("lb_model is not returning the proper type")

        # figure out the dimension of the output
        if len(local_output_values.shape) <= 1:
            output_dim = 1
        else:
            output_dim = local_output_values.shape[1]

        output_sample_set = sample.sample_set(output_dim)
        output_sample_set.set_values_local(local_output_values)
        lam_ref = self.input_sample_set.get_reference_value()

        if lam_ref is not None:
            try:
                if not isinstance(lam_ref, collections.abc.Iterable):
                    lam_ref = np.array([lam_ref])
                Q_ref = self.lb_model(lam_ref)
                output_sample_set.set_reference_value(Q_ref)
            except ValueError:
                try:
                    msg = "Model not mapping reference value as expected."
                    msg += "Attempting reshape..."
                    logging.log(20, msg)
                    q_ref = self.lb_model(lam_ref.reshape(1, -1))
                    output_sample_set.set_reference_value(q_ref)
                except ValueError:
                    logging.log(20, 'Unable to map reference value.')

        if self.error_estimates:
            output_sample_set.set_error_estimates_local(local_output_ee)

        if self.jacobians:
            self.input_sample_set.set_jacobians_local(local_output_jac)

        if globalize:
            self.input_sample_set.local_to_global()
            output_sample_set.local_to_global()
        else:
            self.input_sample_set._values = None

        comm.barrier()

        self.discretization = sample.discretization(self.input_sample_set,
                                                    output_sample_set)
        comm.barrier()

        if savefile is not None:
            self.discretization.save(filename=savefile, globalize=globalize)

        comm.barrier()

        return self.discretization

    def copy(self):
        """
        Returns a copy of the sampler object.
        """
        import copy
        return copy.deepcopy(self)

    def create_random_discretization(self, rv, input_obj,
                                     savefile=None, num_samples=None,
                                     globalize=True):
        """
        Create a sample set by sampling random variates from continuous distributions
        from :class:`scipy.stats.rv_continuous`. See https://docs.scipy.org/doc/scipy/reference/stats.html,
        and evaluate the model to calculate quantities of interest and make a discretization.

        `rv` can take multiple types of formats depending on type of distribution.

        A string is used for the same distribution with default parameters in each dimension.
        ex. rv = 'uniform' or rv = 'beta'

        A list or tuple of length 2 is used for the same distribution with user-defined parameters in each dimension as a
        dictionary.
        ex. rv = ['uniform', {'loc':-2, 'scale':5}] or rv = ['beta', {'a': 2, 'b':5, 'loc':-2, 'scale':5}]

        A list of length dim which entries of lists or tuples of length 2 is used for different distributions with
        user-defined parameters in each dimension as a
        dictionary.
        ex. rv = [['uniform', {'loc':-2, 'scale':5}],
                  ['beta', {'a': 2, 'b':5, 'loc':-2, 'scale':5}]]

        :param rv: Type and parameters for continuous random variables.
        :type rv: str, list, or tuple
        :param input_obj: :class:`~bet.sample.sample_set` object containing the dimension to sample from, or the dimension.
        :type input_obj: :class:`~bet.sample.sample_set` or int
        :param string savefile: filename to save discretization
        :param num_samples: Number of samples
        :type num_samples: int
        :param globalize: Whether or not to globalize vectors.
        :type globalize: bool

        :rtype: :class:`~bet.sample.discretization`
        :returns: :class:`~bet.sample.discretization` object which contains
            input and output sample sets with ``num_samples`` total samples

        """
        # Create N samples
        if num_samples is None:
            num_samples = self.num_samples

        input_sample_set = self.random_sample_set(rv, input_obj,
                                                  num_samples, globalize)

        return self.compute_qoi_and_create_discretization(input_sample_set,
                                                          savefile, globalize)
