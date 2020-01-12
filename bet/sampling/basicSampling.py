# Copyright (C) 2014-2019 The BET Development Team

"""
This module contains functions for sampling. We assume we are given access to a
model, a parameter space, and a data space. The model is a map from the
paramter space to the data space. We desire to build up a set of samples to
sovle an inverse problem this guving use information about the inverse mapping.
Each sample consists for a paramter coordinate, data coordinate pairing. We
assume the measure on both spaces in Lebesgue.
"""

import collections
import os
import logging
import warnings
import glob
import numpy as np
from pyDOE import lhs
from bet.Comm import comm
import bet.sample as sample


class bad_object(Exception):
    """
    Exception for when the wrong type of object is used.
    """


def loadmat(save_file, disc_name=None, model=None):
    """
    Loads data from ``save_file`` into a
    :class:`~bet.basicSampling.sampler` object.

    :param string save_file: file name
    :param string disc_name: name of :class:`~bet.sample.discretization` in
        file
    :param model: runs the model at a given set of parameter samples and
        returns data
    :type model: callable

    :rtype: tuple
    :returns: (sampler, discretization)

    """
    # check to see if parallel save
    if not (os.path.exists(save_file) or os.path.exists(save_file + '.mat')):
        save_dir = os.path.dirname(save_file)
        base_name = os.path.basename(save_file)
        mdat_files = glob.glob(os.path.join(save_dir,
                                            "proc*_{}".format(base_name)))
        # load the data from a *.mat file
        mdat = sample.loadmat(mdat_files[0])
    else:
        # load the data from a *.mat file
        mdat = sample.loadmat(save_file)
    num_samples = mdat['num_samples']
    # load the discretization
    discretization = sample.load_discretization(save_file, disc_name)
    loaded_sampler = sampler(model, num_samples)
    return (loaded_sampler, discretization)


def random_sample_set(sample_type, input_obj, num_samples,
                      criterion='center', globalize=True):
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
    :param input_obj: :class:`~bet.sample.sample_set` object containing
        the dimension/domain to sample from, domain to sample from, or the
        dimension
    :type input_obj: :class:`~bet.sample.sample_set` or
        :class:`numpy.ndarray` of shape (dim, 2) or ``int``
    :param string savefile: filename to save discretization
    :param int num_samples: N, number of samples
    :param string criterion: latin hypercube criterion see
        `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
    :param bool globalize: Makes local variables global. Only applies if
        ``parallel==True``.

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

    if input_sample_set.get_domain() is None:
        # create the domain
        input_domain = np.array([[0., 1.]] * dim)
        input_sample_set.set_domain(input_domain)
    else:
        input_domain = input_sample_set.get_domain()

    if sample_type == "lhs":
        # update the bounds based on the number of samples
        input_sample_set.update_bounds(num_samples)
        input_values = np.copy(input_sample_set._width)
        input_values = input_values * lhs(dim,
                                          num_samples, criterion)
        input_values = input_values + input_sample_set._left
        input_sample_set.set_values_local(np.array_split(input_values,
                                                         comm.size)[comm.rank])
    elif sample_type == "random" or "r":
        # define local number of samples
        num_samples_local = (num_samples // comm.size) +\
            int(comm.rank < num_samples % comm.size)
        # update the bounds based on the number of samples
        input_sample_set.update_bounds_local(num_samples_local)
        input_values_local = np.copy(input_sample_set._width_local)
        input_values_local = input_values_local * \
            np.random.random(input_values_local.shape)
        input_values_local = input_values_local + input_sample_set._left_local

        input_sample_set.set_values_local(input_values_local)
        from scipy.stats import uniform
        input_sample_set.set_distribution(uniform, loc=input_domain[:,0], scale=input_domain[:,1]-input_domain[:,0])
    comm.barrier()

    if globalize:
        input_sample_set.local_to_global()
    else:
        input_sample_set._values = None
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

    if not isinstance(num_samples_per_dim, collections.Iterable):
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
            num_samples_per_dim[i] + 2))[1:int(num_samples_per_dim[i] + 1)]

    arrays_samples_dimension = np.meshgrid(
        *[vec_samples_dimension[i] for i in np.arange(0, dim)],
        indexing='ij')

    for i in range(dim):
        input_values[:, i:i + 1] = np.vstack(arrays_samples_dimension[i]
                                             .flat[:])

    input_sample_set.set_values(input_values)
    input_sample_set.global_to_local()

    return input_sample_set


class sampler(object):
    """
    This class provides methods for adaptive sampling of parameter space to
    provide samples to be used by algorithms to solve inverse problems.

    num_samples
        total number of samples OR list of number of samples per dimension such
        that total number of samples is prob(num_samples)
    lb_model
        callable function that runs the model at a given set of input and
        returns output
    """

    def __init__(self, lb_model, num_samples=None,
                 error_estimates=False, jacobians=False):
        """
        Initialization

        :param lb_model: Interface to physics-based model takes an input of
            shape (N, ndim) and returns an output of shape (N, mdim)
        :type lb_model: callable function
        :param int num_samples: N, number of samples
        :param bool error_estimates: Whether or not the model returns error
            estimates
        :param bool jacobians: Whether or not the model returns Jacobians

        """
        #: int, total number of samples OR list of number of samples per
        #: dimension such that total number of samples is prob(num_samples)
        self.num_samples = num_samples
        #: callable function that runs the model at a given set of input and
        #: returns output
        #: parameter samples and returns data

        self.lb_model = lb_model
        self.error_estimates = error_estimates
        self.jacobians = jacobians

    def set_num_samples(self, num_samples):
        """
        Overwrite `num_samples`, the default value for sample size.

        :param int num_samples: Number of samples for sample-set objects.
        """
        self.num_samples = int(num_samples)

    def save(self, mdict, save_file, discretization=None, globalize=False):
        """
        Save matrices to a ``*.mat`` file for use by ``MATLAB BET`` code and
        :meth:`~bet.basicSampling.loadmat`

        :param dict mdict: dictonary of sampler parameters
        :param string save_file: file name
        :param discretization: input and output from sampling
        :type discretization: :class:`bet.sample.discretization`
        :param bool globalize: Makes local variables global.

        """

        if comm.size > 1 and not globalize:
            local_save_file = os.path.join(os.path.dirname(save_file),
                                           "proc{}_{}".
                                           format(comm.rank, os.path.
                                                  basename(save_file)))
        else:
            local_save_file = save_file

        if (globalize and comm.rank == 0) or not globalize:
            sample.savemat(local_save_file, mdict)
        comm.barrier()

        if discretization is not None:
            sample.save_discretization(discretization, save_file,
                                       globalize=globalize)

    def update_mdict(self, mdict):
        """
        Set up references for ``mdict``

        :param dict mdict: dictonary of sampler parameters

        """
        mdict['num_samples'] = self.num_samples

    def random_sample_set(self, sample_type, input_obj,
                          num_samples=None, criterion='center', globalize=True):
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
        :param input_obj: :class:`~bet.sample.sample_set` object containing
            the dimension/domain to sample from, domain to sample from, or the
            dimension
        :type input_obj: :class:`~bet.sample.sample_set` or
            :class:`numpy.ndarray` of shape (dim, 2) or ``int``
        :param string savefile: filename to save discretization
        :param int num_samples: N, number of samples (optional)
        :param string criterion: latin hypercube criterion see
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        :param bool globalize: Makes local variables global.

        :rtype: :class:`~bet.sample.sample_set`
        :returns: :class:`~bet.sample.sample_set` object which contains
            input ``num_samples``

        """
        if num_samples is None:
            num_samples = self.num_samples

        return random_sample_set(sample_type, input_obj, num_samples,
                                 criterion, globalize)

    def regular_sample_set(self, input_obj, num_samples_per_dim=1):
        """
        Sampling algorithm for generating a regular grid of samples taken
        on the domain present with ``input_obj`` (a default unit hypercube
        is used if no domain has been specified)

        :param input_obj: :class:`~bet.sample.sample_set` object containing
            the dimension or domain to sample from, the domain to sample from,
            or the dimension
        :type input_obj: :class:`~bet.sample.sample_set` or
            :class:`numpy.ndarray` of shape (dim, 2) or ``int``
        :param num_samples_per_dim: number of samples per dimension
        :type num_samples_per_dim: :class:`~numpy.ndarray` of dimension
            (dim,)

        :rtype: :class:`~bet.sample.sample_set`
        :returns: :class:`~bet.sample.sample_set` object which contains
            input ``num_samples``

        """
        self.num_samples = np.product(num_samples_per_dim)
        return regular_sample_set(input_obj, num_samples_per_dim)

    def compute_QoI_and_create_discretization(self, input_sample_set,
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
            input and output of ``num_samples``

        """
        if isinstance(input_sample_set, int):
            msg = "No sample set. Using uniform random sampling in [0,1]."
            logging.warning(msg)
            if self.num_samples is None:
                logging.warning("Missing num_samples. Using 100 as default.")
                self.num_samples = 100
            input_sample_set = self.random_sample_set('r', input_sample_set)

        # Update the number of samples
        self.num_samples = input_sample_set.check_num()

        # Solve the model at the samples
        if input_sample_set._values_local is None:
            input_sample_set.global_to_local()

        local_output = self.lb_model(
            input_sample_set.get_values_local())

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
        lam_ref = input_sample_set._reference_value

        if lam_ref is not None:
            try:
                if not isinstance(lam_ref, collections.Iterable):
                    lam_ref = np.array([lam_ref])
                Q_ref = self.lb_model(lam_ref)
                if not isinstance(Q_ref, collections.Iterable):
                    Q_ref = np.array([Q_ref])
                output_sample_set.set_reference_value(Q_ref.ravel())
            except ValueError:
                try:
                    msg = "Model not mapping reference value as expected."
                    msg += "Attempting reshape..."
                    logging.warning(msg)
                    Q_ref = self.lb_model(lam_ref.reshape(1, -1))
                    output_sample_set.set_reference_value(Q_ref)
                except ValueError:
                    sample.dim_not_matching('Unable to map reference value.')

        if self.error_estimates:
            output_sample_set.set_error_estimates_local(local_output_ee)

        if self.jacobians:
            input_sample_set.set_jacobians_local(local_output_jac)

        if globalize:
            input_sample_set.local_to_global()
            output_sample_set.local_to_global()
        else:
            input_sample_set._values = None

        comm.barrier()

        discretization = sample.discretization(input_sample_set,
                                               output_sample_set)
        comm.barrier()

        mdat = dict()
        self.update_mdict(mdat)

        if savefile is not None:
            self.save(mdat, savefile, discretization, globalize=globalize)

        comm.barrier()

        # Attach necessary attributes to discretization
        discretization._setup[0]['model'] = self.lb_model
        discretization._setup[0]['ind'] = list(np.arange(output_dim))

        return discretization

    def create_random_discretization(self, sample_type, input_obj,
                                     savefile=None, num_samples=None,
                                     criterion='center',
                                     globalize=True, param_ref=None):
        """
        Sampling algorithm with three basic options

            * ``random`` (or ``r``) generates ``num_samples`` samples in
                ``lam_domain`` assuming a Lebesgue measure.
            * ``lhs`` generates a latin hyper cube of samples.

        .. note::

            This function is designed only for generalized rectangles and
            assumes a Lebesgue measure on the parameter space.


        :param string sample_type: type sampling random (or r),
            latin hypercube(lhs), regular grid (rg), or space-filling
            curve(TBD)
        :param input_obj: Either a :class:`bet.sample.sample_set` object for an
            input space, an array of min and max bounds for the input values
            with ``min = input_domain[:, 0]`` and ``max = input_domain[:, 1]``,
            or the dimension of an input space
        :type input_obj: :class:`~bet.sample.sample_set`,
            :class:`numpy.ndarray` of shape (ndim, 2), or :class: `int`
        :param string savefile: filename to save discretization
        :param int num_samples: N, number of samples (optional)
        :param string criterion: latin hypercube criterion see
            `PyDOE <http://pythonhosted.org/pyDOE/randomized.html>`_
        :param bool globalize: Makes local variables global.

        :rtype: :class:`~bet.sample.discretization`
        :returns: :class:`~bet.sample.discretization` object which contains
            input and output sample sets with ``num_samples`` total samples

        """
        # Create N samples
        if num_samples is None:
            num_samples = self.num_samples

        input_sample_set = self.random_sample_set(sample_type, input_obj,
                                                  num_samples, criterion, globalize)
        if param_ref is not None:
            input_sample_set.set_reference_value(param_ref)
        return self.compute_QoI_and_create_discretization(input_sample_set,
                                                          savefile, globalize)

    def add_qoi(self, discretization, data=None,
                savefile=None, globalize=True):
        """
        Add a column of data to the output sample set, update the
        reference parameter, dimension information.
        :param disc: A :class:`bet.sample.discretization` object with
        :type disc: :class:`~bet.sample.discretization` for problem set

        :rtype: :class:`~bet.sample.discretization`
        :returns: :class:`~bet.sample.discretization` object which contains
            input and output sample sets.
        """
        disc = discretization.copy()  # do not overwrite original
        Q_ref = disc._output_sample_set._reference_value

        # Take advantage of parallelism in other method to map values
        input_sample_set = disc._input_sample_set
        new = self.compute_QoI_and_create_discretization(input_sample_set,
                                                         None, globalize)

        # Append output values
        num_new_obs = new._output_sample_set.get_dim()
        num_old_obs = disc._output_sample_set.get_dim()
        new._output_sample_set._dim = num_old_obs + num_new_obs
        new_outputs = new._output_sample_set._values  # reference
        old_outputs = disc._output_sample_set._values

        new_outputs = np.column_stack((old_outputs, new_outputs))
        new._output_sample_set.set_values(new_outputs)

        # If reference input specified, use it to set output reference
        new_ref_val = new._output_sample_set._reference_value
        if new_ref_val is not None:
            if not isinstance(new_ref_val, collections.Iterable):
                new_ref_val = np.array(new_ref_val).ravel()
            new_ref_val = np.concatenate((Q_ref.ravel(), new_ref_val))
            new._output_sample_set.set_reference_value(new_ref_val)

        if globalize:
            new._output_sample_set.global_to_local()

        new._iteration = discretization._iteration
        new._setup = discretization._setup.copy()
        new.iterate()
        new._setup[new._iteration]['model'] = self.lb_model
        # if inds is None, keep using all the data.
        if discretization._setup[discretization._iteration]['ind'] is not None:
            # use just new data by default if previous inds existed
            new._setup[new._iteration]['ind'] = np.arange(
                num_new_obs) + num_old_obs
        else:  # if None, copy.
            new._setup[new._iteration]['ind'] = None

        # Noisy (raw) data passed, no reference output value needed.
        if disc._output_probability_set is not None:
            out_prob_set = sample.sample_set(num_old_obs + num_new_obs)
            new.set_output_probability_set(out_prob_set)
            if data is None:  # otherwise, append values
                if new_ref_val is not None:  # ref input must be specified
                    msg = "Setting data for new discretization"
                    msg += "Using reference value and noise distribution."
                    logging.warning(msg)
                    new.set_data_from_reference()
                else:
                    msg = "No data and no reference output"
                    msg += "for new output_probability_set."
                    logging.warning(msg)
            else:  # data that is passed is assumed to be noisy.
                ref = disc._output_probability_set._reference_value
                data = np.concatenate((ref.ravel(), data.ravel()))
            new._output_probability_set.set_reference_value(data)

        mdat = dict()
        if savefile is not None:
            self.save(mdat, savefile, new, globalize=globalize)

        return new
