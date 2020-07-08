# Copyright (C) 2014-2020 The BET Development Team

"""
This module contains the main data structures and exceptions for BET. Notably:

* :class:`~bet.sample.sample_set_base` provides the basic data structure for input and output sets
* :class:`~bet.sample.sample_set` is the default sample set.
* :class:`~bet.sample.voronoi_sample_set` is a sample set based on a Voronoi discretization (same as default).
* :class:`~bet.sample.rectangle_sample_set` is a sample set based on a hyper-rectangle.
* :class:`~bet.sample.ball_sample_set` is a sample set based on balls in R^n
* :class:`~bet.sample.cartesian_sample_set` is a sample set based on a Cartesian grid.
* :class:`~bet.sample.discretization` provides the basic data structure for and input to output stochastic map.
* :class:`~bet.sample.length_not_matching` is an Exception class.
* :class:`~bet.sample.dim_not_matching` is an Exception class.
* :func:`~bet.evaluate_pdf` evaluates probability density functions.
* :func:`~bet.evaluate_pdf_marginal` evaluates marginal probability density functions.

"""

import os
import logging
import copy
import glob
import warnings
import numpy as np
import math as math
import numpy.linalg as linalg
import scipy.spatial as spatial
import scipy.stats
import bet
from bet.Comm import comm, MPI
import bet.util as util
import bet.sampling.LpGeneralizedSamples as lp


class length_not_matching(Exception):
    """
    Exception for when the length of the array is inconsistent.
    """


class dim_not_matching(Exception):
    """
    Exception for when the dimension of the array is inconsistent.
    """


class domain_not_matching(Exception):
    """
    Exception for when the domain does not match.
    """


class wrong_p_norm(Exception):
    """
    Exception for when the dimension of the array is inconsistent.
    """


class wrong_input(Exception):
    """
    Exception for when the input is of the wrong type.
    """

def evaluate_pdf(prob_type, prob_parameters, vals):
    """
    Evaluate the probability density function defined by `prob_type` and `prob_parameters`
    at points defined by `vals`.

    :param prob_type: Type of probability description. Options are 'kde' (weighted kernel
        density estimate), 'rv' (random variable), 'gmm' (Gaussian mixture model), and 'voronoi'.
    :type prob_type: str
    :param prob_parameters: Parameters that define the probability measure of type `prob_type`
    :param vals: Values at which to evaluate the PDF.
    :type vals: :class:`numpy.ndarray`
    :return: probability density evaluated at `vals`
    :rtype `numpy.ndarray`
    """
    dim = vals.shape[1]
    if prob_type == "kde":
        mar = np.ones((vals.shape[0], ))
        for i in range(dim):
            mar *= evaluate_pdf_marginal(prob_type, prob_parameters, vals, i)
        return mar
    elif prob_type == "rv":
        mar = np.ones((vals.shape[0],))
        for i in range(dim):
            mar *= evaluate_pdf_marginal(prob_type, prob_parameters, vals, i)
        return mar
    elif prob_type == "gmm":
        from scipy.stats import multivariate_normal
        means, covs, cluster_weights = prob_parameters
        mar = np.zeros((vals.shape[0],))
        num_clusters = len(cluster_weights)
        for i in range(num_clusters):
            mar += cluster_weights[i] * multivariate_normal.pdf(vals, means[i], covs[i])
        return mar
    elif prob_type == "voronoi":
        _, pt = prob_parameters.query(vals)
        return prob_parameters.get_densities()[pt]
    else:
        raise wrong_input("This type of probability density is not yet supported.")


def evaluate_pdf_marginal(prob_type, prob_parameters, vals, i):
    """
    Evaluate the marginal probability density function of index `i` defined by `prob_type`
    and `prob_parameters` at points defined by `vals`.

    :param prob_type: Type of probability description. Options are 'kde' (weighted kernel
        density estimate), 'rv' (random variable), 'gmm' (Gaussian mixture model), and 'voronoi'.
    :type prob_type: str
    :param prob_parameters: Parameters that define the probability measure of type `prob_type`
    :param vals: Values at which to evaluate the PDF.
    :type vals: :class:`numpy.ndarray`
    :param i: index of marginal
    :type i: int
    :return: marginal probability density evaluated at `vals`
    :rtype `numpy.ndarray`
    """
    if len(vals.shape) == 2:
        if vals.shape[1] == 1:
            x = vals[:, 0]
        else:
            x = vals[:, i]
    elif len(vals.shape) == 1:
        x = vals

    if prob_type == "kde":
        param_marginals, cluster_weights = prob_parameters
        num_clusters = len(cluster_weights)
        mar = np.zeros(x.shape[0])
        for j in range(num_clusters):
            mar += param_marginals[i][j](x) * cluster_weights[j]
        return mar
    elif prob_type == "rv":
        import scipy.stats as stats
        rv = prob_parameters
        rv_continuous = getattr(stats, rv[i][0])
        args = rv[i][1]
        mar = rv_continuous.pdf(x, **args)
        return mar
    elif prob_type == 'gmm':
        import scipy.stats as stats
        means, covs, cluster_weights = prob_parameters
        mar = np.zeros(x.shape)
        num_clusters = len(cluster_weights)
        for j in range(num_clusters):
            mar += stats.norm.pdf(x, loc=means[j][i], scale=(covs[j][i, i] ** 0.5)) * cluster_weights[j]
        return mar
    elif prob_type == 'voronoi':
        from scipy.stats import gaussian_kde
        logging.warning("Using kernel density estimate to estimate marginal PDF.")
        sam_set = prob_parameters
        kde = gaussian_kde(sam_set.get_values()[:, i], weights=sam_set.get_probabilities())
        return kde(vals.T)
    else:
        raise wrong_input("This type of probability density is not yet supported.")


class sample_set_base(object):
    """

    A data structure containing values that define a set of samples.

    """
    # fields defining the object
    meta_fields = ['_bounding_box', '_densities', '_densities_local', '_dim', '_domain', '_domain_original',
                   '_error_estimates', '_error_estimates_local', '_error_id', '_error_id_local', '_jacobians',
                   '_jacobians_local', '_kdtree_values', '_kdtree_values_local', '_left', '_left_local',
                   '_local_index', '_normalized_radii', '_normalized_radii_local', '_p_norm', '_probabilities',
                   '_probabilities_local', '_radii', '_radii_local', '_reference_value', '_region', '_region_local',
                   '_right', '_right_local', '_values', '_values_local', '_volumes', '_volumes_local', '_width',
                   '_width_local', '_prob_type', '_prob_type_init', '_prob_parameters', '_prob_parameters_init',
                   '_label', '_labels', '_cluster_maps', '_weights', '_weights_init']
    #: List of global attribute names for attributes that are :class:`numpy.ndarray`
    array_names = ['_values', '_volumes', '_probabilities',
                   '_densities', '_jacobians',
                   '_error_estimates', '_right', '_left', '_width',
                   '_kdtree_values', '_radii', '_normalized_radii',
                   '_region', '_error_id']

    def __init__(self, dim):
        """

        Initialization

        :param int dim: Dimension of the space in which these samples reside.

        """
        #: Dimension of the sample space
        self._dim = dim
        #: :class:`numpy.ndarray` of sample values of shape (num, dim)
        self._values = None
        #: :class:`numpy.ndarray` of sample Voronoi volumes of shape (num,)
        self._volumes = None
        #: :class:`numpy.ndarray` of sample probabilities of shape (num,)
        self._probabilities = None
        #: :class:`numpy.ndarray` of sample densities of shape (num,)
        self._densities = None
        #: :class:`numpy.ndarray` of Jacobians at samples of shape (num,
        #: other_dim, dim)
        self._jacobians = None
        #: :class:`numpy.ndarray` of model error estimates at samples of shape
        #: (num, dim)
        self._error_estimates = None
        #: The sample domain :class:`numpy.ndarray` of shape (dim, 2)
        self._domain = None
        #: The sample domain before normalization :class:`numpy.ndarray` of shape (dim, 2)
        self._domain_original = None
        #: Bounding box of values, :class:`numpy.ndarray`of shape (dim, 2)
        self._bounding_box = None
        #: Local values for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num, dim)
        self._values_local = None
        #: Local volumes for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num,)
        self._volumes_local = None
        #: Local probabilities for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num,)
        self._probabilities_local = None
        #: Local densities for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num,)
        self._densities_local = None
        #: Local Jacobians for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num, other_dim, dim)
        self._jacobians_local = None
        #: Local error_estimates for parallelism, :class:`numpy.ndarray` of
        #: shape (local_num,)
        self._error_estimates_local = None
        #: Local indicies of global arrays, :class:`numpy.ndarray` of shape
        #: (local_num, dim)
        self._local_index = None
        #: :class:`scipy.spatial.KDTree`
        self._kdtree = None
        #: Values defining kd tree, :class:`numpy.ndarray` of shape (num, dim)
        self._kdtree_values = None
        #: Local values defining kd tree, :class:`numpy.ndarray` of
        #: shape (num, dim)
        self._kdtree_values_local = None
        #: Local pointwise left (local_num, dim)
        self._left_local = None
        #: Local pointwise right (local_num, dim)
        self._right_local = None
        #: Local pointwise width (local_num, dim)
        self._width_local = None

        #: Pointwise left (num, dim)
        self._left = None
        #: Pointwise right (num, dim)
        self._right = None
        #: Pointwise width (num, dim)
        self._width = None
        #: p-norm for discretization
        self._p_norm = 2.0
        #: :class:`numpy.ndarray` of sample radii of shape (num,)
        self._radii = None
        #: :class:`numpy.ndarray` of sample radii of shape (local_num,)
        self._radii_local = None
        #: :class:`numpy.ndarray` of normalized sample radii of shape (num,)
        self._normalized_radii = None
        #: :class:`numpy.ndarray` of normalized sample radii of shape
        #: (local_num,)
        self._normalized_radii_local = None
        #: :class:`numpy.ndarray` of integers marking regions of the domain
        self._region = None
        #: :class:`numpy.ndarray` of integers marking regions of the domain
        self._region_local = None
        #: :class:`numpy.ndarray` of error identifiers  of shape (num,)
        self._error_id = None
        #: :class:`numpy.ndarray` of error identifiers  of shape (local_num,)
        self._error_id_local = None
        #: :class:`numpy.ndarray` of reference value of shape (dim,)
        self._reference_value = None
        #: string defining type of probability
        self._prob_type = None
        #: parameters defining probability measure
        self._prob_parameters = None
        #: string defining type of initial probability
        self._prob_type_init = None
        #: parameters defining initial probability measure
        self._prob_parameters_init = None
        #: label for sample set
        self._label = None
        #: list of labels for each dimension of sample set
        self._labels = None
        #: list of arrays of cluster maps from LUQ package
        self._cluster_maps = None
        #: :class:`numpy.ndarray` of weights of shape (num,)
        self._weights = None
        #: :class:`numpy.ndarray` of initial weights of shape (num,)
        self._weights_init = None

    def __eq__(self, other):
        """
        Redefines equality to easily check the equivalence of two sample sets as having identical
        values in meta_fields.
        :param other: other object set to which compare
        :return: True for equality and False for not
        :rtype: bool
        """
        if self.__class__ == other.__class__:
            fields = self.meta_fields
            for field in fields:
                if type(getattr(self, field)) is np.ndarray:
                    if np.any(getattr(self, field) != getattr(other, field)):
                        return False
                elif field == "_cluster_maps":
                    cluster_maps = getattr(self, field)
                    cluster_maps_other = getattr(other, field)
                    if type(cluster_maps_other) != type(cluster_maps):
                        return False
                    if type(cluster_maps) is list:
                        for k in range(len(cluster_maps)):
                            if not np.array_equal(cluster_maps[k], cluster_maps_other[k]):
                                return False
                elif type(getattr(self, field)) is list:
                    compare = getattr(self, field) == getattr(other, field)
                    if type(compare) is bool:
                        if compare is False:
                            return False
                    else:
                        if compare.any() is False:
                            return False
                else:
                    if getattr(self, field) != getattr(other, field):
                        return False
            return True
        else:
            raise TypeError('Comparing object is not of the same type.')

    def save(self, filename, globalize=True):
        """
        Save the set using pickle.

        :param filename: filename to save to
        :type filename: str
        :param globalize: whether or not to globalize local variables before saving
        :type globalize: bool
        """
        util.save_object(save_set=self, file_name=filename, globalize=globalize)

    def normalize_domain(self):
        """

        Normalize the domain and attributes to a unit hyperbox.

        """
        if self._domain is None:
            logging.warning("Not normalizing because domain is not defined.")
            pass
        else:
            rescale_list = ['_jacobians', '_jacobians_local']
            for obj in rescale_list:
                val = getattr(self, obj)
                if val is not None:
                    val *= (self._domain[:, 1] - self._domain[:, 0])
                    setattr(self, obj, val)

            shift_list = ['_values', '_values_local',
                          '_error_estimates', '_error_estimates_local',
                          '_left', '_left_local',
                          '_right', '_right_local', '_reference_value']

            for obj in shift_list:
                val = getattr(self, obj)
                if val is not None:
                    val -= self._domain[:, 0]
                    val = val / (self._domain[:, 1] - self._domain[:, 0])
                    setattr(self, obj, val)

            self._domain_original = np.copy(self._domain)
            self._domain = np.repeat([[0.0, 1.0]], self._dim, axis=0)

    def undo_normalize_domain(self):
        """

        Undoes normalization of the domain and attributes if they have been
        normalized.

        """
        if self._domain is None:
            logging.warning(
                "Not undoing normalizing because domain is not defined.")
            pass
        elif self._domain_original is None:
            logging.warning("Doing nothing because set never normalized")
            pass
        else:
            rescale_list = ['_jacobians', '_jacobians_local']
            for obj in rescale_list:
                val = getattr(self, obj)
                if val is not None:
                    val = val / \
                        (self._domain_original[:, 1] -
                         self._domain_original[:, 0])
                    setattr(self, obj, val)

            shift_list = ['_values', '_values_local',
                          '_error_estimates', '_error_estimates_local',
                          '_left', '_left_local',
                          '_right', '_right_local', '_reference_value']
            for obj in shift_list:
                val = getattr(self, obj)
                if val is not None:
                    val = val * \
                        (self._domain_original[:, 1] -
                         self._domain_original[:, 0])

                    val = val + self._domain_original[:, 0]
                    setattr(self, obj, val)

            self._domain = np.copy(self._domain_original)
            self._domain_original = None

    def set_p_norm(self, p_norm):
        """
        Sets p-norm for sample set.

        :param float p_norm: p-norm to use

        """
        self._p_norm = p_norm

    def get_p_norm(self):
        """
        Returns p-norm for sample set
        """
        return self._p_norm

    def set_cluster_maps(self, cluster_maps):
        """
        Sets cluster maps (generally coming from LUQ).

        :param cluster_maps: List of arrays containing values in each cluster.
        :type cluster_maps: list
        """
        self._cluster_maps = cluster_maps

    def get_cluster_maps(self):
        """
        Returns cluster maps.
        """
        return self._cluster_maps

    def set_label(self, label):
        """
        Sets label for set.
        :param label: Label for set.
        :type label: str
        """
        self._label = label

    def get_label(self):
        """
        Returns label for set.
        """
        return self._label

    def set_labels(self, labels):
        """
        Sets labels for each dimension of set.
        :param labels: list or tuple containing strings which label parameters in each dimension.
        :type labels: list or tuple of length `dim`
        :return:
        """
        self._labels = labels

    def get_labels(self):
        """
        Returns labels for each dimension of set.
        """
        return self._labels

    def set_weights(self, weights):
        """
        Set weights for samples
        :type weights: :class:`numpy.ndarray` of shape (num,)
        :param weights: weights of samples
        """
        self._weights = weights

    def get_weights(self):
        """
        Returns weights of samples.
        """
        return self._weights

    def set_weights_init(self, weights):
        """
        Set initial weights for samples
        :type weights: :class:`numpy.ndarray` of shape (num,)
        :param weights: initial weights of samples
        """
        self._weights_init = weights

    def get_weights_init(self):
        """
        Returns initial weights of samples
        """
        return self._weights_init

    def set_prob_type_init(self, prob_type_init):
        """
        Set the type of initial probability measure.
        :param prob_type_init: Type of initial probability measure ('kde', 'gmm', 'voronoi', 'rv')
        :type prob_type_init: str
        """
        self._prob_type_init = prob_type_init

    def get_prob_type_init(self):
        """
        Returns the type of initial probability measure.
        """
        return self._prob_type_init

    def set_prob_parameters_init(self, prob_parameters_init):
        """
        Set initial probability measure parameters.
        :param prob_parameters_init:  Initial probability measure parameters.
        """
        self._prob_parameters_init = prob_parameters_init

    def get_prob_parameters_init(self):
        """
        Returns initial probability measure parameters.
        """
        return self._prob_parameters_init

    def set_prob_type(self, prob_type):
        """
        Set the type of updated probability measure.
        :param prob_type: Type of updated probability measure ('kde', 'gmm', 'voronoi', 'rv')
        :type prob_type: str
        """
        self._prob_type = prob_type

    def get_prob_type(self):
        """
        Returns the type of updated probability measure.
        """
        return self._prob_type

    def set_prob_parameters(self, prob_parameters):
        """
        Set updated probability measure parameters.
        :param prob_parameters:  Updated probability measure parameters.
        """
        self._prob_parameters = prob_parameters

    def get_prob_parameters(self):
        """
        Returns the updated probability measure parameters.
        """
        return self._prob_parameters

    def set_reference_value(self, ref_val):
        """
        Sets reference value for sample set.

        :param ref_val: reference value
        :type ref_val: :class:`numpy.ndarray` of shape (dim,)
        """
        if ref_val.shape != (self._dim,):
            raise dim_not_matching("Reference value is of wrong dimension.")

        self._reference_value = ref_val

    def get_reference_value(self):
        """
        Returns the reference value of a sample set.
        """
        return self._reference_value

    def set_region(self, region):
        """
        Sets region for sample set.

        :param region: array of regions
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._region = region

    def get_region(self):
        """
        Returns region.
        """
        return self._region

    def set_region_local(self, region):
        """
        Sets local region for sample set.

        :param region: array of regions
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._region_local = region

    def get_region_local(self):
        """
        Returns local region.
        """
        return self._region_local

    def set_error_id(self, error_id):
        """
        Sets error_id for sample set.

        :param error_id: array of error identifiers
        :type error_id: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._error_id = error_id

    def get_error_id(self):
        """
        Returns error identifiers.
        """
        return self._error_id

    def set_error_id_local(self, error_id):
        """
        Sets local error id for sample set.

        :param error_id: array of error identifiers
        :type error_id: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._error_local = error_id

    def get_error_id_local(self):
        """
        Returns local error identifier.
        """
        return self._error_id_local

    def update_bounds(self, num=None):
        """
        Creates ``self._right``, ``self._left``, ``self._width``.

        :param int num: Determines shape of pointwise bounds (num, dim)

        """
        if num is None:
            num = self._values.shape[0]
        self._left = np.repeat([self._domain[:, 0]], num, 0)
        self._right = np.repeat([self._domain[:, 1]], num, 0)
        self._width = self._right - self._left

    def update_bounds_local(self, local_num=None):
        """
        Creates local versions of ``self._right``, ``self._left``,
        ``self._width`` (``self._right_local``, ``self._left_local``,
        ``self._width_local``).

        :param int local_num: Determines shape of local pointwise bounds
            (local_num, dim)

        """
        if local_num is None:
            local_num = self._values_local.shape[0]
        self._left_local = np.repeat([self._domain[:, 0]], local_num, 0)
        self._right_local = np.repeat([self._domain[:, 1]], local_num, 0)
        self._width_local = self._right_local - self._left_local

    def append_values(self, values):
        """
        Appends the values in ``_values`` to ``self._values``.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values: values to append
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._values = np.concatenate((self._values,
                                       util.fix_dimensions_data(values, self._dim)), 0)

    def append_values_local(self, values_local):
        """
        Appends the values in ``_values_local`` to ``self._values``.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values_local: values to append
        :type values_local: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._values_local = np.concatenate((self._values_local,
                                             util.fix_dimensions_data(values_local, self._dim)), 0)

    def clip(self, cnum):
        """
        Creates and returns a sample set with the the first `cnum`
        entries of the sample set.

        :param int cnum: number of values of sample set to return

        :rtype: :class:`~bet.sample.sample_set`
        :returns: the clipped sample set

        """
        sset = self.copy()
        sset.check_num()
        if sset._values is None:
            sset.local_to_global()
        for array_name in self.array_names:
            current_array = getattr(sset, array_name)
            if current_array is not None:
                new_array = current_array[0:cnum]
                setattr(sset, array_name, new_array)
        if sset._values_local is not None:
            sset.global_to_local()
        sset.set_kdtree()
        return sset

    def check_num(self):
        """

        Checks that the number of entries in ``self._values``,
        ``self._volumes``, ``self._probabilities``, ``self._jacobians``, and
        ``self._error_estimates`` all match (assuming the named array exists).

        :rtype: int
        :returns: num

        """
        num = None
        for array_name in self.array_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                if num is None:
                    num = current_array.shape[0]
                    first_array = array_name
                else:
                    if num != current_array.shape[0]:
                        errortxt = "length of {} inconsistent with {}"
                        raise length_not_matching(errortxt.format(array_name,
                                                                  first_array))
        if self._values is not None and self._values.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")

        if num is None:
            num_local = self.check_num_local()
            if num_local is None:
                num_local = 0
            num = comm.allreduce(num_local, op=MPI.SUM)

        return num

    def check_num_local(self):
        """

        Checks that the number of entries in ``self._values_local``,
        ``self._volumes_local``, ``self._probabilities_local``,
        ``self._jacobians_local``, and ``self._error_estimates_local``
        all match (assuming the named array exists).

        :rtype: int
        :returns: num

        """
        num = None
        for array_name in self.array_names:
            array_name_local = array_name + "_local"
            current_array = getattr(self, array_name_local)
            if current_array is not None:
                if num is None:
                    num = current_array.shape[0]
                    first_array = array_name
                else:
                    if num != current_array.shape[0]:
                        errortxt = "length of {} inconsistent with {}"
                        raise length_not_matching(errortxt.format(array_name,
                                                                  first_array))
        if self._values is not None and self._values.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")

        return num

    def get_dim(self):
        """

        Return the dimension of the sample space.

        :rtype: int
        :returns: Dimension of the sample space.

        """
        return self._dim

    def set_bounding_box(self):
        """
        Set the bounding box of the values.
        """
        mins = np.min(self._values, axis=0)
        maxes = np.max(self._values, axis=0)
        self._bounding_box = np.vstack((mins, maxes)).transpose()

    def get_bounding_box(self):
        """
        Get the bounding box of the values.
        """
        if self._bounding_box is None:
            self.set_bounding_box()
        return self._bounding_box

    def set_values(self, values):
        """
        Sets the sample values.

        :param values: sample values
        :type values: :class:`numpy.ndarray` of shape (num, dim)

        """
        self._values = util.fix_dimensions_data(values, self._dim)
        if self._values.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")

    def get_values(self):
        """
        Returns sample values.

        :rtype: :class:`numpy.ndarray`
        :returns: sample values

        """
        return self._values

    def set_domain(self, domain):
        """
        Sets the domain.

        :param domain: Sample domain
        :type domain: :class:`numpy.ndarray` of shape (dim, 2)

        """
        if (domain.shape[0], 2) != (self._dim, 2):
            raise dim_not_matching("dimension of values incorrect")
        else:
            self._domain = domain

    def get_domain(self):
        """
        Returns the sample domain,

        :rtype: :class:`numpy.ndarray` of shape (dim, 2)
        :returns: Sample domain

        """
        return self._domain

    def set_volumes(self, volumes):
        """
        Sets sample cell volumes.

        :type volumes: :class:`numpy.ndarray` of shape (num,)
        :param volumes: sample cell volumes

        """
        self._volumes = volumes

    def get_volumes(self):
        """
        Returns sample cell volumes.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample cell volumes

        """
        return self._volumes

    def set_probabilities(self, probabilities):
        """
        Set sample probabilities.

        :type probabilities: :class:`numpy.ndarray` of shape (num,)
        :param probabilities: sample probabilities

        """
        self._probabilities = probabilities

    def get_probabilities(self):
        """
        Returns sample probabilities.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample probabilities

        """
        return self._probabilities

    def set_densities(self, densities=None):
        """
        Set sample densities.

        :type densities: :class:`numpy.ndarray` of shape (num,)
        :param densities: sample densities

        """
        if densities is not None:
            self._densities = densities
        else:
            logging.warning("Setting densities with probability/volume.")
            if self._domain is None:
                total_vol = 1.0
            else:
                total_vol = np.product(self._domain[:, 1] - self._domain[:, 0])
            probs = self._probabilities
            vols = self._volumes * total_vol
            self._densities = probs / vols

    def get_densities(self):
        """
        Returns sample densities.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample densities

        """
        return self._densities

    def pdf(self, vals):
        """
        Evaluate the probability density function of the updated probability measure at values.
        :param vals: Values at which to evaluated the PDF.
        :type vals: :class:`numpy.ndarray` of shape (num_vals, dim)
        :return probability densities
        :rtype :class:`numpy.ndarray` of shape (num_vals, )
        """
        if len(vals.shape) == 1:
            vals = np.reshape(vals, (vals.shape[0], 1))
        if vals.shape[1] != self._dim:
            raise dim_not_matching("Array does not have the correct dimension.")

        if self._prob_type == 'voronoi':
            if self._probabilities_local is None and self._probabilities is None:
                raise wrong_input("Missing probabilities for Voronoi cells.")
            if self._densities_local is None:
                if self._volumes_local is None:
                    logging.warning("Using Monte Carlo Assumption to Estimate Volumes.")
                    self.estimate_volume_mc(globalize=False)
                self.set_densities_local(self._probabilities_local/self._volumes_local)
            self.local_to_global()
            return evaluate_pdf(self._prob_type, self, vals)
        else:
            return evaluate_pdf(self._prob_type, self._prob_parameters, vals)

    def pdf_init(self, vals):
        """
        Evaluate the probability density function of the initial probability measure at values.
        :param vals: Values at which to evaluated the PDF.
        :type vals: :class:`numpy.ndarray` of shape (num_vals, dim)
        :return probability densities
        :rtype :class:`numpy.ndarray` of shape (num_vals, )
        """
        if len(vals.shape) == 1:
            vals = np.reshape(vals, (vals.shape[0], 1))
        if vals.shape[1] != self._dim:
            raise dim_not_matching("Array does not have the correct dimension.")
        if self._prob_type_init == "voronoi":
            raise wrong_input("Voronoi probability not valid for initial PDF.")
        else:
            return evaluate_pdf(self._prob_type_init, self._prob_parameters_init, vals)

    def marginal_pdf(self, vals, i):
        """
        Evaluate the marginal (with index `i`) probability density function of the updated
        probability measure at values.

        :param vals: Values at which to evaluated the PDF.
        :type vals: :class:`numpy.ndarray` of shape (num_vals, dim) or (num_vals, )
        :param i: index defining marginal
        :type i: int
        :return probability densities
        :rtype :class:`numpy.ndarray` of shape (num_vals, )
        """
        if self._prob_type == 'voronoi':
            if self._probabilities_local is None and self._probabilities is None:
                raise wrong_input("Missing probabilities for Voronoi cells.")
            if self._probabilities is None:
                self.local_to_global()
            return evaluate_pdf_marginal(self._prob_type, self, vals, i)
        else:
            return evaluate_pdf_marginal(self._prob_type, self._prob_parameters, vals, i)

    def marginal_pdf_init(self, vals, i):
        """
        Evaluate the marginal (with index `i`) probability density function of the initial
        probability measure at values.

        :param vals: Values at which to evaluated the PDF.
        :type vals: :class:`numpy.ndarray` of shape (num_vals, dim) or (num_vals, )
        :param i: index defining marginal
        :type i: int
        :return probability densities
        :rtype :class:`numpy.ndarray` of shape (num_vals, )
        """
        if self._prob_type_init == "voronoi":
            raise wrong_input("Voronoi probability not valid for initial PDF.")
        else:
            return evaluate_pdf_marginal(self._prob_type_init, self._prob_parameters_init, vals, i)

    def set_jacobians(self, jacobians):
        """
        Returns sample jacobians.

        :type jacobians: :class:`numpy.ndarray` of shape (num, other_dim, dim)
        :param jacobians: sample jacobians

        """
        self._jacobians = jacobians

    def get_jacobians(self):
        """
        Returns sample jacobians.

        :rtype: :class:`numpy.ndarray` of shape (num, other_dim, dim)
        :returns: sample jacobians

        """
        return self._jacobians

    def append_jacobians(self, new_jacobians):
        """
        Appends the ``new_jacobians`` to ``self._jacobians``.

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_jacobians: New jacobians to append.
        :type new_jacobians: :class:`numpy.ndarray` of shape (num, other_dim,
            dim)

        """
        self._jacobians = np.concatenate((self._jacobians, new_jacobians),
                                         axis=0)

    def set_error_estimates(self, error_estimates):
        """
        Returns sample error estimates.

        :type error_estimates: :class:`numpy.ndarray` of shape (num,)
        :param error_estimates: sample error estimates

        """
        self._error_estimates = error_estimates

    def get_error_estimates(self):
        """
        Returns sample error_estimates.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample error_estimates

        """
        return self._error_estimates

    def append_error_estimates(self, new_error_estimates):
        """
        Appends the ``new_error_estimates`` to ``self._error_estimates``.

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_error_estimates: New error_estimates to append.
        :type new_error_estimates: :class:`numpy.ndarray` of shape (num,)

        """
        self._error_estimates = np.concatenate((self._error_estimates,
                                                new_error_estimates), axis=0)

    def set_values_local(self, values_local):
        """
        Sets the local sample values.

        :param values_local: sample local values
        :type values_local: :class:`numpy.ndarray` of shape (local_num, dim)

        """
        self._values_local = util.fix_dimensions_data(values_local, self._dim)
        if len(self._values_local.shape) > 1 and \
                self._values_local.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")

    def set_kdtree(self):
        """
        Creates a :class:`scipy.spatial.KDTree` for this set of samples.
        """
        self._kdtree = spatial.KDTree(self._values)
        self._kdtree_values = self._kdtree.data

    def get_kdtree(self):
        """
        Returns a :class:`scipy.spatial.KDTree` for this set of samples.

        :rtype: :class:`scipy.spatial.KDTree`
        :returns: :class:`scipy.spatial.KDTree` for this set of samples.

        """
        return self._kdtree

    def get_values_local(self):
        """
        Returns sample local values.

        :rtype: :class:`numpy.ndarray`
        :returns: sample local values

        """
        return self._values_local

    def set_volumes_local(self, volumes_local):
        """
        Sets local sample cell volumes.

        :type volumes_local: :class:`numpy.ndarray` of shape (num,)
        :param volumes_local: local sample cell volumes

        """
        self._volumes_local = volumes_local

    def get_volumes_local(self):
        """
        Returns sample local volumes.

        :rtype: :class:`numpy.ndarray`
        :returns: sample local volumes

        """
        return self._volumes_local

    def set_probabilities_local(self, probabilities_local):
        """
        Set sample local probabilities.

        :type probabilities_local: :class:`numpy.ndarray` of shape (num,)
        :param probabilities_local: local sample probabilities

        """
        self._probabilities_local = probabilities_local

    def get_probabilities_local(self):
        """
        Returns sample local probabilities.

        :rtype: :class:`numpy.ndarray`
        :returns: sample local probabilities

        """

        return self._probabilities_local

    def set_densities_local(self, densities_local=None):
        """
        Set sample local densities.

        :type densities_local: :class:`numpy.ndarray` of shape (num,)
        :param densities_local: local sample densities

        """
        if densities_local is not None:
            self._densities_local = densities_local
        else:
            msg = "Setting densities with probability/volume."
            logging.warning(msg)
            probs = self._probabilities_local
            vols = self._volumes_local
            self._densities_local = probs / vols

    def get_densities_local(self):
        """
        Returns sample local densities.

        :rtype: :class:`numpy.ndarray`
        :returns: sample local densities

        """

        return self._densities_local

    def set_jacobians_local(self, jacobians_local):
        """
        Returns local sample jacobians.

        :type jacobians_local: :class:`numpy.ndarray` of shape (num, other_dim,
            dim)
        :param jacobians_local: local sample jacobians

        """
        self._jacobians_local = jacobians_local

    def get_jacobians_local(self):
        """
        Returns local sample jacobians.

        :rtype: :class:`numpy.ndarray` of shape (num, other_dim, dim)
        :returns: local sample jacobians

        """
        return self._jacobians_local

    def set_error_estimates_local(self, error_estimates_local):
        """
        Returns local sample error estimates.

        :type error_estimates_local: :class:`numpy.ndarray` of shape (num,)
        :param error_estimates_local: local sample error estimates

        """
        self._error_estimates_local = error_estimates_local

    def get_error_estimates_local(self):
        """
        Returns sample error_estimates_local.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample error_estimates_local

        """
        return self._error_estimates_local

    def local_to_global(self):
        """
        Makes global arrays from available local ones.
        """
        for array_name in self.array_names:
            current_array_local = getattr(self, array_name + "_local")
            if current_array_local is not None:
                setattr(self, array_name,
                        util.get_global_values(current_array_local))

    def query(self, x, k=1):
        """
        Identify which value points x are associated with for discretization.

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param int k: number of nearest neighbors to return

        """
        pass

    def calculate_volumes(self):
        """
        Calculate the volumes of cells. Depends on sample set type.
        """

    def estimate_volume(self, n_mc_points=int(1E4)):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration.

        :param int n_mc_points: If estimate is True, number of MC points to use
        """
        num = self.check_num()
        n_mc_points = int(n_mc_points)
        n_mc_points_local = int(n_mc_points / comm.size) + \
            int(comm.rank < n_mc_points % comm.size)
        width = self._domain[:, 1] - self._domain[:, 0]
        mc_points = width * np.random.random((n_mc_points_local,
                                              self._domain.shape[0])) + self._domain[:, 0]
        (_, emulate_ptr) = self.query(mc_points)
        vol = np.zeros((num,))
        for i in range(num):
            vol[i] = np.sum(np.equal(emulate_ptr, i))
        cvol = np.copy(vol)
        comm.Allreduce([vol, MPI.DOUBLE], [cvol, MPI.DOUBLE], op=MPI.SUM)
        vol = cvol
        vol = vol / float(n_mc_points)
        self._volumes = vol
        self.global_to_local()

    def estimate_volume_emulated(self, emulated_sample_set):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration.

        .. note ::

            This could be re-written to just use an ``emulated_ii_ptr`` instead
            of an ``emulated_sample_set``.

        :param emulated_sample_set: The set of samples used to approximate the
            volume measure.
        :type emulated_sample_set: :class:`bet.sample.sample_set_base`

        """
        num = self.check_num()

        if emulated_sample_set._values_local is None:
            emulated_sample_set.global_to_local()

        (_, emulate_ptr) = self.query(emulated_sample_set._values_local)

        vol = np.zeros((num,))
        for i in range(num):
            vol[i] = np.sum(np.equal(emulate_ptr, i))
        cvol = np.copy(vol)
        comm.Allreduce([vol, MPI.DOUBLE], [cvol, MPI.DOUBLE], op=MPI.SUM)
        num_emulate = emulated_sample_set._values_local.shape[0]
        num_emulate = comm.allreduce(num_emulate, op=MPI.SUM)
        vol = cvol
        vol = vol / float(num_emulate)
        self._volumes = vol
        self.global_to_local()

    def estimate_volume_mc(self, globalize=True):
        """
        Give all cells the same volume fraction based on the Monte Carlo
        assumption.
        """
        num = self.check_num()
        if globalize:
            self._volumes = 1.0 / float(num) * np.ones((num,))
            self.global_to_local()
        else:
            num_local = self.check_num_local()
            self._volumes_local = 1.0 / float(num) * np.ones((num_local,))

    def global_to_local(self):
        """
        Makes local arrays from available global ones.
        """
        num = self.check_num()
        global_index = np.arange(num, dtype=np.int)
        self._local_index = np.array_split(global_index, comm.size)[comm.rank]
        for array_name in self.array_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(self, array_name + "_local",
                        np.array_split(current_array, comm.size)[comm.rank])
        comm.barrier()

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: Copy of this :class:`~bet.sample.sample_set_base`

        """
        # my_copy = type(self)(self.get_dim())
        # for array_name in self.all_ndarray_names:
        #     current_array = getattr(self, array_name)
        #     if current_array is not None:
        #         setattr(my_copy, array_name,
        #                 np.copy(current_array))
        # for vector_name in self.vector_names:
        #     if vector_name is not "_dim":
        #         current_vector = getattr(self, vector_name)
        #         if current_vector is not None:
        #             setattr(my_copy, vector_name, np.copy(current_vector))
        # if self._kdtree is not None:
        #     my_copy.set_kdtree()
        # return my_copy
        import copy
        return copy.deepcopy(self)

    def shape(self):
        """

        Returns the shape of ``self._values``

        :rtype: tuple
        :returns: (num, dim)

        """
        return self._values.shape

    def shape_local(self):
        """

        Returns the shape of ``self._values_local``

        :rtype: tuple
        :returns: (local_num, dim)

        """
        return self._values_local.shape


class voronoi_sample_set(sample_set_base):
    """

    A data structure containing arrays specific to a set of samples defining
    a Voronoi tesselation.

    """

    def query(self, x, k=1):
        """
        Identify which value points x are associated with for discretization.

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param int k: number of nearest neighbors to return

        :rtype: tuple
        :returns: (dist, ptr)
        """
        if self._kdtree is None:
            self.set_kdtree()
        else:
            self.check_num()

        (dist, ptr) = self._kdtree.query(x, p=self._p_norm, k=k)
        return (dist, ptr)

    def exact_volume_1D(self):
        r"""

        Exactly calculates the volume fraction of the Voronoi cells.
        Specifically we are calculating
        :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.

        """
        self.check_num()
        if self._dim != 1:
            raise dim_not_matching("Only applicable for 1D domains.")

        # sort the samples
        sort_ind = np.squeeze(np.argsort(self._values, 0))
        sorted_samples = self._values[sort_ind]
        domain_width = self._domain[:, 1] - self._domain[:, 0]

        # determine the mid_points which are the edges of the associated
        # voronoi cells and bound the cells by the domain
        edges = np.concatenate(([self._domain[:, 0]],
                                (sorted_samples[:-1, :] +
                                 sorted_samples[1:, :]) * .5,
                                [self._domain[:, 1]]))
        # calculate difference between right and left of each cell
        # and renormalize
        sorted_lam_vol = np.squeeze(edges[1:, :] - edges[:-1, :])
        lam_vol = np.zeros(sorted_lam_vol.shape)
        lam_vol[sort_ind] = sorted_lam_vol
        lam_vol = lam_vol / domain_width
        self._volumes = lam_vol
        self.global_to_local()

    def exact_volume_2D(self, side_ratio=0.25):
        r"""

        Exactly calculates the volume fraction of the Voronoi cells.
        Specifically we are calculating
        :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.

        :param float side_ratio: ratio of width to reflect across boundary

        """
        # Check inputs
        num = self.check_num()
        if self._dim != 2:
            raise dim_not_matching("Only applicable for 2D domains.")
        new_samp = np.copy(self._values)

        # Add points around boundary
        add_points = np.less(self._values[:, 0],
                             self._domain[0][0] +
                             side_ratio * (self._domain[0][1] -
                                           self._domain[0][0]))
        points_new = self._values[add_points, :]
        points_new[:, 0] = self._domain[0][0] - \
            (points_new[:, 0] - self._domain[0][0])
        new_samp = np.vstack((new_samp, points_new))

        add_points = np.greater(self._values[:, 0],
                                self._domain[0][1] -
                                side_ratio * (self._domain[0][1] -
                                              self._domain[0][0]))
        points_new = self._values[add_points, :]
        points_new[:, 0] = self._domain[0][1] + \
            (-points_new[:, 0] + self._domain[0][1])
        new_samp = np.vstack((new_samp, points_new))

        add_points = np.less(self._values[:, 1],
                             self._domain[1][0] +
                             side_ratio * (self._domain[1][1] -
                                           self._domain[1][0]))
        points_new = self._values[add_points, :]
        points_new[:, 1] = self._domain[1][0] - \
            (points_new[:, 1] - self._domain[1][0])
        new_samp = np.vstack((new_samp, points_new))

        add_points = np.greater(self._values[:, 1],
                                self._domain[1][1] -
                                side_ratio * (self._domain[1][1] -
                                              self._domain[1][0]))
        points_new = self._values[add_points, :]
        points_new[:, 1] = self._domain[1][1] + \
            (-points_new[:, 1] + self._domain[1][1])
        new_samp = np.vstack((new_samp, points_new))

        # Make Voronoi diagram and calculate volumes
        vor = spatial.Voronoi(new_samp)
        local_index = np.arange(0 + comm.rank, num, comm.size)
        local_array = np.array(local_index, dtype='int64')
        lam_vol_local = np.zeros(local_array.shape)
        for I, i in enumerate(local_index):
            val = vor.point_region[i]
            region = vor.regions[val]
            if not -1 in region:
                polygon = [vor.vertices[k] for k in region]
                delan = spatial.Delaunay(polygon)
                simplices = delan.points[delan.simplices]
                vol = 0.0
                for j in range(simplices.shape[0]):
                    mat = np.empty((self._dim, self._dim))
                    mat[:, :] = (simplices[j][1::, :] -
                                 simplices[j][0, :]).transpose()
                    vol += abs(1.0 / math.factorial(self._dim)
                               * linalg.det(mat))
                lam_vol_local[I] = vol
        lam_size = np.prod(self._domain[:, 1] - self._domain[:, 0])
        lam_vol_local = lam_vol_local / lam_size
        lam_vol_global = util.get_global_values(lam_vol_local)
        global_index = util.get_global_values(local_array)
        lam_vol = np.zeros(lam_vol_global.shape)
        self._volumes = np.zeros((num,))
        self._volumes[global_index] = lam_vol_global[:]
        self.global_to_local()

    def estimate_radii(self, n_mc_points=int(1E4), normalize=True):
        """
        Calculate the radii of cells approximately using Monte
        Carlo integration.

        .. todo::

           This currently presumes a uniform Lesbegue measure on the
           ``domain``. Currently the way this is written
           ``emulated_input_sample_set`` is NOT used to calculate the volume.
           This should at least be an option.

        :param int n_mc_points: If estimate is True, number of MC points to use
        :param bool normalize: estimate normalized radius

        """
        num = self.check_num()
        n_mc_points = int(n_mc_points)
        samples = np.copy(self.get_values())
        n_mc_points_local = int(n_mc_points / comm.size) + \
            int(comm.rank < n_mc_points % comm.size)

        # normalize the samples
        if normalize:
            self.update_bounds()
            samples = samples - self._left
            samples = samples / self._width
            self._left = None
            self._right = None
            self._width = None

        width = self._domain[:, 1] - self._domain[:, 0]
        mc_points = width * np.random.random((n_mc_points_local,
                                              self._domain.shape[0])) +\
            self._domain[:, 0]

        (_, emulate_ptr) = self.query(mc_points)

        if normalize:
            self.update_bounds(n_mc_points_local)
            mc_points = mc_points - self._left
            mc_points = mc_points / self._width
            self._left = None
            self._right = None
            self._width = None

        rad = np.zeros((num,))

        for i in range(num):
            rad[i] = np.max(np.linalg.norm(
                            mc_points[np.equal(emulate_ptr, i), :] -
                            samples[i, :], ord=self._p_norm, axis=1))

        crad = np.copy(rad)
        comm.Allreduce([rad, MPI.DOUBLE], [crad, MPI.DOUBLE], op=MPI.MAX)
        rad = crad

        if normalize:
            self._normalized_radii = rad
        else:
            self._radii = rad

        self.global_to_local()

    def estimate_radii_and_volume(self, n_mc_points=int(1E4), normalize=True):
        """
        Calculate the radii and volume faction of cells approximately using
        Monte Carlo integration.

        .. todo::

           This currently presumes a uniform Lesbegue measure on the
           ``domain``. Currently the way this is written
           ``emulated_input_sample_set`` is NOT used to calculate the volume.
           This should at least be an option.

        :param int n_mc_points: If estimate is True, number of MC points to use
        :param bool normalize: estimate normalized radius

        """
        num = self.check_num()
        n_mc_points = int(n_mc_points)
        samples = np.copy(self.get_values())
        n_mc_points_local = int(n_mc_points / comm.size) + \
            int(comm.rank < n_mc_points % comm.size)

        # normalize the samples
        if normalize:
            self.update_bounds()
            samples = samples - self._left
            samples = samples / self._width

        width = self._domain[:, 1] - self._domain[:, 0]
        mc_points = width * np.random.random((n_mc_points_local,
                                              self._domain.shape[0])) +\
            self._domain[:, 0]

        (_, emulate_ptr) = self.query(mc_points)

        if normalize:
            self.update_bounds(n_mc_points_local)
            mc_points = mc_points - self._left
            mc_points = mc_points / self._width
            self._left = None
            self._right = None
            self._width = None

        vol = np.zeros((num,))
        rad = np.zeros((num,))
        for i in range(num):
            vol[i] = np.sum(np.equal(emulate_ptr, i))
            rad[i] = np.max(np.linalg.norm(
                            mc_points[np.equal(emulate_ptr, i), :] -
                            samples[i, :], ord=self._p_norm, axis=1))

        crad = np.copy(rad)
        comm.Allreduce([rad, MPI.DOUBLE], [crad, MPI.DOUBLE], op=MPI.MAX)
        rad = crad

        if normalize:
            self._normalized_radii = rad
        else:
            self._radii = rad

        cvol = np.copy(vol)
        comm.Allreduce([vol, MPI.DOUBLE], [cvol, MPI.DOUBLE], op=MPI.SUM)
        vol = cvol
        vol = vol / float(n_mc_points)
        self._volumes = vol
        self.global_to_local()

    def estimate_local_volume(self, num_emulate_local=500,
                              max_num_emulate=int(1e4)):
        r"""

        Estimates the volume fraction of the Voronoi cells associated
        with ``samples``. Specifically we are calculating
        :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
        Here all of the samples are drawn from the generalized Lp uniform
        distribution.

        .. note ::

            If this :class:`~bet.sample.voronoi_sample_set` has exact/estimated
            radii of the Voronoi cell associated with each sample for a domain
            normalized to the unit hypercube (``_normalized_radii``). Note that
            these are not centroidal Voronoi tesselations meaning that the
            centroid is NOT the generator of the Voronoi cell. What we desire
            for the radius is actually
            :math:`sup_{\lambda \in \mathcal{V}_{i, N}} d_v(\lambda,
            \lambda^{(i)})`.

        .. todo ::

            When we move away from domains defined on hypercubes this will need
            to be updated to use whatever ``_in_domain`` method exists.

        Volume of the L-p ball is obtained from  Wang, X.. (2005). Volumes of
        Generalized Unit Balls. Mathematics Magazine, 78(5), 390-395.
        `DOI 10.2307/30044198 <http://doi.org/10.2307/30044198>`_

        :param int num_emulate_local: The number of emulated samples.
        :param int max_num_emulate: Maximum number of local emulated samples

        """
        self.check_num()
        # normalize the samples
        samples = np.copy(self.get_values())
        self.update_bounds()
        samples = samples - self._left
        samples = samples / self._width
        num_emulate_local = int(num_emulate_local)
        max_num_emulate = int(max_num_emulate)
        kdtree = spatial.KDTree(samples)

        # for each sample determine the appropriate radius of the Lp ball (this
        # should be the distance to the farthest neighboring Voronoi cell)
        # calculating this exactly is hard so we will estimate it as follows
        # TODO it is unclear whether to use min, mean, or the first n nearest
        # samples
        sample_radii = None
        if self._normalized_radii is not None:
            sample_radii = np.copy(self._normalized_radii)

        if sample_radii is None:
            num_mc_points = np.max([1e4, samples.shape[0] * 20])
            self.estimate_radii(n_mc_points=int(num_mc_points))
            sample_radii = 1.5 * np.copy(self._normalized_radii)
        if np.sum(sample_radii <= 0) > 0:
            # Calculate the pairwise distances
            if not np.isinf(self._p_norm):
                pairwise_distance = spatial.distance.pdist(samples,
                                                           p=self._p_norm)
            else:
                pairwise_distance = spatial.distance.pdist(samples,
                                                           p='chebyshev')
            pairwise_distance = spatial.distance.squareform(pairwise_distance)
            pairwise_distance_ma = np.ma.masked_less_equal(pairwise_distance,
                                                           0.)
            prob_est_radii = np.std(pairwise_distance_ma * .5, 0) * 2.
            # Calculate mean, std of pairwise distances
            # TODO this may be too large/small
            # Estimate radius as 2.*STD of the pairwise distance
            sample_radii[sample_radii <= 0] = prob_est_radii[sample_radii <= 0]

        # determine the volume of the Lp ball
        if not np.isinf(self._p_norm):
            sample_Lp_ball_vol = sample_radii**self._dim * \
                scipy.special.gamma(1 + 1. / self._p_norm) / \
                scipy.special.gamma(1 + float(self._dim) / self._p_norm)
        else:
            sample_Lp_ball_vol = (2.0 * sample_radii)**self._dim

        # Set up local arrays for parallelism
        self.global_to_local()
        lam_vol_local = np.zeros(self._local_index.shape)

        # parallize

        for i, iglobal in enumerate(self._local_index):
            samples_in_cell = 0
            total_samples = 10
            while samples_in_cell < num_emulate_local and \
                    total_samples < max_num_emulate:
                total_samples = total_samples * 10
                # Sample within an Lp ball until num_emulate_local samples are
                # present in the Voronoi cell
                local_lambda_emulate = \
                    lp.Lp_generalized_uniform(self._dim, total_samples,
                                              self._p_norm,
                                              scale=sample_radii[iglobal],
                                              loc=samples[iglobal])

                # determine the number of samples in the Voronoi cell
                # (intersected with the input_domain)
                if self._domain is not None:
                    inside = np.all(np.logical_and(
                        local_lambda_emulate >= 0.0,
                        local_lambda_emulate <= 1.0), 1)
                    local_lambda_emulate = local_lambda_emulate[inside]

                (_, emulate_ptr) = kdtree.query(local_lambda_emulate,
                                                p=self._p_norm,
                                                distance_upper_bound=sample_radii[iglobal])

                samples_in_cell = np.sum(np.equal(emulate_ptr, iglobal))

            # the volume for the Voronoi cell corresponding to this sample is
            # the the volume of the Lp ball times the ratio
            # "num_samples_in_cell/num_total_local_emulated_samples"
            lam_vol_local[i] = sample_Lp_ball_vol[iglobal] *\
                float(samples_in_cell) / float(total_samples)

        self.set_volumes_local(lam_vol_local)
        self.local_to_global()

        # normalize by the volume of the input_domain
        domain_vol = np.sum(self.get_volumes())
        self.set_volumes(self._volumes / domain_vol)
        self.set_volumes_local(self._volumes_local / domain_vol)

    def merge(self, sset):
        """
        Merges a given sample set with this one by merging the values.

        :param sset: Sample set object to merge with.
        :type sset: :class:`bet.sample.voronoi_sample_set`

        :rtype: :class:`bet.sample.voronoi_sample_set`
        :returns: Merged discretization
        """
        # check dimensions
        if self._dim != sset._dim:
            msg = "These sample sets must have the same dimension."
            raise dim_not_matching(msg)
        # check domain
        if self._domain is not None and sset._domain is not None:
            if not np.allclose(self._domain, sset._domain):
                msg = "These sample sets have different domains."
                raise domain_not_matching(msg)

        # create merged set
        mset = voronoi_sample_set(self._dim)

        # set domain
        if self._domain is not None:
            mset.set_domain(self._domain)
        elif sset._domain is not None:
            mset.set_domain(sset._domain)

        # merge and set values
        if self._values_local is None:
            self.global_to_local()
        if sset._values_local is None:
            sset.global_to_local()
        mset.set_values_local(np.concatenate((self._values_local,
                                              sset._values_local), 0))
        mset.local_to_global()
        return mset


class sample_set(voronoi_sample_set):
    """
    Set Voronoi cells as the default for now.
    """


class rectangle_sample_set(sample_set_base):
    r"""
    A data structure containing arrays specific to a set of samples defining a
    hyperrectangle discretization.

    A series of n hyperrectangles :math:`A_i \subset \Lambda` with
    :math:`A_i \cap A_j = \emptyset`
    for :math:`i \neq j`. The last entry represents the remainder
    :math:`\Lambda \setminus ( \cup_{i-1}^n A_i)`.

    """

    def setup(self, maxes, mins):
        """

        Initialization

        :param maxes: array or list of maxes for hyperrectangles
        :type maxes: iterable with components of length dim
        :param mins: array or list of mins for hyperrectangles
        :type mins: iterable with components of length dim

        """
        # Check dimensions
        if len(maxes) != len(mins):
            raise length_not_matching("Different number of maxes and mins")
        for i in range(len(maxes)):
            if (len(maxes[i]) != self._dim) or (len(mins[i]) != self._dim):
                msg = "Rectangle " + \
                    repr(i) + " has the wrong number of entries."
                raise length_not_matching(msg)

        values = np.zeros((len(maxes) + 1, self._dim))
        self._right = np.zeros((len(maxes) + 1, self._dim))
        self._left = np.zeros((len(mins) + 1, self._dim))
        for i in range(len(maxes)):
            values[i, :] = 0.5 * (np.array(maxes[i]) + np.array(mins[i]))
            self._right[i, :] = maxes[i]
            self._left[i, :] = mins[i]
        values[-1, :] = np.inf
        self._right[-1, :] = np.inf
        self._left[-1, :] = -np.inf
        self._width = self._right - self._left
        self.set_values(values)
        if len(maxes) > 1:
            msg = "If rectangles intersect on a set nonzero measure, "
            msg += "calculated values will be wrong."
            logging.warning(msg)
        self._region = np.arange(len(maxes) + 1)

    def update_bounds(self, num=None):
        """
        Does nothing for this type of sample set.

        """
        logging.warning(
            "Bounds cannot be updated for this type of sample set.")

    def update_bounds_local(self, num_local=None):
        """
        Does nothing for this type of sample set.

        """
        logging.warning(
            "Bounds cannot be updated for this type of sample set.")

    def append_values(self, values):
        """
        Does nothing for this type of sample_set.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values: values to append
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        msg = "Values cannot be appended for this type of sample set."
        logging.warning(msg)

    def append_values_local(self, values_local):
        """
        Does nothing for this type of sample_set.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values_local: values to append
        :type values_local: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        msg = "Values cannot be appended for this type of sample set."
        logging.warning(msg)

    def append_jacobians(self, new_jacobians):
        """
        Does nothing for this type of sample set.

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_jacobians: New jacobians to append.
        :type new_jacobians: :class:`numpy.ndarray` of shape (num, other_dim,
            dim)

        """
        msg = "Values cannot be appended for this type of sample set."
        logging.warning(msg)

    def append_error_estimates(self, new_error_estimates):
        """
        Does nothing for this type of sample set.

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_error_estimates: New error_estimates to append.
        :type new_error_estimates: :class:`numpy.ndarray` of shape (num,)

        """
        msg = "Values cannot be appended for this type of sample set."
        logging.warning(msg)

    def query(self, x, k=1):
        r"""
        Identify which value points x are associated with for discretization.
        Only returns the neighbors for which :math:`x_i \in A_k`. The distance
        is set to 0 if it is in the rectangle and infinity if it is not.
        It is only considered in or out.

        .. seealso::

            :meth:`scipy.spatial.KDTree.query`

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param int k: number of nearest neighbors to return
        :rtype: tuple
        :returns: (dist, ptr)

        """
        num = self.check_num()
        dist = np.inf * np.ones((x.shape[0], k), dtype=np.float)
        pt = (num - 1) * np.ones((x.shape[0], k), dtype=np.int)
        for i in range(num - 1):
            in_r = np.all(np.less_equal(x, self._right[i, :]), axis=1)
            in_l = np.all(np.greater(x, self._left[i, :]), axis=1)
            in_rec = np.logical_and(in_r, in_l)
            for j in range(k):
                if j == 0:
                    in_rec_now = np.logical_and(np.equal(pt[:, j], num - 1),
                                                in_rec)
                else:
                    in_rec_now = np.logical_and(np.logical_and(
                        np.equal(pt[:, j], num - 1), in_rec),
                        np.not_equal(pt[:, j - 1], i))
                pt[:, j][in_rec_now] = i
                dist[:, j][in_rec_now] = 0.0
        if k == 1:
            dist = dist[:, 0]
            pt = pt[:, 0]

        return (dist, pt)

    def exact_volume_lebesgue(self):
        r"""

        Exactly calculates the Lebesgue volume fraction of the cells.

        """
        num = self.check_num()
        self._volumes = np.zeros((num, ))
        domain_width = self._domain[:, 1] - self._domain[:, 0]
        self._volumes[0:-1] = np.prod(self._width[0:-1] / domain_width, axis=1)
        self._volumes[-1] = 1.0 - np.sum(self._volumes[0:-1])


class ball_sample_set(sample_set_base):
    r"""
    A data structure containing arrays specific to a set of samples defining
    discretization containing a number of balls.
    Only returns the neighbors for which :math:`x_i \in A_k`.

    A series of n balls :math:`A_i \subset \Lambda` with
    :math:`A_i \cap A_j = \emptyset`
    for :math:`i \neq j`. The last entry represents the remainder
    :math:`\Lambda \setminus ( \cup_{i-1}^n A_i)`.

    """

    def setup(self, centers, radii):
        """
        Initialize.

        :param centers: centers of balls
        :type centers: iterable of shape (num-1, dim)
        :param radii: radii of balls
        :type radii: iterable of length num-1

        """
        if len(centers) != len(radii):
            raise length_not_matching("Different number of centers and radii.")
        for i in range(len(centers)):
            if len(centers[i]) != self._dim:
                msg = "Center " + repr(i) + " has the wrong number of entries."
                raise length_not_matching(msg)
        values = np.zeros((len(centers) + 1, self._dim))
        values[0:-1, :] = centers
        values[-1, :] = np.nan
        self.set_values(values)
        self._radii = np.zeros((len(centers) + 1,))
        self._radii[0:-1] = radii
        self._radii[-1] = np.inf
        if len(centers) > 1:
            msg = "If balls intersect on a set nonzero measure, "
            msg += "calculated values will be wrong."
            logging.warning(msg)
        self._region = np.arange(len(centers) + 1)

    def append_values(self, values):
        """
        Does nothing for this type of sample_set.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values: values to append
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        msg = "Values cannot be appended for this type of sample set."
        logging.warning(msg)

    def append_values_local(self, values_local):
        """
        Does nothing for this type of sample_set.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values_local: values to append
        :type values_local: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        msg = "Values cannot be appended for this type of sample set."
        logging.warning(msg)

    def append_jacobians(self, new_jacobians):
        """
        Does nothing for this type of sample set.

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_jacobians: New jacobians to append.
        :type new_jacobians: :class:`numpy.ndarray` of shape (num, other_dim,
            dim)

        """
        msg = "Values cannot be appended for this type of sample set."
        logging.warning(msg)

    def append_error_estimates(self, new_error_estimates):
        """
        Does nothing for this type of sample set.

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_error_estimates: New error_estimates to append.
        :type new_error_estimates: :class:`numpy.ndarray` of shape (num,)

        """
        msg = "Values cannot be appended for this type of sample set."
        logging.warning(msg)

    def update_bounds(self, num=None):
        """
        Does nothing for this type of sample set.

        """
        logging.warning(
            "Bounds cannot be updated for this type of sample set.")

    def update_bounds_local(self, num_local=None):
        """
        Does nothing for this type of sample set.

        """
        logging.warning(
            "Bounds cannot be updated for this type of sample set.")

    def query(self, x, k=1):
        """
        Identify which value points x are associated with for discretization.
        The distance is set to 0 if it is in the rectangle and infinity
        if it is not.
        It is only considered in or out.

        .. seealso::

            :meth:`scipy.spatial.KDTree.query`

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param int k: number of nearest neighbors to return
        :rtype: tuple
        :returns: (dist, ptr)
        """
        num = self.check_num()
        dist = np.inf * np.ones((x.shape[0], k), dtype=np.float)
        pt = (num - 1) * np.ones((x.shape[0], k), dtype=np.int)
        for i in range(num - 1):
            in_rec = np.less(linalg.norm(x - self._values[i, :], self._p_norm,
                                         axis=1), self._radii[i])
            for j in range(k):
                if j == 0:
                    in_rec_now = np.logical_and(np.equal(pt[:, j], num - 1),
                                                in_rec)
                else:
                    in_rec_now = np.logical_and(np.logical_and(
                        np.equal(pt[:, j], num - 1), in_rec),
                        np.not_equal(pt[:, j - 1], i))
                pt[:, j][in_rec_now] = i
                dist[:, j][in_rec_now] = 0.0
        if k == 1:
            dist = dist[:, 0]
            pt = pt[:, 0]

        return (dist, pt)

    def exact_volume(self):
        """
        Calculate the exact volume fraction given the given p-norm.


        """
        num = self.check_num()
        self._volumes = np.zeros((num, ))
        domain_vol = np.product(self._domain[:, 1] - self._domain[:, 0])
        self._volumes[0:-1] = 2.0**self._dim * self._radii[0:-1]**self._dim * \
            scipy.special.gamma(1 + 1. / self._p_norm)**self._dim / \
            scipy.special.gamma(1 + float(self._dim) / self._p_norm)
        self._volumes[0:-1] *= 1.0 / domain_vol
        self._volumes[-1] = 1.0 - np.sum(self._volumes[0:-1])


class cartesian_sample_set(rectangle_sample_set):
    """
    Defines a hyperrectangle discretization based on a Cartesian grid.

        .. seealso::

            :meth:`bet.sample.rectangle_sample_set`

    """

    def setup(self, xi):
        """
        Initialize.

        :param xi: x1, x2,..., xn, 1-D arrays representing the coordinates of a
            grid
        :type xi: array_like

        .. seealso::

            :meth:`numpy.meshgrid`


        """
        if len(xi) != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        xmin = []
        xmax = []
        for xv in xi:
            xmin.append(xv[0:-1])
            xmax.append(xv[1::])
        if len(xmax) == 1:
            maxes = np.transpose(np.array([xmax]))
            mins = np.transpose(np.array([xmin]))
        else:
            maxes = np.vstack(np.array(np.meshgrid(*xmax)).T)
            mins = np.vstack(np.array(np.meshgrid(*xmin)).T)
        shp = np.array(maxes.shape)
        pd = np.product(shp[0:-1])
        maxes = maxes.reshape((pd, shp[-1]))
        mins = mins.reshape((pd, shp[-1]))

        rectangle_sample_set.setup(self, maxes, mins)


class discretization(object):
    """
    A data structure to store all of the :class:`~bet.sample.sample_set_base`
    objects and associated pointers to solve an stochastic inverse problem.
    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray`
    vector_names = ['_io_ptr', '_io_ptr_local', '_emulated_ii_ptr',
                    '_emulated_ii_ptr_local', '_emulated_oo_ptr', '_emulated_oo_ptr_local']
    #: List of attribute names for attributes that are
    #: :class:`sample.sample_set_base`
    sample_set_names = ['_input_sample_set', '_output_sample_set',
                        '_emulated_input_sample_set', '_emulated_output_sample_set',
                        '_output_probability_set', '_output_observed_set']

    def __init__(self, input_sample_set, output_sample_set,
                 output_probability_set=None,
                 emulated_input_sample_set=None,
                 emulated_output_sample_set=None,
                 output_observed_set=None):
        """
        Initialize the discretization.
        
        :param input_sample_set: Input sample set
        :type input_sample_set: :class:`bet.sample.sample_set_base`
        :param output_sample_set: Output sample set
        :type output_sample_set: :class:`bet.sample.sample_set_base`
        :param output_probability_set: Output probability set
        :type output_probability_set: :class:`bet.sample.sample_set_base`
        :param emulated_input_sample_set: Emulated input set
        :type emulated_input_sample_set: :class:`bet.sample.sample_set_base`
        :param emulated_output_sample_set: Emulated output set
        :type emulated_output_sample_set: :class:`bet.sample.sample_set_base`
        :param output_observed_set: Observed output set
        :type output_observed_set: :class:`bet.sample.sample_set_base`

        """
        #: Input sample set :class:`~bet.sample.sample_set_base`
        self._input_sample_set = input_sample_set
        #: Output sample set :class:`~bet.sample.sample_set_base`
        self._output_sample_set = output_sample_set
        #: Emulated Input sample set :class:`~bet.sample.sample_set_base`
        self._emulated_input_sample_set = emulated_input_sample_set
        #: Emulated output sample set :class:`~bet.sample.sample_set_base`
        self._emulated_output_sample_set = emulated_output_sample_set
        #: Output probability set :class:`~bet.sample.sample_set_base`
        self._output_probability_set = output_probability_set
        #: Pointer from ``self._output_sample_set`` to
        #: ``self._output_probability_set``
        #: Observed output sample set :class:`~bet.sample.sample_set_base`
        self._output_observed_set = output_observed_set
        self._io_ptr = None
        #: Pointer from ``self._emulated_input_sample_set`` to
        #: ``self._input_sample_set``
        self._emulated_ii_ptr = None
        #: Pointer from ``self._emulated_output_sample_set`` to
        #: ``self._output_probability_set``
        self._emulated_oo_ptr = None
        #: local io pointer for parallelism
        self._io_ptr_local = None
        #: local emulated ii ptr for parallelism
        self._emulated_ii_ptr_local = None
        #: local emulated oo ptr for parallelism
        self._emulated_oo_ptr_local = None

        if output_sample_set is not None:
            self.check_nums()
        else:
            logging.info("No output_sample_set")

    def __eq__(self, other):
        """
        Redefines equality to easily check the equivalence of two discretizations sets as having
        identical values in meta_fields for each sample set and vector.
        :param other: other object set to which compare
        :return: True for equality and False for not
        :rtype: bool
        """
        if self.__class__ == other.__class__:
            fields = self.sample_set_names + self.vector_names
            for field in fields:
                if type(getattr(self, field)) is np.ndarray:
                    if np.any(getattr(self, field) != getattr(other, field)):
                        return False
                elif type(getattr(self, field)) is list:
                    compare = getattr(self, field) == getattr(other, field)
                    if compare is bool:
                        if compare is False:
                            return False
                    else:
                        if compare.any() is False:
                            return False
                else:
                    if getattr(self, field) != getattr(other, field):
                        return False
            return True
        else:
            raise TypeError('Comparing object is not of the same type.')

    def save(self, filename, globalize=True):
        """

        Save the discretization using pickle.

        :return:
        """
        util.save_object(save_set=self, file_name=filename, globalize=globalize)

    def check_nums(self):
        """

        Checks that ``self._input_sample_set`` and ``self._output_sample_set``
        both have the same number of samples.

        :rtype: int
        :returns: Number of samples

        """
        out_num = self._output_sample_set.check_num()
        in_num = self._input_sample_set.check_num()
        if out_num != in_num and self._output_sample_set._values is not None \
                and self._input_sample_set._values is not None:
            raise length_not_matching("input {} and output {} lengths do not\
                    match".format(in_num, out_num))
        else:
            return in_num

    def globalize_ptrs(self):
        """
        Globalizes discretization pointers.

        """
        if (self._io_ptr_local is not None) and (self._io_ptr is None):
            self._io_ptr = util.get_global_values(self._io_ptr_local)
        if (self._emulated_ii_ptr_local is not None) and\
                (self._emulated_ii_ptr is None):
            self._emulated_ii_ptr = util.get_global_values(
                self._emulated_ii_ptr_local)
        if (self._emulated_oo_ptr_local is not None) and\
                (self._emulated_oo_ptr is None):
            self._emulated_oo_ptr = util.get_global_values(
                self._emulated_oo_ptr_local)

    def set_io_ptr(self, globalize=True):
        """

        Creates the pointer from ``self._output_sample_set`` to
        ``self._output_probability_set``

        :param bool globalize: flag whether or not to globalize
            ``self._output_sample_set``

        """
        if self._output_sample_set._values_local is None:
            self._output_sample_set.global_to_local()
        (_, self._io_ptr_local) = self._output_probability_set.query(
            self._output_sample_set._values_local)

        if globalize:
            self._io_ptr = util.get_global_values(self._io_ptr_local)

    def get_io_ptr(self):
        """

        Returns the pointer from ``self._output_sample_set`` to
        ``self._output_probability_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._output_sample_set._values.shape[0],)
        :returns: self._io_ptr

        """
        return self._io_ptr

    def set_emulated_ii_ptr(self, globalize=True):
        """

        Creates the pointer from ``self._emulated_input_sample_set`` to
        ``self._input_sample_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._output_sample_set``
        :param int p: Which Minkowski p-norm to use. (1 <= p <= infinity)

        """
        if self._emulated_input_sample_set._values_local is None:
            self._emulated_input_sample_set.global_to_local()
        (_, self._emulated_ii_ptr_local) = self._input_sample_set.query(
            self._emulated_input_sample_set._values_local)
        if globalize:
            self._emulated_ii_ptr = util.get_global_values(
                self._emulated_ii_ptr_local)

    def get_emulated_ii_ptr(self):
        """

        Returns the pointer from ``self._emulated_input_sample_set`` to
        ``self._input_sample_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._output_sample_set._values.shape[0],)
        :returns: self._emulated_ii_ptr

        """
        return self._emulated_ii_ptr

    def set_emulated_oo_ptr(self, globalize=True):
        """

        Creates the pointer from ``self._emulated_output_sample_set`` to
        ``self._output_probability_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._output_sample_set``
        :param int p: Which Minkowski p-norm to use. (1 <= p <= infinity)

        """
        if self._emulated_output_sample_set._values_local is None:
            self._emulated_output_sample_set.global_to_local()
        (_, self._emulated_oo_ptr_local) = self._output_probability_set.query(
            self._emulated_output_sample_set._values_local)

        if globalize:
            self._emulated_oo_ptr = util.get_global_values(
                self._emulated_oo_ptr_local)

    def get_emulated_oo_ptr(self):
        """

        Returns the pointer from ``self._emulated_output_sample_set`` to
        ``self._output_probability_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._output_sample_set._values.shape[0],)
        :returns: self._emulated_ii_ptr

        """
        return self._emulated_oo_ptr

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.sample.discretization`
        :returns: Copy of this :class:`~bet.sample.discretization`

        """
        return copy.deepcopy(self)

    def get_input_sample_set(self):
        """

        Returns a reference to the input sample set for this discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: input sample set

        """
        return self._input_sample_set

    def set_input_sample_set(self, input_sample_set):
        """

        Sets the input sample set for this discretization.

        :param input_sample_set: input sample set.
        :type input_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(input_sample_set, sample_set_base):
            self._input_sample_set = input_sample_set
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def get_output_sample_set(self):
        """

        Returns a reference to the output sample set for this discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: output sample set

        """
        return self._output_sample_set

    def set_output_sample_set(self, output_sample_set):
        """

        Sets the output sample set for this discretization.

        :param output_sample_set: output sample set.
        :type output_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(output_sample_set, sample_set_base):
            self._output_sample_set = output_sample_set
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def get_output_observed_set(self):
        """

        Returns a reference to the output observed sample set for this discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: output sample set

        """
        return self._output_observed_set

    def set_output_observed_set(self, output_sample_set):
        """

        Sets the output observed sample set for this discretization.

        :param output_sample_set: output observed sample set.
        :type output_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(output_sample_set, sample_set_base):
            self._output_observed_set = output_sample_set
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def get_output_probability_set(self):
        """

        Returns a reference to the output probability sample set for this
        discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: output probability sample set

        """
        return self._output_probability_set

    def set_output_probability_set(self, output_probability_set):
        """

        Sets the output probability sample set for this discretization.

        :param output_probability_set: output probability sample set.
        :type output_probability_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(output_probability_set, sample_set_base):
            output_dims = []
            output_dims.append(output_probability_set.get_dim())
            if self._output_sample_set is not None:
                output_dims.append(self._output_sample_set.get_dim())
            if self._emulated_output_sample_set is not None:
                output_dims.append(self._emulated_output_sample_set.get_dim())
            if len(output_dims) == 1:
                self._output_probability_set = output_probability_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._output_probability_set = output_probability_set
            else:
                raise dim_not_matching("dimension of values incorrect")
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")
        if self._output_sample_set._values_local is not None:
            if output_probability_set._values is not None:
                self.set_io_ptr(globalize=False)

    def get_emulated_output_sample_set(self):
        """

        Returns a reference to the emulated_output sample set for this
        discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: emulated_output sample set

        """
        return self._emulated_output_sample_set

    def set_emulated_output_sample_set(self, emulated_output_sample_set):
        """

        Sets the emulated_output sample set for this discretization.

        :param emulated_output_sample_set: emupated output sample set.
        :type emulated_output_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(emulated_output_sample_set, sample_set_base):
            output_dims = []
            output_dims.append(emulated_output_sample_set.get_dim())
            if self._output_sample_set is not None:
                output_dims.append(self._output_sample_set.get_dim())
            if self._output_probability_set is not None:
                output_dims.append(self._output_probability_set.get_dim())
            if len(output_dims) == 1:
                self._emulated_output_sample_set = emulated_output_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._emulated_output_sample_set = emulated_output_sample_set
            else:
                raise dim_not_matching("dimension of values incorrect")
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def get_emulated_input_sample_set(self):
        """

        Returns a reference to the emulated_input sample set for this
        discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: emulated_input sample set

        """
        return self._emulated_input_sample_set

    def set_emulated_input_sample_set(self, emulated_input_sample_set):
        """

        Sets the emulated_input sample set for this discretization.

        :param emulated_input_sample_set: emupated input sample set.
        :type emulated_input_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(emulated_input_sample_set, sample_set_base):
            if self._input_sample_set is not None:
                if self._input_sample_set.get_dim() == \
                        emulated_input_sample_set.get_dim():
                    self._emulated_input_sample_set = emulated_input_sample_set
                else:
                    raise dim_not_matching("dimension of values incorrect")
            else:
                self._emulated_input_sample_set = emulated_input_sample_set
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def estimate_input_volume_emulated(self):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration.

        .. note ::

            This could be re-written to just use ``emulated_ii_ptr`` instead
            of ``_emulated_input_sample_set``.

        """
        if self._emulated_input_sample_set is None:
            raise AttributeError("Required: _emulated_input_sample_set")
        else:
            self._input_sample_set.estimate_volume_emulated(
                self._emulated_input_sample_set)

    def estimate_output_volume_emulated(self):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration.

        .. note ::

            This could be re-written to just use ``emulated_oo_ptr`` instead
            of ``_emulated_output_sample_set``.


        """
        if self._emulated_output_sample_set is None:
            raise AttributeError("Required: _emulated_output_sample_set")
        else:
            self._output_sample_set.estimate_volume_emulated(
                self._emulated_output_sample_set)

    def clip(self, cnum):
        """
        Creates and returns a discretization with the the first `cnum`
        entries of the input and output sample sets.

        :param int cnum: number of values of sample set to return

        :rtype: :class:`~bet.sample.discretization`
        :returns: clipped discretization

        """
        ci = self._input_sample_set.clip(cnum)
        co = self._output_sample_set.clip(cnum)

        return discretization(input_sample_set=ci,
                              output_sample_set=co,
                              output_probability_set=self._output_probability_set,
                              emulated_input_sample_set=self._emulated_input_sample_set,
                              emulated_output_sample_set=self._emulated_output_sample_set)

    def merge(self, disc):
        """
        Merges a given discretization with this one by merging the input and
        output sample sets.

        :param disc: Discretization object to merge with.
        :type disc: :class:`bet.sample.discretization`

        :rtype: :class:`bet.sample.discretization`
        :returns: Merged discretization
        """
        mi = self._input_sample_set.merge(disc._input_sample_set)
        mo = self._output_sample_set.merge(disc._output_sample_set)
        mei = self._emulated_input_sample_set.merge(disc.
                                                    _emulated_input_sample_set)
        meo = self._emulated_output_sample_set.merge(disc.
                                                     _emulated_output_sample_set)

        return discretization(input_sample_set=mi,
                              output_sample_set=mo,
                              output_probability_set=self._output_probability_set,
                              emulated_input_sample_set=mei,
                              emulated_output_sample_set=meo)

    def choose_inputs_outputs(self,
                              inputs=None,
                              outputs=None):
        """
        Slices the inputs and outputs of the discretization.

        :param list inputs: list of indices of input sample set to include
        :param list outputs: list of indices of output sample set to include

        :rtype: :class:`~bet.sample.discretization`
        :returns: sliced discretization

        """
        slice_list = ['_values', '_values_local',
                      '_error_estimates', '_error_estimates_local']
        slice_list2 = ['_jacobians', '_jacobians_local']

        input_ss = sample_set(len(inputs))
        output_ss = sample_set(len(outputs))
        input_ss.set_p_norm(self._input_sample_set._p_norm)
        if self._input_sample_set._domain is not None:
            input_ss.set_domain(self._input_sample_set._domain[inputs, :])
        if self._input_sample_set._reference_value is not None:
            input_ss.set_reference_value(
                self._input_sample_set._reference_value[inputs])

        output_ss.set_p_norm(self._output_sample_set._p_norm)
        if self._output_sample_set._domain is not None:
            output_ss.set_domain(self._output_sample_set._domain[outputs, :])
        if self._output_sample_set._reference_value is not None:
            output_ss.set_reference_value(
                self._output_sample_set._reference_value[outputs])

        for obj in slice_list:
            val = getattr(self._input_sample_set, obj)
            if val is not None:
                setattr(input_ss, obj, val[:, inputs])
            val = getattr(self._output_sample_set, obj)
            if val is not None:
                setattr(output_ss, obj, val[:, outputs])
        for obj in slice_list2:
            val = getattr(self._input_sample_set, obj)
            if val is not None:
                nval = np.copy(val)
                nval = nval.take(outputs, axis=1)
                nval = nval.take(inputs, axis=2)
                setattr(input_ss, obj, nval)
        disc = discretization(input_sample_set=input_ss,
                              output_sample_set=output_ss)
        return disc

    def local_to_global(self):
        """
        Call local_to_global for ``input_sample_set`` and
        ``output_sample_set``.
        """
        if self._input_sample_set is not None:
            self._input_sample_set.local_to_global()
        if self._output_sample_set is not None:
            self._output_sample_set.local_to_global()
        if self._output_probability_set is not None:
            self._output_probability_set.local_to_global()
        if self._emulated_input_sample_set is not None:
            self._emulated_input_sample_set.local_to_global()
        if self._emulated_output_sample_set is not None:
            self._emulated_output_sample_set.local_to_global()

    def global_to_local(self):
        """
        Call global_to_local for ``input_sample_set`` and
        ``output_sample_set``.
        """
        if self._input_sample_set is not None:
            self._input_sample_set.global_to_local()
        if self._output_sample_set is not None:
            self._output_sample_set.global_to_local()
        if self._output_probability_set is not None:
            self._output_probability_set.global_to_local()
        if self._emulated_input_sample_set is not None:
            self._emulated_input_sample_set.global_to_local()
        if self._emulated_output_sample_set is not None:
            self._emulated_output_sample_set.global_to_local
