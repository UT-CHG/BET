# Copyright (C) 2014-2019 The BET Development Team

"""
This module contains data structure/storage classes for BET. Notably:
    :class:`bet.sample.sample_set`
    :class:`bet.sample.discretization`
    :class:`bet.sample.length_not_matching`
    :class:`bet.sample.dim_not_matching`
"""

import os
import logging
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
import pickle

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


def loadmat(filename):
    """
    Replacement for sio.loadmat
    """
    try:
        with open(filename, 'rb') as input_file:
            mdat = pickle.load(input_file)  # , encoding='utf-8')
    except EOFError:
        mdat = {}
    return mdat


def savemat(filename, data):
    """
    Replacement for sio.loadmat
    """
    with open(filename, 'wb') as output_file:
        pickle.dump(data, output_file)


def save_sample_set(save_set, file_name,
                    sample_set_name=None, globalize=False):
    """
    Saves this :class:`bet.sample.sample_set` as a ``.mat`` file. Each
    attribute is added to a dictionary of names and arrays which are then
    saved to a MATLAB-style file.

    :param save_set: sample set to save
    :type save_set: :class:`bet.sample.sample_set_base`
    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string sample_set_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.sample_set_base` objects to a single
        ``.mat`` file
    :param bool globalize: flag whether or not to globalize

    :rtype: string
    :returns: local file name

    """
    # create processor specific file name
    if comm.size > 1 and not globalize:
        local_file_name = os.path.join(os.path.dirname(file_name),
                                       "proc{}_{}".format(comm.rank,
                                                          os.path.
                                                          basename(file_name)))
    else:
        local_file_name = file_name

    # globalize
    if globalize and save_set._values_local is not None:
        save_set.local_to_global()
    comm.barrier()

    new_mdat = dict()
    # create temporary dictionary
    if os.path.exists(local_file_name) or \
            os.path.exists(local_file_name + '.mat'):
        new_mdat = loadmat(local_file_name)

    # store sample set in dictionary
    if sample_set_name is None:
        sample_set_name = 'default'
    for attrname in save_set.vector_names:
        curr_attr = getattr(save_set, attrname)
        if curr_attr is not None:
            new_mdat[sample_set_name + attrname] = curr_attr
        elif sample_set_name + attrname in new_mdat:
            new_mdat.pop(sample_set_name + attrname)
    for attrname in save_set.all_ndarray_names:
        curr_attr = getattr(save_set, attrname)
        if curr_attr is not None:
            new_mdat[sample_set_name + attrname] = curr_attr
        elif sample_set_name + attrname in new_mdat:
            new_mdat.pop(sample_set_name + attrname)
    new_mdat[sample_set_name + '_sample_set_type'] = \
        str(type(save_set)).split("'")[1]
    dist = save_set.get_dist()
    if dist is not None:
        kwds = dist.kwds
        for kwd in kwds.keys():
            new_mdat[sample_set_name + '_dist_kwds_' + kwd] = \
                kwds[kwd]
            print('saved %s' % kwd)
        new_mdat[sample_set_name + '_dist_type'] = \
            str(save_set.get_dist().dist).split(' ')[0].replace(
                '<', '').replace('_gen', '').replace('_continuous_distns.', '')
    comm.barrier()

    # save new file or append to existing file
    if (globalize and comm.rank == 0) or not globalize:
        savemat(local_file_name, new_mdat)
    comm.barrier()
    return local_file_name


def load_sample_set(file_name, sample_set_name=None, localize=True):
    """
    Loads a :class:`~bet.sample.sample_set` from a ``.mat`` file. If a file
    contains multiple :class:`~bet.sample.sample_set` objects then
    ``sample_set_name`` is used to distinguish which between different
    :class:`~bet.sample.sample_set` objects.

    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string sample_set_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.sample_set` objects to a single
        ``.mat`` file
    :param bool localize: Flag whether or not to re-localize arrays. If
        ``file_name`` is prepended by ``proc_{}`` localize is set to ``False``.

    :rtype: :class:`~bet.sample.sample_set`
    :returns: the ``sample_set`` that matches the ``sample_set_name``

    """
    # check to see if parallel file name
    if file_name.startswith('proc_'):
        localize = False
    elif not os.path.exists(file_name) and os.path.exists(os.path.join(
            os.path.dirname(file_name), "proc0_{}".format(
                os.path.basename(file_name)))):
        return load_sample_set_parallel(file_name, sample_set_name)

    mdat = loadmat(file_name)
    if sample_set_name is None:
        sample_set_name = 'default'
    mdat_keys = list(mdat.keys())
    if sample_set_name + "_dim" in mdat_keys:
        loaded_set = eval(mdat[sample_set_name + '_sample_set_type'])(
            np.int(np.squeeze(mdat[sample_set_name + "_dim"])))
        if sample_set_name + '_dist_type' in mdat_keys:
            kwds = {}
            # extract keywords from mdat
            kwd_keys = [k for k in mdat_keys if 'kwds' in k]
            # build dictionary
            for key in kwd_keys:
                newkey = key.replace('dist_kwds', '').replace(
                    sample_set_name, '').replace('_', '')
                kwds[newkey] = mdat[key]
            dist = eval(mdat[sample_set_name + '_dist_type'])(**kwds)
            loaded_set.set_distribution(dist)

    else:
        logging.info("No sample_set named {} with _dim in file".
                     format(sample_set_name))
        return None

    for attrname in loaded_set.vector_names:
        if attrname is not '_dim':
            if sample_set_name + attrname in mdat_keys:
                setattr(loaded_set, attrname,
                        np.squeeze(mdat[sample_set_name + attrname]))
    for attrname in loaded_set.all_ndarray_names:
        if sample_set_name + attrname in mdat_keys:
            setattr(loaded_set, attrname, mdat[sample_set_name + attrname])

    if localize:
        # re-localize if necessary
        loaded_set.global_to_local()

    return loaded_set


def load_sample_set_parallel(file_name, sample_set_name=None):
    """
    Loads a :class:`~bet.sample.sample_set` from a ``.mat`` file in parallel
    and correctly re-localizes data if necessary. If a file contains multiple
    :class:`~bet.sample.sample_set` objects then ``sample_set_name`` is used to
    distinguish which between different :class:`~bet.sample.sample_set`
    objects.

    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string sample_set_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.sample_set` objects to a single
        ``.mat`` file

    :rtype: :class:`~bet.sample.sample_set`
    :returns: the ``sample_set`` that matches the ``sample_set_name``
    """

    if sample_set_name is None:
        sample_set_name = 'default'
    # Find and open save files
    save_dir = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    mdat_files = glob.glob(os.path.join(save_dir,
                                        "proc*_{}".format(base_name)))

    if len(mdat_files) == comm.size:
        logging.info("Loading {} sample set using parallel files (same nproc)"
                     .format(sample_set_name))
        # if the number of processors is the same then set mdat to
        # be the one with the matching processor number (doesn't
        # really matter)
        local_file_name = os.path.join(os.path.dirname(file_name),
                                       "proc{}_{}".format(comm.rank,
                                                          os.path.basename(file_name)))
        return load_sample_set(local_file_name, sample_set_name)
    else:
        logging.info("Loading {} sample set using parallel files (diff nproc)"
                     .format(sample_set_name))
        # Determine how many processors the previous data used
        # otherwise gather the data from mdat and then scatter
        # among the processors and update mdat
        try:
            mdat_files_local = comm.scatter(mdat_files)
            mdat_local = [loadmat(m) for m in mdat_files_local]
            mdat_list = comm.allgather(mdat_local)
        except ValueError:
            mdat_files_local = mdat_files
            mdat_local = [loadmat(m) for m in mdat_files_local]
            mdat_list = mdat_local
        mdat_global = []
        # instead of a list of lists, create a list of mdat
        for mlist in mdat_list:
            mdat_global.append(mlist)

        # set attributes that are not vectors (distributions)
        mdat_keys = list(mdat_global[0].keys())
        if sample_set_name + "_dim" in mdat_keys:
            loaded_set = eval(mdat_global[0][sample_set_name +
                                             '_sample_set_type'])(
                np.int(np.squeeze(mdat_global[0][sample_set_name + "_dim"])))
        else:
            logging.info("No sample_set named {} with _dim in file".
                         format(sample_set_name))
            return None

        if sample_set_name + '_dist_type' in mdat_keys:
            kwds = {}
            # extract keywords from mdat
            kwd_keys = [k for k in mdat_keys if 'kwds' in k]
            # build dictionary
            for key in kwd_keys:
                newkey = key.replace('dist_kwds', '').replace(
                    sample_set_name, '').replace('_', '')
                kwds[newkey] = mdat_global[0][key]
            dist = eval(
                mdat_global[0][sample_set_name + '_dist_type'])(**kwds)
            loaded_set.set_distribution(dist)

        # load attributes
        for attrname in loaded_set.vector_names:
            if attrname is not '_dim':
                if sample_set_name + attrname in mdat_keys:
                    # create lists of local data
                    if attrname.endswith('_local'):
                        temp_input = []
                        for mdat in mdat_global:
                            temp_input.append(np.squeeze(
                                mdat[sample_set_name + attrname]))
                        # turn into arrays
                        temp_input = np.concatenate(temp_input)
                    else:
                        temp_input = np.squeeze(mdat_global[0]
                                                [sample_set_name + attrname])
                    setattr(loaded_set, attrname, temp_input)
        for attrname in loaded_set.all_ndarray_names:
            if sample_set_name + attrname in mdat_keys:
                if attrname.endswith('_local'):
                    # create lists of local data
                    temp_input = []
                    for mdat in mdat_global:
                        temp_input.append(mdat[sample_set_name + attrname])
                    # turn into arrays
                    temp_input = np.concatenate(temp_input)
                else:
                    temp_input = mdat_global[0][sample_set_name + attrname]
                setattr(loaded_set, attrname, temp_input)

        # re-localize if necessary
        loaded_set.local_to_global()
        return loaded_set


class sample_set_base(object):
    """

    A data structure containing arrays specific to a set of samples.

    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray` or int/float
    vector_names = ['_probabilities', '_probabilities_local',
                    '_volumes', '_volumes_local',
                    '_densities', '_densities_local',
                    '_local_index', '_dim', '_p_norm',
                    '_radii', '_normalized_radii', '_region', '_region_local',
                    '_error_id', '_error_id_local', '_reference_value',
                    '_domain_original']

    #: List of global attribute names for attributes that are
    #: :class:`numpy.ndarray`
    array_names = ['_values', '_volumes', '_probabilities',
                   '_densities', '_jacobians',
                   '_error_estimates', '_right', '_left', '_width',
                   '_kdtree_values', '_radii', '_normalized_radii',
                   '_region', '_error_id']

    #: List of attribute names for attributes that are
    #: :class:`numpy.ndarray` with dim > 1
    all_ndarray_names = ['_error_estimates', '_error_estimates_local',
                         '_values', '_values_local', '_left', '_left_local',
                         '_right', '_right_local', '_width', '_width_local',
                         '_domain', '_kdtree_values', '_jacobians',
                         '_jacobians_local', '_domain_original']

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
        #: :class:`scipy.stats.distributions.rv_frozen` describing distribution
        self._distribution = None
        #: :class:`numpy.ndarray` of sample densities of shape (num,)
        self._densities = None
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
        #: The sample domain, :class:`numpy.ndarray` of shape (dim, 2)
        self._domain = None
        #: The sample domain pre-normalization,
        #: :class:`numpy.ndarray` of shape (dim, 2)
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
        vl = util.fix_dimensions_data(values, self._dim)
        self._values = np.concatenate((self._values, vl), 0)

    def append_values_local(self, values_local):
        """
        Appends the values in ``_values_local`` to ``self._values``.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values_local: values to append
        :type values_local: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        vl = util.fix_dimensions_data(values_local, self._dim)
        self._values_local = np.concatenate((self._values_local, vl), 0)

    def clip(self, cnum):
        """
        Creates and returns a sample set with the the first `cnum`
        entries of the sample set if `cnum` is an integer.
        If `cum` is a list, it is used to return a selection of rows.

        :param cnum: number/selection of values of sample set to return
        :type cnum: `int` or iterable indices into rows of sample set.

        :rtype: :class:`~bet.sample.sample_set`
        :returns: the clipped sample set

        """
        sset = self.copy()
        sset.check_num()
        if isinstance(cnum, int):
            indices = list(np.arange(cnum))
        elif isinstance(cnum, list) or isinstance(cnum, tuple):
            indices = cnum

        if sset._values is None:
            sset.local_to_global()
        for array_name in self.array_names:
            current_array = getattr(sset, array_name)
            if current_array is not None:
                new_array = current_array[indices]
                setattr(sset, array_name, new_array)
        if sset._values_local is not None:
            sset.global_to_local()
        sset.set_kdtree()
        return sset

    def clip_samples(self, inds):
        """
        Creates and returns a sample set with the the first `cnum`
        entries of the sample set.

        :param tuple inds: indices in sample set to return

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
                new_array = current_array[inds]
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
        self._values_local = None

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

    def set_volumes(self, volumes=None):
        """
        Sets sample cell volumes.

        :type volumes: :class:`numpy.ndarray` of shape (num,)
        :param volumes: sample cell volumes

        """
        if volumes is not None:
            self._volumes = volumes
        else:
            logging.warning("Setting volumes with prob/density.")
            dens = self._densities
            probs = self._probabilities
            if (dens is None) or (probs is None):
                self._volumes = None
            else:
                self._volumes = probs / dens

    def get_volumes(self):
        """
        Returns sample cell volumes.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample cell volumes

        """
        return self._volumes

    def set_probabilities(self, probabilities=None):
        """
        Set sample probabilities.

        :type probabilities: :class:`numpy.ndarray` of shape (num,)
        :param probabilities: sample probabilities

        """
        if probabilities is not None:
            self._probabilities = probabilities
        else:
            logging.warning("Setting probabilities with density*volume.")
            dens = self._densities
            vols = self._volumes
            if (dens is None) or (vols is None):
                self._probabilities = None
            else:
                self._probabilities = dens * vols

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
            probs = self._probabilities
            vols = self._volumes
            self._densities = probs / vols

    def get_densities(self):
        """
        Returns sample densities.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample densities

        """
        return self._densities

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

    def set_probabilities_local(self, probabilities_local=None):
        """
        Set sample local probabilities.

        :type probabilities_local: :class:`numpy.ndarray` of shape (num,)
        :param probabilities_local: local sample probabilities

        """
        if probabilities_local is not None:
            self._probabilities_local = probabilities_local
        else:
            logging.warning("Setting probabilities with density*volume.")
            dens = self._densities_local
            vols = self._volumes_local
            if (dens is None) or (vols is None):
                self._probabilities_local = None
            else:
                self._probabilities_local = dens * vols

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
                                              self._domain.shape[0])) +\
            self._domain[:, 0]
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
        my_copy = type(self)(self.get_dim())
        for array_name in self.all_ndarray_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(my_copy, array_name,
                        np.copy(current_array))
        for vector_name in self.vector_names:
            if vector_name is not "_dim":
                current_vector = getattr(self, vector_name)
                if current_vector is not None:
                    setattr(my_copy, vector_name, np.copy(current_vector))
        if self._kdtree is not None:
            my_copy.set_kdtree()
        return my_copy

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

    def calculate_volumes(self):
        """

        Calculate the volumes of cells. Depends on sample set type.

        """
        pass

    def set_distribution(self, dist=None, *args, **kwds):
        r"""
        Assign an description of uncertainty for sample set.
        The type is flexible, but needs to contain the following
        methods in order to function:
        - ``.pdf`` - return density values as ``ndarray``
        - ``.rvs`` - generate random variables
        - ``.cdf`` - return cumulative distribution value
        - ``.interval`` - return confidence interval around median
        It is suggested to pass a ``scipy.stats.distributions`` object.
        If one is detected, we will automatically handle the pdf and cdf
        methods to return the product of the marginals as methods.

        Pass any additional keyword arguments to ``dist`` that are required.
        """
        if dist is None:
            from scipy.stats import norm
            dist = norm
            self._domain = np.array([[-np.inf, np.inf]] * self._dim)
        if isinstance(dist, scipy.stats.distributions.rv_frozen):
            self._distribution = dist
        elif isinstance(dist, str):
            if dist in ['uni', 'uniform']:
                dist = scipy.stats.uniform
            elif dist in ['norm', 'normal', 'gauss', 'gaussian']:
                dist = scipy.stats.norm
            elif dist in ['beta']:
                dist = scipy.stats.beta
            else:
                raise AttributeError("Inappropriate dist specified.")
            self._distribution = dist(*args, **kwds)
        else:
            self._distribution = dist(*args, **kwds)
        try:  # set domain based on distribution
            mins, maxs = self._distribution.interval(1)
            domain = np.zeros((self._dim, 2))
            domain[:, 0], domain[:, 1] = mins, maxs
            self._domain = domain
        except ValueError:
            raise dim_not_matching("Dimensions incorrectly specified.")
        except AttributeError:
            logging.warning("Could not infer domain from distribution.")

    def set_dist(self, dist=None, *args, **kwds):
        """
        Wrapper for ``set_distribution``
        """
        return self.set_distribution(dist, *args, **kwds)

    def get_dist(self):
        """
        Wrapper for ``get_distribution``
        """
        return self.get_distribution()

    def get_distribution(self):
        """
        Returns ``distribution``
        """
        return self._distribution

    def rvs(self, num=1, dist=None, *args, **kwds):
        """
        Returns correctly-shaped random variates.
        """
        if dist is None:
            dist = self._distribution
        # instantiate class if need be.
        if isinstance(dist, scipy.stats.distributions.rv_continuous):
            dist = dist(*args, **kwds)
        if isinstance(dist, scipy.stats.gaussian_kde):
            return dist.resample(num).T
        else:
            try:
                if self._dim == 1:
                    return dist.rvs(num)
                else:
                    return dist.rvs(size=(num, self._dim))
            except ValueError:
                return dist.rvs(size=(num, 1))

    def generate_samples(self, num_samples=None, globalize=False,
                         dist=None, *args, **kwds):
        """
        Generate i.i.d samples according to distribution
        """
        if num_samples is None:
            num_samples = self.check_num()
        # define local number of samples
        num_samples_local = int(num_samples / comm.size) +\
            int(comm.rank < num_samples % comm.size)
        self.set_values_local(self.rvs(num_samples_local,
                                       dist, *args, **kwds))

        if dist is not None:
            self.set_distribution(dist, *args, **kwds)
        # clear existing arrays that depend on samples
        self.update_bounds_local()
        self._probabilities_local = None
        self._volumes_local = None
        self._jacobians_local = None
        self._densities_local = None
        self._kdtree_values_local = None
        comm.barrier()

        if not globalize:
            self.local_to_global()
        else:
            self._values = None
        # clear existing arrays that depend on samples
        self._volumes = None
        self._jacobians = None
        self._probabilities = None
        self._densities = None
        self._kdtree_values = None
        self._kdtree = None

    def pdf(self, x=None, dist=None):
        r"""
        Evaluate the probability density at a set of points x

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param dist: distribution with `rvs`, `pdf`, and `cdf` methods
        :type dist: :class:`scipy.stats.distributions.rv_frozen`

        Note
        =====
        If `x` is None, we default to evaluating at `values`.
        If `dist` is None, we use `self._distribution`.
        You can specify an alternative distribution to take advantage
        of the re-formatting of outputs to satisfy our assumptions.

        """
        if dist is None:
            dist = self._distribution

        if x is None:
            if self._values_local is None:
                self.global_to_local()
            # if self._values is None:
            #     self.local_to_global()
            x_local = self._values_local
            num = self.check_num()
        else:  # proessors evaluate subset of points
            if x.ndim == 1:
                x = x.reshape(1, -1)
            num = x.shape[0]
            # num_per_proc = int(num / comm.size) + \
            #                int(comm.rank < num % comm.size)
            # x_local = np.empty((num_per_proc, x.shape[1]), dtype='d')
            # comm.Scatter(x, x_local, root=0)

            x_local = np.array_split(x, comm.size)[comm.rank]

        if isinstance(dist, scipy.stats.gaussian_kde):
            D_local = dist.pdf(x_local.T).T  # needs transpose
        else:
            if self._dim > 1:
                try:  # handle `scipy.stats.rv_frozen` objects
                    D_local = dist.pdf(x_local).prod(axis=1)
                except np.AxisError:
                    D_local = dist.pdf(x_local)
            else:  # 1-dimensional case
                D_local = dist.pdf(x_local)
        comm.barrier()
        densities = util.get_global_values(D_local)
        assert len(densities) == num  # make sure we return correct size
        return densities.ravel()  # always return flattened

    def cdf(self, x=None, dist=None):
        r"""
        Evaluate the cumulative density at a set of points x

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param dist: distribution with `rvs`, `pdf`, and `cdf` methods
        :type dist: :class:`scipy.stats.distributions.rv_frozen`

        Note
        =====
        If `x` is None, we default to evaluating at `values`.
        If `dist` is None, we use `self._distribution`.
        You can specify an alternative distribution to take advantage
        of the re-formatting of outputs to satisfy our assumptions.

        """
        if dist is None:
            dist = self._distribution
        if x is None:
            if self._values_local is None:
                self.global_to_local()
            # if self._values is None:
            #     self.local_to_global()
            x_local = self._values_local
            num = self.check_num()
        else:  # proessors evaluate subset of points
            if x.ndim == 1:
                x = x.reshape(1, -1)
            num = x.shape[0]
            # num_per_proc = int(num / comm.size) + \
            #                int(comm.rank < num % comm.size)
            # x_local = np.empty((num_per_proc, x.shape[1]), dtype='d')
            # comm.Scatter(x, x_local, root=0)

            x_local = np.array_split(x, comm.size)[comm.rank]

        if isinstance(dist, scipy.stats.gaussian_kde):
            C_local = dist.cdf(x_local.T).T  # needs transpose
        else:
            if self._dim > 1:
                try:  # handle `scipy.stats.rv_frozen` objects
                    C_local = dist.cdf(x_local).prod(axis=1)
                except np.AxisError:
                    C_local = dist.cdf(x_local)
            else:  # 1-dimensional case
                C_local = dist.cdf(x_local)
        cumulatives = util.get_global_values(C_local)
        assert len(cumulatives) == num
        return cumulatives.ravel()  # always return flattened

    def estimate_probabilities_mc(self, globalize=True):
        """
        Give all cells the same probability fraction
        based on the Monte Carlo assumption.
        """
        num = self.check_num()
        self._probabilities = 1.0 / float(num) * np.ones((num,))
        if globalize:
            self.global_to_local()


def save_discretization(save_disc, file_name, discretization_name=None,
                        globalize=False):
    """
    Saves this :class:`bet.sample.discretization` as a ``.mat`` file. Each
    attribute is added to a dictionary of names and arrays which are then
    saved to a MATLAB-style file.

    :param save_disc: sample set to save
    :type save_disc: :class:`bet.sample.discretization`
    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string discretization_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.discretization` objects to a single
        ``.mat`` file
    :param bool globalize: flag whether or not to globalize
        :class:`bet.sample.sample_set_base` objects stored in this
        discretization

    :rtype: string
    :returns: local file name

    """
    # create temporary dictionary
    new_mdat = dict()

    # create processor specific file name
    if comm.size > 1 and not globalize:
        local_file_name = os.path.join(os.path.dirname(file_name),
                                       "proc{}_{}".format(comm.rank,
                                                          os.path.
                                                          basename(file_name)))
    else:
        local_file_name = file_name

    # set name if doesn't exist
    if discretization_name is None:
        discretization_name = 'default'

    # globalize the pointers
    if globalize:
        save_disc.globalize_ptrs()
    # save sample sets if they exist
    for attrname in discretization.sample_set_names:
        curr_attr = getattr(save_disc, attrname)
        if curr_attr is not None:
            if attrname in discretization.sample_set_names:
                save_sample_set(curr_attr, file_name,
                                discretization_name + attrname, globalize)

    new_mdat = dict()
    # create temporary dictionary
    if os.path.exists(local_file_name) or \
            os.path.exists(local_file_name + '.mat'):
        new_mdat = loadmat(local_file_name)

    # store discretization in dictionary
    for attrname in discretization.vector_names:
        curr_attr = getattr(save_disc, attrname)
        if curr_attr is not None:
            new_mdat[discretization_name + attrname] = curr_attr
        elif discretization_name + attrname in new_mdat:
            new_mdat.pop(discretization_name + attrname)
    comm.barrier()

    # save new file or append to existing file
    if (globalize and comm.rank == 0) or not globalize:
        savemat(local_file_name, new_mdat)
    comm.barrier()
    return local_file_name


def load_discretization_parallel(file_name, discretization_name=None):
    """
    Loads a :class:`~bet.sample.discretization` from a ``.mat`` file. If a file
    contains multiple :class:`~bet.sample.discretization` objects then
    ``discretization_name`` is used to distinguish which between different
    :class:`~bet.sample.discretization` objects.

    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string discretization_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.discretization` objects to a single
        ``.mat`` file

    :rtype: :class:`~bet.sample.discretization`
    :returns: the ``discretization`` that matches the ``discretization_name``

    """
    # Find and open save files
    save_dir = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    mdat_files = glob.glob(os.path.join(save_dir,
                                        "proc*_{}".format(base_name)))

    if len(mdat_files) == comm.size:
        logging.info("Loading {} sample set using parallel files (same nproc)"
                     .format(discretization_name))
        # if the number of processors is the same then set mdat to
        # be the one with the matching processor number (doesn't
        # really matter)
        return load_discretization(mdat_files[comm.rank], discretization_name)
    else:
        logging.info("Loading {} sample set using parallel files (diff nproc)"
                     .format(discretization_name))

        if discretization_name is None:
            discretization_name = 'default'

        input_sample_set = load_sample_set(file_name, discretization_name +
                                           '_input_sample_set')

        output_sample_set = load_sample_set(file_name, discretization_name +
                                            '_output_sample_set')

        loaded_disc = discretization(input_sample_set, output_sample_set)

        # Determine how many processors the previous data used
        # otherwise gather the data from mdat and then scatter
        # among the processors and update mdat
        mdat_files_local = comm.scatter(mdat_files)
        mdat_local = [loadmat(m) for m in mdat_files_local]
        mdat_list = comm.allgather(mdat_local)
        mdat_global = []
        # instead of a list of lists, create a list of mdat
        for mlist in mdat_list:
            mdat_global.extend(mlist)

        # load attributes
        for attrname in discretization.vector_names:
            if discretization_name + attrname in list(mdat_global[0].keys()):
                if attrname.endswith('_local') and comm.size != \
                        len(mdat_list):
                    # create lists of local data
                    temp_input = None
                else:
                    temp_input = np.squeeze(mdat_global[0][
                        discretization_name + attrname])
                setattr(loaded_disc, attrname, temp_input)

        # load sample sets
        for attrname in discretization.sample_set_names:
            if attrname is not '_input_sample_set' and \
                    attrname is not '_output_sample_set':
                setattr(loaded_disc, attrname,
                        load_sample_set(file_name,
                                        discretization_name + attrname))

        # re-localize if necessary
        if file_name.startswith('proc_') and comm.size > 1 \
                and comm.size != len(mdat_list):
            warn_string = "Local pointers have been removed and will be"
            warn_string += " re-created as necessary)"
            warnings.warn(warn_string)
            loaded_disc._io_ptr_local = None
            loaded_disc._emulated_ii_ptr_local = None
            loaded_disc._emulated_oo_ptr_local = None
    return loaded_disc


def load_discretization(file_name, discretization_name=None):
    """
    Loads a :class:`~bet.sample.discretization` from a ``.mat`` file. If a file
    contains multiple :class:`~bet.sample.discretization` objects then
    ``discretization_name`` is used to distinguish which between different
    :class:`~bet.sample.discretization` objects.

    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string discretization_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.discretization` objects to a single
        ``.mat`` file

    :rtype: :class:`~bet.sample.discretization`
    :returns: the ``discretization`` that matches the ``discretization_name``

    """

    # check to see if parallel file name
    if file_name.startswith('proc_'):
        pass
    elif not os.path.exists(file_name) and os.path.exists(os.path.join(
            os.path.dirname(file_name),
            "proc{}_{}".format(comm.rank, os.path.basename(file_name)))):
        return load_discretization_parallel(file_name, discretization_name)

    mdat = loadmat(file_name)

    if discretization_name is None:
        discretization_name = 'default'

    input_sample_set = load_sample_set(file_name,
                                       discretization_name +
                                       '_input_sample_set')

    output_sample_set = load_sample_set(file_name,
                                        discretization_name +
                                        '_output_sample_set')

    loaded_disc = discretization(input_sample_set, output_sample_set)

    for attrname in discretization.sample_set_names:
        if attrname is not '_input_sample_set' and \
                attrname is not '_output_sample_set':
            setattr(loaded_disc, attrname,
                    load_sample_set(file_name, discretization_name + attrname))

    for attrname in discretization.vector_names:
        if discretization_name + attrname in list(mdat.keys()):
            setattr(loaded_disc, attrname,
                    np.squeeze(mdat[discretization_name + attrname]))

    # re-localize if necessary
    if file_name.rfind('proc_') == 0 and comm.size > 1:
        loaded_disc._io_ptr_local = None
        loaded_disc._emulated_ii_ptr_local = None
        loaded_disc._emulated_oo_ptr_local = None

    return loaded_disc


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

        Estimates the volume fraction of the Voronoice cells associated
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
            sample_radii[sample_radii <=
                         0] = prob_est_radii[sample_radii <= 0]

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
#             logging.warning(msg)
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
        self._volumes[0:-
                      1] = np.prod(self._width[0:-
                                               1] /
                                   domain_width, axis=1)
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
            raise length_not_matching(
                "Different number of centers and radii.")
        for i in range(len(centers)):
            if len(centers[i]) != self._dim:
                msg = "Center " + repr(i) + \
                    " has the wrong number of entries."
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
                    '_emulated_ii_ptr_local', '_emulated_oo_ptr',
                    '_emulated_oo_ptr_local']
    #: List of attribute names for attributes that are
    #: :class:`sample.sample_set_base`
    sample_set_names = ['_input_sample_set', '_output_sample_set',
                        '_emulated_input_sample_set',
                        '_emulated_output_sample_set',
                        '_output_probability_set']

    def __init__(self, input_sample_set, output_sample_set,
                 output_probability_set=None,
                 emulated_input_sample_set=None,
                 emulated_output_sample_set=None):
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
        self._io_ptr = None
        #: Pointer from ``self._emulated_input_sample_set`` to
        #: ``self._input_sample_set``
        self._emulated_ii_ptr = None
        #: Pointer from ``self._emulated_output_sample_set`` to
        #: ``self._output_probability_set``
        self._emulated_oo_ptr = None
        #: local io pointer for parallelism
        self._io_ptr_local = None
        #: local emulated ii ptr for parallelsim
        self._emulated_ii_ptr_local = None
        #: local emulated oo ptr for parallelism
        self._emulated_oo_ptr_local = None
        #: iteration number
        self._iteration = 0
        #: iteration dictionary to hold information
        self._setup = {0: {'col': False,
                           'rep': False,
                           'ind': None,
                           'qoi': 'SWE',
                           'std': None,
                           'obs': None,
                           'pre': None,
                           'model': None}}

        if output_sample_set is not None:
            self.check_nums()
            # TK - edit this out
            # if output_probability_set is not None:
            #    self.set_io_ptr(globalize=True)
        else:
            logging.info("No output_sample_set")

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

    def local_to_global(self):
        """
        Call local_to_global for ``input_sample_set`` and
        ``output_sample_set``.
        """
        if self._input_sample_set is not None:
            self._input_sample_set.local_to_global()
        if self._output_sample_set is not None:
            self._output_sample_set.local_to_global()

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
        my_copy = discretization(self._input_sample_set.copy(),
                                 self._output_sample_set.copy())

        for attrname in discretization.sample_set_names:
            if attrname is not '_input_sample_set' and \
                    attrname is not '_output_sample_set':
                curr_sample_set = getattr(self, attrname)
                if curr_sample_set is not None:
                    setattr(my_copy, attrname, curr_sample_set.copy())

        for array_name in discretization.vector_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(my_copy, array_name, np.copy(current_array))
        # copy setup information
        if self._setup is not None:
            import copy
            my_copy._setup = copy.deepcopy(self._setup)
        initial = self.get_initial()
        if initial is not None:
            my_copy.set_initial(
                initial.dist,
                gen=False,
                *initial.args,
                **initial.kwds)
        return my_copy

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
        if self._output_sample_set is not None:
            if self._output_sample_set._values_local is not None:
                num = self._output_sample_set._values_local.shape[1]
                if output_probability_set._values is not None:
                    try:
                        self.set_io_ptr(globalize=False)
                    except dim_not_matching:  # handle data-driven case
                        raise AttributeError("DATA DRIVEN")
                        self._io_ptr_local = np.arange(num)

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
        if self._output_sample_set is not None:
            if self._output_sample_set._values_local is not None:
                if emulated_output_sample_set._values is not None:
                    self.set_emulated_oo_ptr(globalize=False)

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
        if self._input_sample_set._values_local is not None:
            if emulated_input_sample_set._values is not None:
                self.set_emulated_ii_ptr(globalize=False)

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
        ps = self._output_probability_set
        ei = self._emulated_input_sample_set
        eo = self._emulated_output_sample_set
        return discretization(input_sample_set=ci,
                              output_sample_set=co,
                              output_probability_set=ps,
                              emulated_input_sample_set=ei,
                              emulated_output_sample_set=eo)

    def clip_samples(self, inds):
        """
        Creates and returns a discretization with samples indexed by
        `inds`, of the input and output sample sets.

        :param tuple inds: indices of values of sample set to return

        :rtype: :class:`~bet.sample.discretization`
        :returns: clipped discretization

        """
        ci = self._input_sample_set.clip(inds)
        co = self._output_sample_set.clip(inds)
        ps = self._output_probability_set
        ei = self._emulated_input_sample_set
        eo = self._emulated_output_sample_set
        return discretization(input_sample_set=ci,
                              output_sample_set=co,
                              output_probability_set=ps,
                              emulated_input_sample_set=ei,
                              emulated_output_sample_set=eo)

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
        mei = self._emulated_input_sample_set.\
            merge(disc._emulated_input_sample_set)
        meo = self._emulated_output_sample_set.\
            merge(disc._emulated_output_sample_set)

        return discretization(input_sample_set=mi,
                              output_sample_set=mo,
                              output_probability_set=self.
                              _output_probability_set,
                              emulated_input_sample_set=mei,
                              emulated_output_sample_set=meo)

    def choose_outputs(self, outputs=None):
        """
        Slices outputs of discretization (columns) and returns object with the
        same input sample set. For new instances, use `choose_inputs_outputs`.
        This function is of particular use for iterated ansatzs.

        :param list outputs: list of indices of output sample set to include

        :rtype: :class:`~bet.sample.discretization`
        :returns: sliced discretization

        """
        slice_list = ['_values', '_values_local',
                      '_error_estimates', '_error_estimates_local']

        output_ss = sample_set(len(outputs))
        output_ss.set_p_norm(self._output_sample_set._p_norm)
        if self._output_sample_set._domain is not None:
            output_ss.set_domain(self._output_sample_set._domain[outputs, :])
        if self._output_sample_set._reference_value is not None:
            output_ss.set_reference_value(
                self._output_sample_set._reference_value[outputs])

        for obj in slice_list:
            val = getattr(self._output_sample_set, obj)
            if val is not None:
                setattr(output_ss, obj, val[:, outputs])

        disc = discretization(input_sample_set=self._input_sample_set,
                              output_sample_set=output_ss)
        return disc

    def choose_inputs_outputs(self,
                              inputs=None,
                              outputs=None):
        """
        Slices the inputs and outputs of the discretization (columns).

        :param list inputs: list of indices of input sample set to include.
        :param list outputs: list of indices of output sample set to include

        :rtype: :class:`~bet.sample.discretization`
        :returns: sliced discretization

         .. note ::
            If you pass None instead of list, all indices are kept.
            This can be useful for re-arranging the order of variables,
            creating repeated columns of data, or truncating spaces.
        """
        slice_list = ['_values', '_values_local',
                      '_error_estimates', '_error_estimates_local']
        slice_list2 = ['_jacobians', '_jacobians_local']

        if inputs is None:  # instead of error message, copy input.
            inputs = np.arange(self._input_sample_set._dim)
        if outputs is None:  # instead of error message, copy output.
            outputs = np.arange(self._output_sample_set._dim)

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

    def likelihood(self, x=None):
        L = self._output_probability_set._distribution
        flip = False
        if self._setup[self._iteration]['col']:
            flip = True
            self._setup[self._iteration]['col'] = False
        if x is None:
            x = self.format_output_values()
        else:
            x = self.format_output_values(x)
        if flip:
            self._setup[self._iteration]['col'] = True
        # our pdf function already returns products.
        return self._output_probability_set.pdf(x=x, dist=L)

    def set_likelihood(self, dist=None):
        """
        TO DO: set this up properly
        """
        flip = False
        if self._setup[self._iteration]['col']:
            flip = True
            self._setup[self._iteration]['col'] = False
        if dist is None:
            obs = self.get_observed_distribution()
        else:
            obs = dist
        std = self.get_std()
        data = self.get_data()
        if data is None:
            logging.warning("Missing data. Using mean of observed.")
            data = obs.mean()
        if flip:
            self._setup[self._iteration]['col'] = True
        self._output_probability_set._dim = len(data)
        self._output_probability_set.set_distribution(
            dist=obs.dist, scale=std, loc=data)

    def get_initial_distribution(self):
        return self._input_sample_set.get_distribution()

    def get_initial(self):
        return self.get_initial_distribution()

    def get_observed_distribution(self, iteration=None):
        if iteration is None:
            iteration = self._iteration
        return self._setup[iteration]['obs']

    def get_observed(self, iteration=None):
        return self.get_observed_distribution(iteration)

    def get_predicted(self, iteration=None):
        return self.get_predicted_distribution(iteration)

    def get_predicted_distribution(self, iteration=None):
        if iteration is None:
            iteration = self._iteration
        return self._setup[iteration]['pre']

    def set_predicted_distribution(self, dist=None,
                                   iteration=None, *args, **kwds):
        r"""
        Wrapper for `compute_pushforward` if `dist==None`. Otherwise this
        function sets the predicted distribution (the distribution associated
        with the `output_sample_set` in the `discretization`.
        """
        if dist is None:
            return self.compute_pushforward(dist, iteration)
        elif isinstance(dist, scipy.stats._continuous_distns.rv_continuous) or \
                isinstance(dist, scipy.stats._distn_infrastructure.rv_frozen):
            self._output_sample_set.set_distribution(dist, *args, **kwds)
            if iteration is None:
                iteration = self._iteration
            self._setup[iteration]['pre'] = self._output_sample_set.\
                get_distribution()
        else:
            return self.compute_pushforward(dist, iteration, *args, **kwds)

    def set_predicted(self, dist=None, iteration=None, *args, **kwds):
        r"""
        Wrapper for `set_predicted_distribution`.
        """
        return self.set_predicted_distribution(dist, iteration, *args, **kwds)

    def set_initial_distribution(
            self, dist=None, num=None, gen=True, *args, **kwds):
        r"""
        If dist=None and num is not None, then just re-generate
        samples and map outputs using existing distribution.
        """

        if num is None:
            num = self._input_sample_set.check_num()

        if dist is None:
            if self._input_sample_set._distribution is None:
                logging.warning("Assuming independent Gaussian.")
                self._input_sample_set.set_distribution()
            else:  # do not change existing distribution.
                pass
        else:  # if a new distribution is specified, re-generate samples
            self._input_sample_set.set_distribution(dist, *args, **kwds)
        if gen:
            self._input_sample_set.generate_samples(num)
            
            # map output samples
            lam_ref = self._input_sample_set._reference_value
            # initialize output samples
            y = np.zeros((num, 1))  # temporary vector of correct shape
            v = np.array([])
            # TK - TODO: support string-indexed dictionaries, consistent order,
            # unify with pdf methods.
            for iteration in self._setup.keys():  # map through every model
                # clear all push-forwards
                self._setup[iteration]['pre'] = None
                model = self._setup[iteration]['model']
                # ensure model output size is consistent
                if model is not None:
                    z = model(self._input_sample_set._values)
                    if lam_ref is not None:
                        try:  # handle reference value
                            v = np.concatenate((v, model(lam_ref)), axis=0)
                        except ValueError:  # handle scalar models
                            v = np.column_stack(
                                (v, model(lam_ref).reshape(1,)))
                    try:
                        y = np.column_stack((y, z))
                    except np.AxisError:  # handle scalar models
                        y = np.concatenate((y, z.reshape(-1, 1)), axis=1)
            y = y[:, 1:]  # remove zeros
            self._output_sample_set._dim = y.shape[1]
            self._output_sample_set.set_values(y)
            if lam_ref is not None:
                self._output_sample_set.set_reference_value(v)

    def set_initial(self, dist=None, num=None, gen=True, *args, **kwds):
        r"""
        Wrapper for `set_initial_distribution`.
        """
        return self.set_initial_distribution(dist, num, gen, *args, **kwds)

    def set_observed_distribution(
            self, dist=None, iteration=None, *args, **kwds):
        r"""
        Set output_probability_set._distribution. Default assumption is N(0,1).
        """

        # the purpose of the probability set is to hold the evaluations
        # of the QoI at the current iteration step in its values.
        # also, to build in support for voronoi-cell approach (TODO later)
        if iteration is None:
            iteration = self._iteration

        inds = self.get_output_indices(iteration)  # returns ALL if None
        dim = len(inds)
        if self._output_probability_set is None:
            self.set_output_probability_set(sample_set(dim))
        self._output_probability_set._dim = dim  # will need this to match.

        if dist is None:  # normal by default
            from scipy.stats.distributions import norm
            dist = norm
            logging.info("Assuming normal distribution of noise.")
        try:  # is user providing information about error?
            scale = kwds.pop('scale')
            # enforce correct size of distribution.
            if isinstance(scale, float) or isinstance(scale, int):
                scale = scale * np.ones(dim)
        except KeyError:
            scale = np.ones(dim)

        self._output_probability_set._dim = dim  # write dimension info
        self._output_probability_set.set_distribution(
            dist, scale=scale, *args, **kwds)
        # Store distribution for iterated re-use
        obs_dist = self._output_probability_set._distribution
        self._setup[iteration]['obs'] = obs_dist

        # Store information about standard deviation for later use.
        logging.info(
            "Setting standard deviation information for output data.")
        self.set_std(obs_dist.std(), iteration=iteration)

    def set_observed(self, dist=None, iteration=None, *args, **kwds):
        r"""
        Wrapper for ``set_observed_distribution``
        """
        return self.set_observed_distribution(dist, iteration, *args, **kwds)

    def compute_pushforward(self, dist=None, iteration=None, *args, **kwds):
        # avoid accept/reject if possible
        if iteration is None:  # if not provided, assume "current"
            iteration = self._iteration

        self._output_sample_set.local_to_global()
        data = self.format_output_values(iteration=iteration)
        if data is None:
            raise AttributeError("Missing output values")

        if dist is None:
            from scipy.stats import gaussian_kde as gkde
            self._output_sample_set._distribution = gkde(
                data.T, *args, **kwds)
        elif isinstance(dist, str):
            self._output_sample_set.set_distribution(dist, *args, **kwds)
        else:
            self._output_sample_set._distribution = dist(data, *args, **kwds)

        # Store distribution for iterated re-use
        self._setup[iteration]['pre'] = self._output_sample_set._distribution

    # move this somewhere else:
    def iterate(self):
        # what to copy over? what to leave as default?
        self._iteration += 1
        it = self._iteration
        self.default_setup()
        # inherit all features in the new iteration
        self._setup[it] = self._setup[it - 1].copy()
        pass

    def set_iteration(self, iteration=0):
        self._iteration = iteration

    def initial_pdf(self, x=None):
        return self._input_sample_set.pdf(x)

    def predicted_pdf(self, x=None, iteration=None):
        if iteration is None:  # if not provided, assume "current"
            iteration = self._iteration

        if self._setup[iteration]['pre'] is None:
            logging.info("Missing predicted distribution. Computing now.")
            self.compute_pushforward(iteration=iteration)

        for i in range(0, iteration + 1):  # get all previous
            data_driven_status = self._setup[i]['col']
            if data_driven_status:
                s = self._setup[i]['qoi']
                logging.info("Iteration %i is using %s." % (i, s))
            pre = self._setup[i]['pre']  # load predicted dist
            data = self.format_output_values(x=x, iteration=i)
            temp_eval = np.log(self._output_sample_set.pdf(x=data,
                                                           dist=pre))
            if i == 0:
                out = temp_eval
            else:
                out += temp_eval

        return np.exp(out)

    def observed_pdf(self, x=None, iteration=None):
        r"""
        Evaluate the observed pdf on a provided set of points.

        Notes
        -----
        This is an alias for `~bet.sample.sample_set.pdf`.  See the ``pdf``
        docstring for more details.

        :param x: points for evaluation of probability density function
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``

        """
        if iteration is None:  # if not provided, assume "current"
            iteration = self._iteration

        for i in range(0, iteration + 1):  # get all previous
            data = self.format_output_values(x=x, iteration=i)
            dim = data.shape[1]
            data_driven_status = self._setup[i]['col']
            if data_driven_status:
                data_driven_mode = self._setup[i]['qoi']
                if data_driven_mode is 'SWE':
                    from scipy.stats.distributions import norm
                    obs = norm(loc=0, scale=1)
                elif data_driven_mode is 'MSE':
                    from scipy.stats.distributions import gamma
                    obs = gamma(a=dim / 2.0, scale=2.0 / dim)
                elif data_driven_mode is 'SSE':
                    from scipy.stats.distributions import chi2
                    obs = chi2(df=dim)
                else:
                    raise ValueError("Could not infer QoI type.")
            else:  # attempt to set observed based on stored info
                obs = self._setup[i]['obs']  # load observed dist

            temp_eval = self._output_probability_set.pdf(x=data, dist=obs)
            if i == 0:
                out = temp_eval
            else:
                out *= temp_eval
        return out

    def ratio_pdf(self, x=None):
        r"""
        Evaluate the estimated ratio pdf on a provided set of points.
        The ratio is the observed to the predicted densities.

        Notes
        -----
        This is a convenience alias for division between two evaluations of
        `~bet.sample.sample_set.pdf`.
        See the ``pdf`` docstring for more details.

        :param x: points for evaluation of probability density function
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``

        """
        return self.observed_pdf(x) / self.predicted_pdf(x)

    def normalized_ratio(self, x=None):
        r"""
        Evaluate the estimated ratio pdf on a provided set of points.
        The ratio is the observed to the predicted densities.
        Then, divide by the maximum.

        Notes
        -----
        This is a convenience alias for `~bet.sample.discretization.ratio_pdf`.
        It performs normalization, returning ratio/max(ratio).
        This is particularly helpful for accept/reject procedures.

        """
        ratio = self.ratio_pdf(x)
        return ratio / max(ratio)

    def get_input_values(self):
        return self._input_sample_set._values

    def get_output_values(self):
        return self._output_sample_set._values

    def set_model(self, model, iteration=None):
        if iteration is None:
            iteration = self._iteration
        if self._setup[iteration]['model'] is not None:
            msg = "Overwriting existing model, but not mapping outputs. "
            msg += "Run`.set_initial(gen=False)` to re-compute output values."
            logging.warning(msg)
        self._setup[iteration]['model'] = model

    def updated_pdf(self, x=None, iteration=None):
        r"""
        Evaluate the updated pdf on a provided set of points.

        :param x: points for evaluation of probability density function
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``

        """
        if iteration is None:
            iteration = self._iteration

        check_num = False
        if x is None:
            x = self._input_sample_set._values
            y = self._output_sample_set._values
            check_num = True
        else:
            num = x.shape[0]
            y = np.empty((num, 0))  # temporary vector of correct shape

            for iteration in self._setup.keys():  # map through every model
                inds = self.get_output_indices(iteration)
                unique = len(np.unique(inds))
                model = self._setup[iteration]['model']
                # ensure model output size
                if model is not None:
                    z = model(x).reshape(num, unique)[:, inds]
                    y = np.concatenate((y, z), axis=1)

        den = self.initial_pdf(x) * self.ratio_pdf(y)
        if check_num:
            assert len(den) == self.check_nums()
            self._input_sample_set.set_densities(den)
        else:
            assert len(den) == x.shape[0]
        return den

    def set_probabilities_from_densities(self):
        """
        """
        dens = self.updated_pdf()
        # initial probability is 1/N, density known.
        vols = 1. / (self.check_nums() * self.initial_pdf())
        probs = dens * vols
        self._input_sample_set.set_probabilities(probs / np.sum(probs))
        self._input_sample_set.set_volumes(vols)

    def set_prob_from_den(self):
        return self.set_probabilities_from_densities()

    def clip_accepted_samples(self):
        """
        Perform accept/reject and return a new sample set
        containing only the subset of samples that
        are distributed according to the updated density.
        """
        ratio = self.normalized_ratio()
        num_samples = self.check_nums()
        randnums = np.random.rand(num_samples)
        accept_inds = [i for i in range(num_samples)
                       if ratio[i] > randnums[i]]
        return self.clip_samples(accept_inds)

    def mud_index(self, x=None, iteration=None):
        """
        Return index of maximum updated density value.
        """
        den = self.updated_pdf(x, iteration)
        mud_idx = np.argmax(den)
        return mud_idx

    def mud_point(self, x=None, iteration=None):
        """
        Return maximum updated density value.
        """
        mud_idx = self.mud_index(x, iteration)
        return self._input_sample_set._values[mud_idx, :]

    def mud_value(self, x=None, iteration=None):
        """
        Return output value corresponding to maximum updated density value.
        """
        mud_idx = self.mud_index(x, iteration)
        return self._output_sample_set._values[mud_idx, :]

    def posterior_pdf(self, x=None, iteration=None):
        r"""
        Evaluate the updated pdf on a provided set of points.

        :param x: points for evaluation of probability density function
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``

        """
        if iteration is None:
            iteration = self._iteration

        if x is None:
            x = self._input_sample_set._values
            y = self._output_sample_set._values
        else:
            y = np.zeros((x.shape[0], 1))  # temporary vector of correct shape

            for iteration in self._setup.keys():  # map through every model
                inds = self.get_output_indices(iteration)
                unique = len(np.unique(inds))
                model = self._setup[iteration]['model']
                # ensure model output size
                if model is not None:
                    z = model(x).reshape(-1, unique)[:, inds]
                    y = np.concatenate((y, z), axis=1)

            y = y[:, 1:]  # remove zeros
        den = self.initial_pdf(x) * self.likelihood(y)
        if x is not None:
            assert len(den) == x.shape[0]
        else:
            assert len(den) == self.check_nums()
        return den

    def map_index(self, x=None, iteration=None):
        """
        Return index of maximum aposteriori point.
        """
        den = self.posterior_pdf(x, iteration)
        map_idx = np.argmax(den)
        return map_idx

    def map_point(self, x=None, iteration=None):
        """
        Return maximum aposteriori point.
        """
        map_idx = self.map_index(x, iteration)
        return self._input_sample_set._values[map_idx, :]

    def map_value(self, x=None, iteration=None):
        """
        Return output value corresponding to maximum aposteriori point.
        """
        map_idx = self.map_index(x, iteration)
        return self._output_sample_set._values[map_idx, :]

    def set_noise_model(self, dist=None, iteration=None, *args, **kwds):
        """
        dist can be a number, in which case dimension is inferred
        from indices
        """
        if iteration is None:
            iteration = self._iteration
        if dist is None:
            logging.warning("Assuming Normal, pulling from get_std.")
            dist = self.get_std(iteration)
        logging.info(
            "With this option, you will be inferring std from observed.")
        from scipy.stats.distributions import norm
        dim_output = len(self.get_output_indices(iteration))
        if self._setup[iteration]['rep'] is not None:
            dim_output = 1
        if isinstance(dist, int) or isinstance(dist, float):
            std = np.ones(dim_output) * dist
            dist = norm(scale=std)
        elif isinstance(dist, list) or isinstance(dist, tuple):
            if len(dist) != dim_output:
                raise dim_not_matching("Does not match data dimension.")
            std = np.array(dist)
            dist = norm(scale=std)
        elif isinstance(dist, np.ndarray):
            std = dist
            dist = norm(scale=std)

        if isinstance(dist, scipy.stats.distributions.rv_frozen):
            self._setup[self._iteration]['obs'] = dist  # store it
            std = self._setup[self._iteration]['obs'].std()
        else:
            self._setup[self._iteration]['obs'] = dist(*args, **kwds)
            std = self._setup[self._iteration]['obs'].std()
        self._setup[self._iteration]['std'] = None
        self._setup[self._iteration]['pre'] = None

    def loss_fun(self, outputs, data, data_std, mode='SWE'):
        # if std vector has shape mismatch, this will error out:
        weighted_residuals = np.divide((outputs - data), data_std)
        if mode is 'SWE':  # sum weighted errors
            qoi = np.sum(weighted_residuals, axis=1)
        elif mode is 'MSE':  # mean squared error
            qoi = (1. / np.sqrt(len(data))) * \
                np.sum(np.power(weighted_residuals, 2), axis=1)
        elif mode is 'SSE':  # sum squared error
            qoi = np.sum(np.power(weighted_residuals, 2), axis=1)
        else:
            raise ValueError("Choose mode from [SWE, MSE, SSE]")
        # always returning 1-D output from this function.
        return qoi.reshape(-1, 1)

    def format_output_values(self, x=None, iteration=None):
        if iteration is None:  # get current if None
            iteration = self._iteration
        inds = self.get_output_indices(iteration=iteration)
        if x is None:  # grab most recent values by default
            qoi = self._output_sample_set._values[:, inds]
        else:  # attempt to parse provided input values
            try:
                qoi = x[:, inds]
            except np.AxisError:  # row-vector support
                qoi = x[inds].reshape(-1, 1)
            except IndexError:  # perhaps just passing relevant
                logging.warning("Could not index data. Setting as-is.")
                qoi = x  # support already-formatted data

        # Now our data is the correct dimension to be passed
        # to both observed and predicted, unless data-driven.
        # if data-driven, we have to collapse the data and write it
        # to output_probability_set.
        data_driven_status = self._setup[iteration]['col']

        # we now have to transform our QoI data if we are in data-driven mode.
        if data_driven_status:
            data_driven_mode = self._setup[iteration]['qoi']
            std = self.get_std(iteration)
            data = self.get_data(iteration)
            qoi = self.loss_fun(outputs=qoi, data=data,
                                data_std=std, mode=data_driven_mode)

        return qoi

    def set_data_driven_mode(self, qoi='SWE', iteration=None):
        if iteration is None:
            iteration = self._iteration
        if qoi in ['SSE', 'MSE', 'SWE']:
            self._setup[iteration]['qoi'] = qoi
        else:
            raise ValueError('Please specify one of SWE/MSE/SWE')

    def set_repeated(self, repeat=None, iteration=None):
        r"""
        Toggle mode for repeated observations.

        :param repeat: output indices for repeated observations
        :type repeat: `float` or `tuple` of :type:`int`, or `int`

        """
        if iteration is None:
            iteration = self._iteration
        if repeat is None:
            repeat = iteration
        if isinstance(repeat, list) or isinstance(repeat, tuple):
            self._setup[iteration]['rep'] = list(np.array(repeat, dtype=int))
        else:
            if repeat < 0 or repeat > self._output_sample_set._dim:
                msg = "Improper output index specified"
                msg += "for the case of repeated observations."
                raise ValueError(msg)
            self._setup[iteration]['rep'] = int(repeat)

    def set_data_driven(self, collapse=True, iteration=None):
        r"""
        Toggle mode for data-driven map.

        :param bool collapse: option to collapse outputs to 1-D

        """
        if iteration is None:
            iteration = self._iteration
        self._setup[iteration]['col'] = collapse

    def get_data(self, iteration=None):
        r"""
        Formats the indices according to `setup` and then
        returns relevant data vector for a particular iteration.
        """
        if iteration is None:
            iteration = self._iteration

        if self._output_probability_set is None:
            inds = np.arange(self._output_sample_set._dim)
            logging.warning("Returning reference value.")
            return self._output_sample_set._reference_value[inds]
        elif self._output_probability_set._reference_value is None:
            inds = np.arange(self._output_sample_set._dim)
            if self._output_sample_set._reference_value is not None:
                return self._output_sample_set._reference_value[inds]
            else:
                msg = "Output reference is None. "
                msg += "Will use mean of observed as reference data!"
                logging.warning(msg)
                # Return zeros as placeholder.
                self.set_data_from_observed(iteration=iteration)
                return self.get_data(iteration)
        else:  # (noisy) data is not None
            inds = self.get_data_indices(iteration)
            return self._output_probability_set._reference_value[inds]

    def set_data(self, data, std=None, inds=None, iteration=None):
        r"""
        Write the reference value to the output probability set.
        (Bypass the checking of dimension).
        If you pass a noise level, we will take note of it.
        If one does not exist, create one to match the dimension.
        inds = None default is to use all data.
        """
        if iteration is None:
            iteration = self._iteration

        dim = len(data)
        self._setup[iteration]['ind'] = inds
        if self._output_probability_set is None:
            logging.warning("Missing output probability set. Creating.")
            self._output_probability_set = sample_set(dim)

        if self._output_probability_set._reference_value is None:
            self._output_probability_set._reference_value = np.copy(data)
        else:
            inds = self.get_data_indices(iteration)
            if len(inds) != dim:
                msg = 'Indices do not match data length. '
                msg += 'Overwriting data entirely.'
                # raise ValueError('Indices do not match data length.')
                logging.warning(msg)
                self._output_probability_set.\
                    _reference_value = np.copy(data)
            else:
                self._output_probability_set.\
                    _reference_value[self.get_data_indices()] = np.copy(data)

        if std is None:
            if self._setup[iteration]['std'] is None:
                if self._setup[iteration]['obs'] is None:
                    logging.warning(
                        "No way to infer std. Will use sample variance.")
                    if len(self.get_data_indices(iteration)) == 1:
                        logging.warning(
                            "Cannot take sample variance of singleton.")
                else:
                    logging.warning(
                        "No std provided. Will use std from observed.")
            else:
                logging.warning(
                    "No std provided. Using existing entry in setup.")
        else:
            self._setup[iteration]['std'] = std
        if self._setup[iteration]['col']:
            # clear predicted since we just changed indices, data, or std, all
            # of which are part of the QoI model definition.
            self._setup[iteration]['pre'] = None

    def set_indices(self, inds, iteration=None):
        r"""
        Whatever is passed is written to `setup`. Clear predicted entry.
        """
        if iteration is None:
            iteration = self._iteration
        self._setup[iteration]['ind'] = inds
        self._setup[iteration]['pre'] = None
        self._setup[iteration]['obs'] = None
        logging.warning('Observed and Predicted entries cleared.')

    def get_data_indices(self, iteration=None):
        r"""
        Reads in value from `setup` and converts it to the proper
        indices for `output_probability_set`.
        """
        if iteration is None:
            iteration = self._iteration
        if self._output_probability_set._reference_value is None:
            if self._output_sample_set._reference_value is None:
                logging.info(
                    "Using length of output samples because reference empty.")
                data_len = self._output_sample_set._dim
            else:
                logging.info(
                    "Using length of output reference because data empty.")
                data_len = len(self._output_sample_set._reference_value)
        else:
            data_len = len(self._output_probability_set._reference_value)
        return self.format_indices(data_len, self._setup[iteration]['ind'])

    def get_output_indices(self, iteration=None):
        r"""
        Reads in value from `setup` and converts it to indices
        for `output_sample_set`. In the case of repeated observations,
        if incomplete data is available, use it.
        """
        if iteration is None:
            iteration = self._iteration
        if self._output_probability_set is not None:
            if self._output_probability_set._reference_value is not None:
                data_len = len(self.get_data(iteration))
            else:
                data_len = self._output_probability_set._dim
        else:
            data_len = self._output_sample_set._dim
        inds = self.format_indices(data_len, self._setup[iteration]['ind'])
        rep = self._setup[iteration]['rep']
        if rep is not False:  # handle repeated observations
            if isinstance(rep, np.ndarray):
                rep = list(rep)

            if isinstance(rep, list) or isinstance(rep, tuple):
                rep_inds = list(np.tile(rep, len(inds) // len(rep)))
                if len(rep_inds) != len(
                        inds):  # data available that is unaccounted for
                    logging.warning(
                        "Data doesn't divide evenly into repeated indices.")
                    num_missing = len(inds) % len(rep_inds)
                    # add "missing" repeated values.
                    rep_inds.append(rep[:num_missing])
                inds = rep_inds  # set as output
            else:  # repeat vector of indices
                inds = np.ones(len(inds), dtype=int) * rep
        else:
            inds = np.arange(data_len)
        return list(inds)

    def format_indices(self, data_len, inds):
        r"""
        Converts entry in the `setup` dictionary into a set of usable
        indices for the data.
        """
        if inds is None:
            inds = np.arange(data_len)
        elif isinstance(inds, float):  # first or last n%
            if inds <= 1.0:
                if inds > 0:
                    inds = np.arange(0, int(data_len * inds))
                elif inds < 0:
                    inds = np.arange(int(data_len * (1 + inds)), data_len)
                else:  # inds = 0.0?
                    raise ValueError("Please specify nonzero index.")
            else:  # bootstrap if >= 1.0 to create 'larger' dataset
                inds = np.random.randint(0, data_len, int(data_len * inds))
        elif isinstance(inds, int):  # first or last n entries
            if inds < 0:
                inds = np.arange(data_len + inds, data_len)
            elif inds > 0:
                inds = np.arange(0, inds)
            else:  # inds = 0
                raise ValueError("Please specify nonzero index.")
        elif isinstance(inds, tuple):
            if len(inds) == 1:
                start = 0
                stop = data_len
                by = inds[0]
            elif len(inds) == 2:
                start = inds[0]
                stop = data_len
                by = inds[1]
            else:
                start = inds[0]
                stop = inds[1]
                by = inds[2]

            # if floats passed for any args, convert them.
            if isinstance(start, float):
                start = int(data_len * start)
            if isinstance(stop, float):
                stop = int(data_len * stop)
            if isinstance(by, float):
                by = int(data_len * by)
            if stop < start or by > data_len:
                raise ValueError(
                    "Ordering mismatch. Check index specifications.")
            inds = np.arange(start, stop, by)
        else:
            pass  # use lists  and numpy-arrays as-is
        return list(inds)

    def get_output(self):
        return self.get_output_sample_set()

    def get_input(self):
        return self.get_input_sample_set()

    def set_data_from_observed(self, obs='mean',
                               alpha=0.99,
                               iteration=None):
        r"""
        Perform a draw from the observed distribution.

        :param string obs: Type of draw to take. Choose from
                 'mean' (default), 'median',
                 'min'/'max' (pass `alpha`), or

        :param float alpha: If using obs='min'/'max',
                this represents the confidence level.
                Default value is 0.99.

        """
        if iteration is None:
            iteration = self._iteration
        obs_dist = self.get_observed_distribution(iteration)
        if obs == 'mean':
            data = obs_dist.mean()
        elif obs == 'median':
            data = obs_dist.median()
        elif obs == 'min':
            data = obs_dist.interval(alpha)[0]
        elif obs == 'max':
            data = obs_dist.interval(alpha)[1]
        std = obs_dist.std()
        self._setup[iteration]['std'] = std

        if self._output_probability_set._reference_value is None:
            self._output_probability_set._reference_value = np.copy(data)
        else:
            self._output_probability_set._reference_value[self.get_data_indices(
                iteration)] = np.copy(data)

    def set_data_from_reference(self, iteration=None, dist=None):
        # goes and grabs the reference output value for a particular iteration
        # and hits it with a noise model.

        if iteration is None:
            iteration = self._iteration
        # since we are accessing output values.
        inds = self.get_output_indices(iteration)

        if dist is None:
            logging.warning("Using observed as noise model.")
            dist = self._setup[iteration]['obs']

        # support repeating data.
        Q_ref = self._output_sample_set._reference_value
        direct = False
        if Q_ref is None:
            logging.info("Problem with output reference value.")
            model = self._setup[iteration]['model']
            lam_ref = self._input_sample_set._reference_value
            if lam_ref is None:
                msg = "Missing input reference value."
                msg += "Using draw directly from observed distribution."
                logging.warning(msg)
                direct = True
            else:
                if model is None:
                    if dist is None:
                        raise AttributeError("Missing model and observed.")
                    else:
                        msg = "Missing model, input reference value present."
                        msg += "Using draw from observed distribution."
                        logging.warning(msg)
                        direct = True
                else:
                    logging.info("Using model to map input reference value.")
                    Q_ref = model(lam_ref)[inds]
        else:  # existing reference is correctly set
            # support repeated observations using indices. correct length.
            Q_ref = Q_ref[inds]

        if (np.max(np.abs(dist.mean())) != 0):
            logging.warning(
                "Non-homogeneous noise. Using random draw for data.")
            direct = True
        if not direct:
            noise = self._output_probability_set.rvs(dist=dist)
            Q_ref = Q_ref + noise
        else:  # if observed distribution is non-homogogenous
            Q_ref = self._output_probability_set.rvs(dist=dist)

        self._output_probability_set._reference_value = Q_ref

        return Q_ref

    def set_std(self, std=None, iteration=None):
        if iteration is None:
            iteration = self._iteration
        # if None,
        if std is None:  # nothing passed
            if self._setup[iteration]['std'] is None:  # nothing written
                if self._setup[iteration]['obs'] is None:
                    logging.warning(
                        "Defaulting to estimating using data sample variance.")
                else:  # use observed to infer std if it is missing.
                    logging.warning(
                        "Inferring standard deviation from observed.")
                    # method std() belongs to distribution
        else:  # write as-is if anything except None
            # but if numpy array, convert to list.
            if isinstance(std, np.ndarray):
                std = std.tolist()
                # if singleton in array, extract it, continue.
                if len(std) == 1:
                    std = std[0]
                else:  # if array is wrong length, raise error.
                    if len(std) != len(self.get_data_indices(iteration)):
                        msg = "Wrong size std (mismatch with data indices)."
                        raise dim_not_matching(msg)

            if not(isinstance(std, int) or isinstance(std, float)):
                if len(std) != len(self.get_data_indices(iteration)):
                    if len(std) == 1:
                        std = std[0]
                    elif len(std) == self._output_sample_set._dim:
                        msg = "Wrong size std."
                        msg = "Assuming these correspond to repeated outputs."
                        logging.warning(msg)
                        pass
                    else:
                        msg = "Wrong size std (mismatch with data indices)."
                        raise dim_not_matching(msg)

        self._setup[iteration]['pre'] = None
        self._setup[self._iteration]['std'] = std
        # return what will result from setting the std this way.
        return self.get_std(iteration)

    def get_std(self, iteration=None):
        if iteration is None:
            iteration = self._iteration
        std = self._setup[iteration]['std']
        if std is None:  # if empty, get observed or estimate
            if self._setup[iteration]['obs'] is None:
                logging.info("Using sample variance estimate.")
                sample_data = self.get_data(iteration)
                if len(sample_data) == 1:
                    msg = "Cannot take variance of singeleton.\n"
                    msg += "Try changing indexes or turn-off data-driven"
                    raise AttributeError(msg)
                else:
                    std = np.std(sample_data)
            else:
                logging.info("Using variance from observed.")
                std = self._setup[iteration]['obs'].std()

        if isinstance(std, int) or isinstance(std, float):
            inds = self.get_data_indices(iteration)
            std = np.ones(len(inds)) * std
        elif isinstance(std, list) or isinstance(std, tuple):
            if len(std) == self._output_sample_set._dim:
                std = np.array(std)[self.get_output_indices(iteration)]

        return list(std)

    def data_driven(self, data=None, std=None, inds=None, iteration=None):
        r"""
        Requires reference output probability value to work.
        Understood to mean "data" already polluted with noise.
        If missing, we attempt to simulate it if a noise model and
        input reference value are peresent.
        If a distribution is present in `output_probability_set`, then we
        perturb the `output_sample_set` reference value and set it as the
        reference value in `output_probability_set`.
        passing inds alone can be like bootstrapping if using repeated.
        """
        self.set_data_driven(True)
        self.set_data(data=data, iteration=iteration, std=std, inds=inds)

    def get_setup(self, iteration=None):
        if iteration is not None:
            try:
                return self._setup[iteration]
            except KeyError:
                logging.warning("Iteration out of bounds. Returning current.")
                return self._setup[self._iteration]
        else:
            return self._setup[self._iteration]

    def default_setup(self):
        # check for data-driven with
        # if not abs(col): ... do normal. if 1, current inds, if -1, all inds.
        D = {'ind': None,
             'rep': False,
             'col': False,
             'qoi': 'SWE',
             'std': None,
             'obs': None,
             'pre': None,
             'model': None
             }
        try:
            self._setup[self._iteration] = D
        except TypeError:
            self._setup = {0: D}
