# Copyright (C) 2014-2020 The BET Development Team

"""
This module contains general tools for BET including saving and loading objects, and reshaping objects. The most
important methods are:

* :mod:`~bet.util.get_global_values` concatenates local arrays into global arrays.
* :mod:`~bet.util.save_object` saves all types of objects.
* :mod:`~bet.util.load_object` loads all types of saved objects.
* :mod:`~bet.util.load_object_parallel` loads all types of saved parallel objects.

"""

import sys
import collections.abc
import os
import glob
import logging
import numpy as np
import bet.sample
from bet.Comm import comm, MPI

possible_types = {int: MPI.INT, float: MPI.DOUBLE}


def meshgrid_ndim(X):
    """
    Return coordinate matrix from two or more coordinate vectors.
    Handles a maximum of 10 vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays (x1, x2,..., xn).


    :param X: A tuple containing the 1d coordinate arrays
    :type X: tuple
    :rtype: :class:`~numpy.ndarray` of shape (num_grid_points,n)
    :returns: X_new
    """
    n = len(X)
    alist = []
    for i in range(n):
        alist.append(X[i])
    for i in range(n, 10):
        alist.append(np.array([0]))

    a, b, c, d, e, f, g, h, i, j = np.meshgrid(alist[0],
                                               alist[1],
                                               alist[2],
                                               alist[3],
                                               alist[4],
                                               alist[5],
                                               alist[6],
                                               alist[7],
                                               alist[8],
                                               alist[9],
                                               indexing='ij')

    X_new = np.vstack(
        (a.flat[:],
         b.flat[:],
         c.flat[:],
         d.flat[:],
         e.flat[:],
         f.flat[:],
         g.flat[:],
         h.flat[:],
         i.flat[:],
         j.flat[:])).transpose()
    X_new = X_new[:, 0:n]

    return X_new


def get_global_values(array, shape=None):
    """
    Concatenates local arrays into global array using :meth:`np.vstack`.

    :param array: Array.
    :type P_samples: :class:`~numpy.ndarray`
    :rtype: :class:`~numpy.ndarray`
    :returns: array
    """
    if comm.size == 1:
        return array
    else:
        # Figure out the subtype of the elements of the array
        dtype = array.dtype
        mpi_dtype = False
        for ptype in possible_types.keys():
            # suppress FutureWarning
            if ptype is int or ptype is MPI.INT:
                comp_type = np.integer
            elif ptype is float or ptype is MPI.FLOAT:
                comp_type = np.floating

            if np.issubdtype(dtype, comp_type):
                mpi_dtype = True
                dtype = ptype

        if shape is None or not mpi_dtype:
            # do a lowercase allgather
            a_shape = len(array.shape)
            array = comm.allgather(array)
            if a_shape <= 1:
                return np.hstack(array)
            else:
                return np.vstack(array)
        else:
            # do an uppercase Allgather
            whole_a = np.empty(shape, dtype=dtype)
            comm.Allgather([array.ravel(), possible_types[dtype]], [whole_a,
                                                                    possible_types[dtype]])
            return whole_a


def fix_dimensions_vector(vector):
    """
    Fix the dimensions of an input so that it is a :class:`numpy.ndarray` of
    shape (N,).
    :param vector: numerical object
    :rtype: :class:`numpy.ndarray`
    :returns: array of shape (N,)
    """
    if not isinstance(vector, collections.abc.Iterable):
        vector = np.array([vector])
    elif not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    return vector.flat[:]


def fix_dimensions_vector_2darray(vector):
    """
    Fix the dimensions of an input so that it is a :class:`numpy.ndarray` of
    shape (N,1).

    :param vector: numerical object
    :rtype: :class:`numpy.ndarray`
    :returns: array of shape (N,1)

    """
    if not isinstance(vector, collections.abc.Iterable):
        vector = np.array([vector])
    elif not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    if len(vector.shape) <= 1:
        vector = np.expand_dims(vector, axis=1)
    return vector


def fix_dimensions_domain(domain):
    """
    Fix the dimensions of an input so that it is a :class:`numpy.ndarray` of
    shape (dim, 2).

    :param vector: numerical object of at least length 2
    :type vector: :class:`collections.abc.Iterable`
    :rtype: :class:`numpy.ndarray`
    :retuns: array of shape (dim, 2)

    """
    if not isinstance(domain, np.ndarray):
        if len(domain) == 2:
            domain = np.expand_dims(domain, axis=0)
        else:
            raise TypeError("The length must be at least 2.")
    elif len(domain.shape) == 1 and domain.shape[0] == 2:
        domain = np.expand_dims(domain, axis=0)
    elif len(domain.shape) == 2 and domain.shape[1] == 2:
        pass  # The shape is already correct!
    elif len(domain.shape) == 2 and domain.shape[0] == 2:
        domain = domain.transpose()
    else:
        raise TypeError("At least one dimension must have a length of 2.")
    return domain


def fix_dimensions_data(data, dim=None):
    """
    Fix the dimensions of an input so that it is a :class:`numpy.ndarray` of
    shape (N, dim).

    If ``dim`` is non-specified:
    If ``data`` is a non-iterable number assumes that ``dim==1``.
    If ``data`` is a numpy array with len(shape) == 1 assumes that ``dim==1``.
    If ``data`` is a numpy array with len(shape) == 2 assumes that
    ``dim==shape[1]``.


    :param data: numerical object
    :param int dim: The dimension of the "data" space.
    :rtype: :class:`numpy.ndarray`
    :returns: array of shape (N, dim)

    """
    if dim is None:
        if not isinstance(data, np.ndarray):
            return fix_dimensions_vector_2darray(data)
        elif len(data.shape) == 1:
            return fix_dimensions_vector_2darray(data)
        else:
            return data

    data = fix_dimensions_vector_2darray(data)
    if len(data.shape) > 1 and data.shape[1] != dim:
        return data.transpose()
    else:
        return data


def clean_data(data):
    """
    Clean data so that NaN->0, inf-> maxfloat, -inf-> -maxfloat

    :param data: numerical object
    :type data: :class:`numpy.ndarray`
    :rtype: :class:`numpy.ndarray`
    :returns: array of shape (data.shape)

    """
    data[np.isnan(data)] = 0.0
    data[np.isinf(data)] = np.sign(data[np.isinf(data)]) * sys.float_info[0]

    return data


def save_object(save_set, file_name, globalize=True):
    """
    Save BET object.

    :param save_set: Object to Save.
    :param file_name: Filename to save to.
    :type file_name: str
    :param globalize: Whether or not to globalize parallel objects.
    :type globalize: bool
    """
    import pickle
    # create processor specific file name
    if comm.size > 1 and not globalize:
        local_file_name = os.path.join(os.path.dirname(file_name),
                                       "proc{}_{}".format(comm.rank,
                                                          os.path.basename(file_name)))
    else:
        local_file_name = file_name
    if os.path.exists(local_file_name + '.p'):
        logging.warn("Warning! Output file already exists. New object will be appended.")
    # globalize
    if globalize:
        save_set.local_to_global()
    comm.barrier()
    pickle.dump(save_set, open(local_file_name + '.p', "wb"))
    comm.barrier()
    return local_file_name


def load_object(file_name, localize=False):
    """
    Load saved objects.

    :param file_name: Filename of object.
    :type file_name: str
    :param localize: Whether or not to localize parallel object.
    :type localize: bool
    :return: The saved object
    """
    import pickle
    # check to see if parallel file name
    if file_name.startswith('proc_'):
        # logging.warning("Avoid starting filenames with 'proc_'. Unable to localize.")
        localize = False
    elif not os.path.exists(file_name+'.p') and os.path.exists('proc0_'+file_name+'.p'):
        return load_object_parallel(file_name)
    loaded_set = pickle.load(open(file_name+'.p', "rb"))
    if localize:
        loaded_set.global_to_local()
    return loaded_set


def load_object_parallel(file_name):
    """
    Load saved paralell objects.

    :param file_name: Filename of object.
    :type file_name: str
    :return: The saved object

    """
    save_dir = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    files = glob.glob(os.path.join(save_dir, "proc*_{}".format(base_name+'.p')))
    if len(files) == comm.size:
        logging.info("Loading sample set using parallel files (same nproc)")
        # if the number of processors is the same then set mdat to
        # be the one with the matching processor number (doesn't
        # really matter)
        local_file_name = os.path.join(os.path.dirname(file_name),
                                       "proc{}_{}".format(comm.rank,
                                                          os.path.basename(file_name)))
        return load_object(local_file_name)
    else:
        raise bet.sample.dim_not_matching("Number of parallel files is different from nproc.")
    # SM possibly re-add the feature to have different numbers. Probably not necessary.
