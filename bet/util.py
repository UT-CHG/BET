# Copyright (C) 2014-2015 The BET Development Team

"""
This module contains general tools for BET.
"""

import sys
import collections
import numpy as np
from bet.Comm import comm, MPI

possible_types = {int:MPI.INT, float:MPI.DOUBLE}

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
        for ptype in possible_types.iterkeys():
            if np.issubdtype(dtype, ptype):
                mpi_dtype = True
                dtype = ptype

        if shape is None or not mpi_dtype:
            # do a lowercase allgather
            a_shape = len(array.shape)
            array = comm.allgather(array)
            if a_shape == 1:
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
    if not isinstance(vector, collections.Iterable):
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
    if not isinstance(vector, collections.Iterable):
        vector = np.array([vector])
    elif not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    if len(vector.shape) == 1:
        vector = np.expand_dims(vector, axis=1)
    return vector

def fix_dimensions_domain(domain):
    """
    Fix the dimensions of an input so that it is a :class:`numpy.ndarray` of
    shape (dim, 2).

    :param vector: numerical object of at least length 2
    :type vector: :class:`collections.Iterable`
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
        pass # The shape is already correct!
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
    if data.shape[1] != dim:
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
    data[np.isinf(data)] = np.sign(data[np.isinf(data)])*sys.float_info[0]

    return data



