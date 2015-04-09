# Lindley Graham 04/07/2015
"""
This module contains unittests for :mod:`~bet.util`
"""

import bet.util as util
from bet.Comm import *
import numpy.testing as nptest
import numpy as np

def get_binary_rep(i, dim):
    """
    A ``dim`` bit representation of ``i`` in binary.

    :param int i: number to represent in binary
    :param int dim: number of bits to use in the representation
    :rtype: string
    :returns: string representation of binary represenation of i

    """
    short = bin(i).partition('b')[-1]
    full = '0'*(dim-len(short))+short
    return full

def compare_to_bin_rep(xnew):
    """
    xnew[i] == get_binar_rep(i, dim)     
    """
    rep_compare = np.zeros((xnew.shape[0],), np.bool)
    for i, row in enumerate(xnew):
        row_rep = ''
        for v in row:
            row_rep += str(v)
        rep_compare[i] = (row_rep == get_binary_rep(i, len(row)))
        print rep_compare[i]
    assert np.all(rep_compare)

def test_meshgrid_ndim():
    """
    Tests :meth:`bet.util.meshgrid_ndim` for upto 10 vectors where each vector is
    equal to ``[0, 1]``.
    """
    for i in xrange(10):
        x = [[0, 1] for v in xrange(i+1)]
        yield compare_to_bin_rep, util.meshgrid_ndim(x)

def test_get_global_values():
    """
    TODO: make sure the newer version of util matches this test

    Tests :meth:`bet.util.get_global_values`.
    """
    for i in xrange(5):
        yield compare_get_global_values, i

def compare_get_global_values(i):
    """
    Compares the results of get global values for a vector of shape ``(size*2,
    i)``.
    
    :param int i: Dimension of the vector of length ``size*2``

    """
    if rank == 0:
        if i == 0:
            original_array = np.array(np.random.random((size*2, )))
        else:
            original_array = np.array(np.random.random((size*2, i)))
    else:
        original_array = None
    original_array = comm.bcast(original_array)
    my_len = original_array.shape[0]/size
    my_index = range(0+rank*my_len, (rank+1)*my_len)
    if i == 0:
        my_array = original_array[my_index]
    else:
        my_array = original_array[my_index, :]
    recomposed_array = util.get_global_values(my_array)
    nptest.assert_array_equal(original_array, recomposed_array)

