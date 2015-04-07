"""
This module contains unittests for :mod:`~bet.util`
"""

import unittest
import bet.util as util
from bet.Comm import *
import numpy.testing as nptest
import numpy as np
from pkgutil import iter_modules

def get_binary_rep(i, dim):
    short = bin(i).partition('b')[-1]
    full = '0'*(dim-len(short))+short
    print full

def compare_to_bin_rep(xnew):
    """
    xnew[i] == binaryformof(i, dim) where binaryformof(i, dim) returns a list
    of length dim of 1s and 0s corresponding to the binary represenation of i
    """
    rep_compare = np.zeros((xnew.shape[0],), np.bool)
    for i, row in enumerate(xnew):
        row_rep = ''
        for v in row:
            row_rep += str(v)
        print row_rep
        rep_compare[i] = (row_rep == get_binary_rep(i, len(row)))
    assert np.all(rep_compare)

def test_meshgrid_ndim():
    """
    Tests :meth:`bet.util.meshgrid_ndim` for 10 vectors where each vector is
    equal to ``[0, 1]``.
    """
    for i in xrange(2):
        x = [[0,1,] for v in xrange(i)]
        yield compare_to_bin_rep, util.meshgrid_ndim(x)

@unittest.skipIf(size == 1 and 'mpi4py' in (name for loader, name, 
    ispkg in iter_modules()), 
    "This test only runs in parallel. If mpi4py is installed")
def test_get_global_values():
    """
    TODO: Separate out parallel tests so that they are run in parallel if
    mpi4py is installed
    Tests :meth:`bet.util.get_global_values`.
    """
    original_array = np.array(range(rank*2))
    my_array = comm.scatter(original_array, root=0)
    recomposed_array = util.get_global_values(my_array)
    nptest.assert_array_equal(original_array, recomposed_array)

