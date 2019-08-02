# Copyright (C) 2014-2019 The BET Development Team

"""
This module contains unittests for :mod:`~bet.util`
"""

import bet.util as util
from bet.Comm import comm
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
    full = '0' * (dim - len(short)) + short
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
        print(rep_compare[i])
    assert np.all(rep_compare)


def test_meshgrid_ndim():
    """
    Tests :meth:`bet.util.meshgrid_ndim` for upto 10 vectors where each vector is
    equal to ``[0, 1]``.
    """
    for i in range(10):
        x = [[0, 1] for v in range(i + 1)]
        yield compare_to_bin_rep, util.meshgrid_ndim(x)


def test_get_global_values():
    """
    Tests :meth:`bet.util.get_global_values`.
    """
    for provide_shape in [True, False]:
        for i in range(5):
            yield compare_get_global_values, i, provide_shape


def compare_get_global_values(i, provide_shape):
    """
    Compares the results of get global values for a vector of shape ``(comm.size*2,
    i)``.

    :param int i: Dimension of the vector of length ``comm.size*2``

    """
    if comm.rank == 0:
        if i == 0:
            original_array = np.array(np.random.random((comm.size * 2, )))
        else:
            original_array = np.array(np.random.random((comm.size * 2, i)))
    else:
        original_array = None
    original_array = comm.bcast(original_array)
    my_len = original_array.shape[0] // comm.size
    my_index = np.arange(0 + comm.rank * my_len, (comm.rank + 1) * my_len)
    if i == 0:
        my_array = original_array[my_index]
    else:
        my_array = original_array[my_index, :]
    if provide_shape:
        recomposed_array = util.get_global_values(
            my_array, original_array.shape)
    else:
        recomposed_array = util.get_global_values(my_array)
    nptest.assert_array_equal(original_array, recomposed_array)


def test_fix_dimensions_vector():
    """
    Tests :meth:`bet.util.fix_dimensions_vector`
    """
    values = [1, [1], np.arange(5), np.arange(
        5), np.ones((5, 1)), np.ones((5, 5))]
    shapes = [(1,), (1,), (5,), (5,), (5,), (25,)]
    for value, shape in zip(values, shapes):
        vector = util.fix_dimensions_vector(value)
        assert vector.shape == shape


def test_fix_dimensions_vector_2darray():
    """
    Tests :meth:`bet.util.fix_dimensions_vector_2darray`
    """
    values = [1, [1], np.empty((1, 1)), np.arange(
        5), np.arange(5), np.empty((5, 1))]
    shapes = [(1, 1), (1, 1), (1, 1), (5, 1), (5, 1), (5, 1)]
    for value, shape in zip(values, shapes):
        vector = util.fix_dimensions_vector_2darray(value)
        assert vector.shape == shape


def test_fix_dimensions_domain():
    """
    Tests :meth:`bet.util.fix_dimensions_domain`
    """
    values = [np.arange(2), np.empty((2,)), np.empty((2, 1)), np.empty((1, 2)),
              np.empty((5, 2)), np.empty((2, 5))]
    shapes = [(1, 2), (1, 2), (1, 2), (1, 2), (5, 2), (5, 2)]
    for value, shape in zip(values, shapes):
        vector = util.fix_dimensions_domain(value)
        assert vector.shape == shape


def test_fix_dimensions_data_nodim():
    """
    Tests :meth`bet.util.fix_dimensions_domain` when `dim` is not specified
    """
    values = [1, [1], np.arange(2), np.empty((2,)), np.empty((2, 1)), np.empty((1, 2)),
              np.empty((5, 2)), np.empty((2, 5))]
    shapes = [(1, 1), (1, 1), (2, 1), (2, 1), (2, 1), (1, 2), (5, 2), (2, 5)]
    print(len(values), len(shapes))
    for value, shape in zip(values, shapes):
        vector = util.fix_dimensions_data(value)
        print(vector, value)
        print(vector.shape, shape)
        assert vector.shape == shape


def test_fix_dimensions_data_dim():
    """
    Tests :meth`bet.util.fix_dimensions_domain` when `dim` is specified
    """
    values = [1, [1], np.arange(2), np.empty((2,)), np.empty((2, 1)), np.empty((1, 2)),
              np.empty((5, 2)), np.empty((2, 5)), np.empty((5, 2)), np.empty((2, 5))]
    shapes = [(1, 1), (1, 1), (1, 2), (1, 2), (1, 2), (1, 2), (5, 2), (5, 2), (2, 5),
              (2, 5)]
    dims = [1, 1, 2, 2, 2, 2, 2, 2, 5, 5]
    for value, shape, dim in zip(values, shapes, dims):
        vector = util.fix_dimensions_data(value, dim)
        print(vector, value)
        print(vector.shape, shape, dim)
        assert vector.shape == shape
