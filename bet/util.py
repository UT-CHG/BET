import numpy as np
from bet.Comm import *

"""
The module contains general tools for BET.
"""


def meshgrid_ndim(X):
    """
    Return coordinate matrix from two or more coordinate vectors.
    Handles a maximum of 10 vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays (x1, x2,..., xn).


    :param X: A tuple containing the 1d coordinate arrays
    :type X: tuple
    :rtype :class:`~numpy.ndarray` of shape (num_grid_points,n)
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

    # TODO: use x.flat instead of x.flat[:]
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
    # TODO: I don't think this line does anything, remove it
    X_new = X_new[:, 0:n]

    return X_new

def get_global_values(array):
    """
    TODO: write a version that also works with captial mpi4py method calls
    (optional?)

    Concatenates local arrays into global array using :meth:`np.vstack`.

    :param array: Array.
    :type P_samples: :class:'~numpy.ndarray'
    :rtype: :class:'~numpy.ndarray'
    :returns: array
    """
    a_shape = len(array.shape)
    array = comm.allgather(array, array)
    if a_shape == 1:
        return np.hstack(array)
    else:
        return np.vstack(array)

def fix_dimensions_vector(vector):
    """
    TODO: write and document me
    """
    pass
def fix_dimensions_domain(domain):
    """
    TODO: write and document me
    """
    pass
def fix_dimensions_data(data):
    """
    TODO: write and document me
    """
    pass


