# Copyright (C) 2014-2015  BET Development Team

"""
This module contains functions choosing optimal QoIs to use in
the stochastic inverse problem.
"""

import numpy as np
from itertools import combinations
from bet.Comm import comm


def chooseOptQoIs(Grad_tensor, indexstart, indexstop, num_qois_returned):
    """

    TODO:   This just cares about skewness, not sensitivity  (That is, we pass
            in normalized gradient vectors).  So we want to implement
            sensitivity analysis as well later.
            Check out 'magical min'.

            If a singular value is zero, we let the condition number be 
            1E7 at that point.  Possibly this should be a function of the
            dimension(?) so that we don't exclude a set simply because
            the vectors are parallel at one point in Lambdam, they could
            be much better in other regions.

    Given gradient vectors at some points(xeval) in the parameter space, a set
    of QoIs to choose from, and the number of desired QoIs to return, this
    method return the set of optimal QoIs to use in the inverse problem by
    choosing the set with optimal skewness properties.

    :param Grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type Grad_tensor: :class:`np.ndarray` of shape (num_xeval,num_qois,Ldim)
        where num_xeval is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param int indexstart: Index of the list of QoIs to start at.
    :param int indexstop: Index of the list of QoIs to stop at.
    :param int num_qois_returned: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, qoiIndices)

    """

    num_xeval = Grad_tensor.shape[0]

    # Find all posible combinations of QoIs
    if comm.rank == 0:
        qoi_combs = np.array(list(combinations(range(indexstart, indexstop + 1),
                    num_qois_returned)))
        print 'Possible sets of QoIs : ', qoi_combs.shape[0]
        qoi_combs = np.array_split(qoi_combs, comm.size)
    else:
        qoi_combs = None

    # Scatter them throughout the processors
    qoi_combs = comm.scatter(qoi_combs, root=0)

    # For each combination, check the skewness and keep the set
    # that has the best skewness, i.e., smallest condition number
    min_condnum = float('inf')
    for qoi_set in range(len(qoi_combs)):
        singvals = np.linalg.svd(
            Grad_tensor[:, qoi_combs[qoi_set], :], compute_uv=False)

        # Find zero singular values
        indz = np.where(singvals[:,-1]==0)
        nonz_condnum = np.sum(
            singvals[:, 0] / singvals[:, -1], axis=0) / (num_xeval-len(indz))

        # If we have found zero singular values, set cond=1E7
        current_condnum = nonz_condnum + 1E7

        if current_condnum < min_condnum:
            min_condnum = current_condnum
            qoiIndices = qoi_combs[qoi_set]

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and conditions number from each processor
    min_condnum_indices = comm.gather([min_condnum, qoiIndices], root=0)

    # Find the minimum of the minimums
    if comm.rank == 0:
        min_list = min(min_condnum_indices)
        min_condnum = min_list[0]
        qoiIndices = min_list[1]

    min_condnum = comm.bcast(min_condnum, root=0)
    qoiIndices = comm.bcast(qoiIndices, root=0)

    return (min_condnum, qoiIndices)
