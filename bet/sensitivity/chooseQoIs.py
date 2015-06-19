# Copyright (C) 2014-2015  BET Development Team

"""
This module contains functions choosing optimal QoIs to use in
the stochastic inverse problem.
"""

import numpy as np
from itertools import combinations
from bet.Comm import comm
import bet.util as util

def chooseOptQoIs(grad_tensor, qoiIndices=None, num_qois_return=None,
        num_optsets_return=None):
    """

    Given gradient vectors at some points(xeval) in the parameter space, a set
    of QoIs to choose from, and the number of desired QoIs to return, this
    method return the set of optimal QoIs to use in the inverse problem by
    choosing the set with optimal skewness properties.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_xeval,num_qois,Ldim)
        where num_xeval is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, optqoiIndices)

    """
    (condnum_indices_mat, optsingvals) = chooseOptQoIs_verbose( \
        grad_tensor, qoiIndices, num_qois_return, num_optsets_return)

    return condnum_indices_mat

def chooseOptQoIs_verbose(grad_tensor, qoiIndices=None, num_qois_return=None,
            num_optsets_return=None):
    """

    TODO:   MAKE THIS RETURN SINGULAR VALUES AS WELL!!!  TENSOR SORT ISSUES...

            Allow user demand certain QoIs, or set of QOIs, must be used

            This just cares about skewness, not sensitivity  (That is, we pass
            in normalized gradient vectors).  So we want to implement
            sensitivity analysis as well later.
            Check out 'magical min'.

            If a singular value is zero, we let the condition number be 
            1E9 at that point.  Possibly this should be a function of the
            dimension(?) so that we don't exclude a set simply because
            the vectors are linearly dependent at one point in :math:\Lambda,
            they could be much better in other regions.

    Given gradient vectors at some points(xeval) in the parameter space, a set
    of QoIs to choose from, and the number of desired QoIs to return, this
    method return the set of optimal QoIs to use in the inverse problem by
    choosing the set with optimal skewness properties.

    :param grad_tensor: Gradient vectors at each xeval in the parameter 
        space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_xeval,num_qois,Ldim)
        where num_xeval is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.
    :param int num_optsets_return: Number of best sets to return

    :rtype: tuple
    :returns: (condnum_indices_mat, optsingvals) where condnum_indices_mat has
        shape (num_optsets_return, num_qois_return+1) and optsingvals
        has shape (num_xeval, Lambda_dim, num_optsets_return)

    """
    num_xeval = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])
    if num_qois_return is None:
        num_qois_return = Lambda_dim
    if num_optsets_return is None:
        num_optsets_return = 10

    # Find all posible combinations of QoIs
    if comm.rank == 0:
        qoi_combs = np.array(list(combinations(list(qoiIndices),
                        num_qois_return)))
        print 'Possible sets of QoIs : ', qoi_combs.shape[0]
        qoi_combs = np.array_split(qoi_combs, comm.size)
    else:
        qoi_combs = None

    # Scatter them throughout the processors
    qoi_combs = comm.scatter(qoi_combs, root=0)

    # For each combination, check the skewness and keep the set
    # that has the best skewness, i.e., smallest condition number
    condnum_indices_mat = np.zeros([num_optsets_return, num_qois_return + 1])
    condnum_indices_mat[:,0] = 1E11
    for qoi_set in range(len(qoi_combs)):
        singvals = np.linalg.svd(
            grad_tensor[:, qoi_combs[qoi_set], :], compute_uv=False)

        # Find the xeval that have atleast one zero sinular value
        indz = singvals[:,-1]==0
        indnz = singvals[:,-1]!=0

        current_condnum = (np.sum(singvals[indnz, 0] / singvals[indnz, -1], \
                          axis=0) + 1E9 * np.sum(indz)) / singvals.shape[0]

        if current_condnum < condnum_indices_mat[-1, 0]:
            condnum_indices_mat[-1, :] = np.append(np.array([current_condnum]),
                qoi_combs[qoi_set])
            condnum_indices_mat = condnum_indices_mat[condnum_indices_mat[:, 
                0].argsort()]
            optsingvals_tensor = singvals

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor

    condnum_indices_mat = np.array(comm.gather(condnum_indices_mat, root=0))

    # Find the minimum of the minimums
    if comm.rank == 0:
        condnum_indices_mat = condnum_indices_mat.reshape(num_optsets_return * \
            comm.size, num_qois_return + 1)
        condnum_indices_mat = condnum_indices_mat[condnum_indices_mat[:, 
            0].argsort()]
        condnum_indices_mat = condnum_indices_mat[:num_optsets_return, :]

    condnum_indices_mat = comm.bcast(condnum_indices_mat, root=0)

    return (condnum_indices_mat, optsingvals_tensor)

########################################
########################################
########################################
# WORK IN PROGRESS BELOW!

def find_bad_pairs(grad_tensor, cond_tol, qoiIndices=None):
    """
    # DO NOT USE, WORK IN PROGRESS!!

    I claim this should not (can not?) be done in parallel.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_xeval,num_qois,Ldim)
        where num_xeval is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, optqoiIndices, optsingvals)

    """
    num_xeval = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])

    # Find all n choose 2 pairs of QoIs
    if comm.rank == 0:
        qoi_combs = np.array(list(combinations(list(qoiIndices), 2)))
        print 'Possible pairs of QoIs : ', qoi_combs.shape
        qoi_combs = np.array_split(qoi_combs, comm.size)
    else:
        qoi_combs = None

    # Scatter them throughout the processors
    qoi_combs = comm.scatter(qoi_combs, root=0)

    # For each combination, check the skewness and throw out one from each pair
    # that has global skewess>cond_tol.  For pairs of vectors we want to
    # consider using the idea that we only know the vectors to within 10% error.
    bad_vec = np.array([])
    for qoi_set in range(len(qoi_combs)):
        curr_set = qoi_combs[qoi_set]
        if curr_set[0] not in bad_vec and curr_set[1] not in bad_vec:
            singvals = np.linalg.svd(
                grad_tensor[:, qoi_combs[qoi_set], :], compute_uv=False)

            # Find the xeval that have atleast one zero sinular value
            indz = singvals[:,-1]==0
            indnz = singvals[:,-1]!=0

            # As it is with 1E9, if and singval is zero (for any xeval that is)
            # we throw out that pair  (unless we have BIG num_xeval)
            current_condnum = (np.sum(singvals[indnz, 0] / singvals[indnz, -1],
                axis=0) + 1E9 * np.sum(indz)) / singvals.shape[0]

            if current_condnum > cond_tol:
                bad_vec = np.append(bad_vec, qoi_combs[qoi_set, -1])

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    bad_vec = np.array(comm.gather(bad_vec, root=0))

    # Find the minimum of the minimums
    if comm.rank == 0:
        bad_vec_new = np.array([])
        for each in bad_vec:
            bad_vec_new = np.append(bad_vec_new, each)

        bad_vec = np.unique(bad_vec_new)

    bad_vec = comm.bcast(bad_vec, root=0)

    return bad_vec

def find_bad_triplets(grad_tensor, cond_tol, qoiIndices=None):
    """
    # DO NOT USE, WORK IN PROGRESS!!

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_xeval,num_qois,Ldim)
        where num_xeval is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, optqoiIndices, optsingvals)

    """
    num_xeval = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])

    # Find all n choose 2 pairs of QoIs
    if comm.rank == 0:
        qoi_combs = np.array(list(combinations(list(qoiIndices), 3)))
        print 'Possible triplets of QoIs : ', qoi_combs.shape
        qoi_combs = np.array_split(qoi_combs, comm.size)
    else:
        qoi_combs = None

    # Scatter them throughout the processors
    qoi_combs = comm.scatter(qoi_combs, root=0)

    # For each combination, check the skewness and throw out one from each pair
    # that has global skewess>cond_tol.  For pairs of vectors we want to
    # consider using the idea that we only know the vectors to within 10% error.
    bad_mat = np.zeros([1, 3])

    for qoi_set in range(len(qoi_combs)):
        curr_set = util.fix_dimensions_vector_2darray(qoi_combs[qoi_set]).transpose()
        singvals = np.linalg.svd(
            grad_tensor[:, qoi_combs[qoi_set], :], compute_uv=False)

        # Find the xeval that have atleast one zero sinular value
        indz = singvals[:,-1]==0
        indnz = singvals[:,-1]!=0

        # As it is with 1E9, if and singval is zero (for any xeval that is)
        # we throw out that pair  (unless we have BIG num_xeval)
        current_condnum = (np.sum(singvals[indnz, 0] / singvals[indnz, -1],
            axis=0) + 1E9 * np.sum(indz)) / singvals.shape[0]

        if current_condnum > cond_tol:
            bad_mat = np.append(bad_mat, curr_set, axis=0)

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    bad_mat = np.array(comm.gather(bad_mat, root=0))

    # Find the minimum of the minimums
    if comm.rank == 0:
        bad_mat_new = np.zeros([1,3])
        for each in bad_mat:
            bad_mat_new = np.append(bad_mat_new, each[1:], axis=0)
            
        bad_mat = bad_mat_new

    bad_mat = comm.bcast(bad_mat, root=0)

    return bad_mat[1:]
