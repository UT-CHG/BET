# Copyright (C) 2014-2015 The BET Development Team

"""
This module contains functions choosing optimal QoIs to use in the stochastic
inverse problem.
"""
import numpy as np
from itertools import combinations
from bet.Comm import comm
import bet.util as util
import sys

def calculate_avg_condnum(grad_tensor, qoi_set):
    r"""
    Given gradient vectors as some points (centers) in the parameter space
    and given a specific set of QoIs, caculate the average condition number
    of the matrices formed by the gradient vectors of each QoI map at each
    center.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois,
        Lambda_dim) where num_centers is the number of points in :math:`\Lambda`
        we have approximated the gradient vectors and num_qois is the total
        number of possible QoIs to choose from
    :param list qoi_set: List of QoI indices

    :rtype: tuple
    :returns: (condnum, singvals) where condnum is a float and singvals
        has shape (num_centers, Data_dim)

    """
    # Calculate the singular values at each center
    singvals = np.linalg.svd(grad_tensor[:, qoi_set, :], compute_uv=False)

    # Find the centers that have atleast one zero sinular value
    indz = singvals[:, -1] == 0
    indnz = singvals[:, -1] != 0

    # Compute the average condition number
    condnum = (np.sum(singvals[indnz, 0] / singvals[indnz, -1], \
                      axis=0) + 1E9 * np.sum(indz)) / singvals.shape[0]

    return condnum, singvals

def chooseOptQoIs(grad_tensor, qoiIndices=None, num_qois_return=None,
        num_optsets_return=None):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    set of QoIs to choose from, and the number of desired QoIs to return, this
    method returns the ``num_optsets_return`` best sets of QoIs with with repsect
    to skewness properties.  This method is brute force, i.e., if the method is
    given 10,000 QoIs and told to return the N best sets of 3, it will check all
    10,000 choose 3 possible sets.  See chooseOptQoIs_large for a less
    computationally expensive approach.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois,
        Lambda_dim) where num_centers is the number of points in :math:`\Lambda`
        we have approximated the gradient vectors and num_qois is the total
        number of possible QoIs to choose from
    :param qoiIndices: Set of QoIs to consider from grad_tensor.  Default is
        range(0, grad_tensor.shape[1])
    :type qoiIndices: :class:`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.  Default is Lambda_dim
    :param int num_optsets_return: Number of best sets to return
        Default is 10

    :rtype: `np.ndarray` of shape (num_optsets_returned, num_qois_returned + 1)
    :returns: condnum_indices_mat

    """
    (condnum_indices_mat, _) = chooseOptQoIs_verbose(grad_tensor,
        qoiIndices, num_qois_return, num_optsets_return)

    return condnum_indices_mat

def chooseOptQoIs_verbose(grad_tensor, qoiIndices=None, num_qois_return=None,
            num_optsets_return=None, inner_prod_tol=1.0):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    set of QoIs to choose from, and the number of desired QoIs to return, this
    method returns the ``num_optsets_return`` best sets of QoIs with with repsect
    to skewness properties and a tensor that represents the singular values of
    the matrices formed by the gradient vectors of the optimal QoIs at each
    center is returned..  This method is brute force, i.e., if the method is
    given 10,000 QoIs and told to return the N best sets of 3, it will check all
    10,000 choose 3 possible sets.  See chooseOptQoIs_large for a less
    computationally expensive approach.  Also a tensor that represents the
    singular values of the matrices formed by the gradient vectors of the
    optimal QoIs at each center is returned.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois,
        Lambda_dim) where num_centers is the number of points in :math:`\Lambda`
        we have approximated the gradient vectors and num_qois is the total
        number of possible QoIs to choose from
    :param qoiIndices: Set of QoIs to consider from grad_tensor.  Default is
        range(0, grad_tensor.shape[1])
    :type qoiIndices: :class:`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.  Default is Lambda_dim
    :param int num_optsets_return: Number of best sets to return
        Default is 10

    :rtype: tuple
    :returns: (condnum_indices_mat, optsingvals) where condnum_indices_mat has
        shape (num_optsets_return, num_qois_return+1) and optsingvals
        has shape (num_centers, num_qois_return, num_optsets_return)

    """
    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])
    if num_qois_return is None:
        num_qois_return = Lambda_dim
    if num_optsets_return is None:
        num_optsets_return = 10

    qoiIndices = find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices)

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

    # For each combination, check the skewness and keep the sets
    # that have the best skewness, i.e., smallest condition number
    condnum_indices_mat = np.zeros([num_optsets_return, num_qois_return + 1])
    condnum_indices_mat[:, 0] = 1E11
    optsingvals_tensor = np.zeros([num_centers, num_qois_return,
        num_optsets_return])
    for qoi_set in range(len(qoi_combs)):
        (current_condnum, singvals) = calculate_avg_condnum(grad_tensor, qoi_combs[qoi_set])

        if current_condnum < condnum_indices_mat[-1, 0]:
            condnum_indices_mat[-1, :] = np.append(np.array([current_condnum]),
                qoi_combs[qoi_set])
            order = condnum_indices_mat[:, 0].argsort()
            condnum_indices_mat = condnum_indices_mat[order]

            optsingvals_tensor[:, :, -1] = singvals
            optsingvals_tensor = optsingvals_tensor[:, :, order]

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    condnum_indices_mat = np.array(comm.gather(condnum_indices_mat, root=0))
    optsingvals_tensor = np.array(comm.gather(optsingvals_tensor, root=0))

    # Find the num_optsets_return smallest condition numbers from all processors
    if comm.rank == 0:
        condnum_indices_mat = condnum_indices_mat.reshape(num_optsets_return * \
            comm.size, num_qois_return + 1)
        optsingvals_tensor = optsingvals_tensor.reshape(num_centers,
            num_qois_return, num_optsets_return * comm.size)
        order = condnum_indices_mat[:, 0].argsort()

        condnum_indices_mat = condnum_indices_mat[order]
        condnum_indices_mat = condnum_indices_mat[:num_optsets_return, :]

        optsingvals_tensor = optsingvals_tensor[:, :, order]
        optsingvals_tensor = optsingvals_tensor[:, :, :num_optsets_return]

    condnum_indices_mat = comm.bcast(condnum_indices_mat, root=0)
    optsingvals_tensor = comm.bcast(optsingvals_tensor, root=0)

    return (condnum_indices_mat, optsingvals_tensor)


def find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices=None):
    r"""
    Given gradient vectors at each center in the parameter space, sort throught
    them and remove any QoI that has a zero vector at any center, then remove
    one from any pair of QoIs that have an average inner product greater than
    some tolerance.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param float inner_prod_tol: A real number between 0 and 1.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: `np.ndarray` of shape (num_unique_vecs, 1)
    :returns: unique_vecs

    """

    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])

    # Remove any QoI that has a zero vector at atleast one of the centers
    # Up for discussion if this is the best move.  Possible the QoI has
    # a zero vector at one centers, and is perfectly orthogonal to another
    # QoI at all the other centers.  For now we simply remove, since during
    # the condition number calculation it would get a high cond_num any way.
    normG = np.linalg.norm(grad_tensor, axis=2)
    indz = np.array([])
    for i in range(normG.shape[1]):
        if np.sum(normG[:, i] == 0) > 0:
            indz = np.append(indz, i)
    qoiIndices = list(set(qoiIndices) - set(indz))

    # Find all n choose 2 pairs of QoIs
    qoi_combs = np.array(list(combinations(list(qoiIndices), 2)))
    if comm.rank == 0:
        print '*** find_unique_vecs ***'
        print 'num_zerovec : ', len(indz)
        print 'Possible pairs of QoIs : ', qoi_combs.shape

    # For each pair, check the angle between the vectors and throw out the
    # second if the angle is below some tolerance.
    repeat_vec = np.array([])
    for qoi_set in range(len(qoi_combs)):
        curr_set = qoi_combs[qoi_set]
        # If neither of the current QoIs are in the repeat_vec, test them
        if curr_set[0] not in repeat_vec and curr_set[1] not in repeat_vec:
            curr_inner_prod = np.sum(grad_tensor[:, curr_set[0], :] * \
                grad_tensor[:, curr_set[1], :]) / grad_tensor.shape[0]

            if curr_inner_prod > inner_prod_tol:
                repeat_vec = np.append(repeat_vec, qoi_combs[qoi_set, -1])

    unique_vecs = np.array(list(set(qoiIndices) - set(repeat_vec)))
    if comm.rank == 0:
        print 'Unique QoIs : ', unique_vecs.shape[0]

    return unique_vecs

def find_good_sets(grad_tensor, good_sets_prev, unique_indices,
        num_optsets_return, inner_prod_tol, cond_tol):
    r"""
    #TODO:  Use the idea we only know vectors are with 10% accuracy to guide
        inner_prod tol and condnum_tol.

    Given gradient vectors at each center in the parameter space and given
    good sets of size n - 1, return good sets of size n.

    :param grad_tensor: Gradient vectors at each centers in the parameter
        space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param good_sets_prev: Good sets of QoIs of size n - 1.
    :type good_sets_prev: :class:`np.ndarray` of size (num_good_sets_prev, n - 1)
    :param unique_indices: Unique QoIs to consider.
    :type unique_indices: :class:'np.ndarray' of size (num_unique_qois, 1)
    :param int num_optsets_return: Number of best sets to return
    :param float inner_prod_tol: Throw out one vectors from each pair of QoIs
        that has average inner product greater than this.
    :param float cond_tol: Throw out all sets of QoIs with average condition
        number greater than this.

    :rtype: tuple
    :returns: (good_sets, best_sets, optsingvals_tensor) where good sets has
        size (num_good_sets, n), best sets has size (num_optsets_return,
        n + 1) and optsingvals_tensor has size (num_centers, n, Lambda_dim)

    """
    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    num_qois_return = good_sets_prev.shape[1] + 1

    comm.Barrier()

    # Inistialize best sets and set all condition numbers large
    best_sets = np.zeros([num_optsets_return, num_qois_return + 1])
    best_sets[:, 0] = 1E11
    good_sets = np.zeros([1, num_qois_return])
    count_qois = 0
    optsingvals_tensor = np.zeros([num_centers, num_qois_return,
        num_optsets_return])

    # For each good set of size n - 1, find the possible sets of size n and
    # compute the average condition number of each
    for i in range(good_sets_prev.shape[0]):
        min_ind = np.max(good_sets_prev[i, :])
        # Find all possible combinations of QoIs that include this set of n - 1
        if comm.rank == 0:
            inds_notin_set = util.fix_dimensions_vector_2darray(list(set(\
                unique_indices) - set(good_sets_prev[i, :])))
            inds_notin_set = util.fix_dimensions_vector_2darray(inds_notin_set[\
                inds_notin_set > min_ind])
            qoi_combs = util.fix_dimensions_vector_2darray(np.append(np.tile(\
                good_sets_prev[i, :], [inds_notin_set.shape[0], 1]),
                inds_notin_set, axis=1))
            qoi_combs = np.array_split(qoi_combs, comm.size)
        else:
            qoi_combs = None

        # Scatter them throughout the processors
        qoi_combs = comm.scatter(qoi_combs, root=0)

        # For each combination, check the skewness and throw out one from each
        # pair that has average skewess>cond_tol.
        for qoi_set in range(len(qoi_combs)):
            count_qois += 1
            curr_set = util.fix_dimensions_vector_2darray(qoi_combs[qoi_set])\
                .transpose()
            (current_condnum, singvals) = calculate_avg_condnum(grad_tensor,
                qoi_combs[qoi_set])
            if current_condnum < cond_tol:
                good_sets = np.append(good_sets, curr_set, axis=0)

                if current_condnum < best_sets[-1, 0]:
                    best_sets[-1, :] = np.append(np.array([current_condnum]),
                        qoi_combs[qoi_set])
                    order = best_sets[:, 0].argsort()
                    best_sets = best_sets[order]

                    optsingvals_tensor[:, :, -1] = singvals
                    optsingvals_tensor = optsingvals_tensor[:, :, order]

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    good_sets = np.array(comm.gather(good_sets, root=0))
    best_sets = np.array(comm.gather(best_sets, root=0))

    # Find the num_optsets_return smallest condition numbers from all processors
    if comm.rank == 0:
        # Organize the best sets
        best_sets = best_sets.reshape(num_optsets_return * \
            comm.size, num_qois_return + 1)
        [temp, uniq_inds] = np.unique(best_sets[:, 0], return_index=True)
        best_sets = best_sets[uniq_inds, :]
        best_sets = best_sets[best_sets[:, 0].argsort()]
        best_sets = best_sets[:num_optsets_return, :]

        # Organize the good sets
        good_sets_new = np.zeros([1, num_qois_return])
        for each in good_sets:
            good_sets_new = np.append(good_sets_new, each[1:], axis=0)

        good_sets = good_sets_new
        print 'Possible sets of QoIs of size %i : '%count_qois
        print 'Good sets of QoIs of size %i : '%good_sets.shape[1],\
            good_sets.shape[0] - 1

    comm.Barrier()
    best_sets = comm.bcast(best_sets, root=0)
    good_sets = comm.bcast(good_sets, root=0)

    return (good_sets[1:].astype(int), best_sets, optsingvals_tensor)

def chooseOptQoIs_large(grad_tensor, qoiIndices=None, max_qois_return=None,
        num_optsets_return=None, inner_prod_tol=None, cond_tol=None):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    large set of QoIs to choose from, and the number of desired QoIs to return,
    this method return the set of optimal QoIs of size 1, 2, ... max_qois_return
    to use in the inverse problem by choosing the set with optimal skewness
    properties.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois,
        Lambda_dim) where num_centers is the number of points in :math:`\Lambda`
        we have approximated the gradient vectors and num_qois is the total
        number of possible QoIs to choose from
    :param qoiIndices: Set of QoIs to consider from grad_tensor.  Default is
        range(0, grad_tensor.shape[1])
    :type qoiIndices: :class:`np.ndarray` of size (1, num QoIs to consider)
    :param int max_qois_return: Maximum number of desired QoIs to use in the
        inverse problem.  Default is Lambda_dim
    :param int num_optsets_return: Number of best sets to return
        Default is 10

    :rtype: tuple
    :returns: (condnum_indices_mat, optsingvals) where condnum_indices_mat has
        shape (num_optsets_return, num_qois_return+1) and optsingvals
        has shape (num_centers, num_qois_return, num_optsets_return)

    """
    (best_sets, _) = chooseOptQoIs_large_verbose(grad_tensor, qoiIndices,
        max_qois_return, num_optsets_return, inner_prod_tol, cond_tol)

    return best_sets

def chooseOptQoIs_large_verbose(grad_tensor, qoiIndices=None,
        max_qois_return=None, num_optsets_return=None, inner_prod_tol=None,
        cond_tol=None):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    large set of QoIs to choose from, and the number of desired QoIs to return,
    this method return the set of optimal QoIs of size 1, 2, ... max_qois_return
    to use in the inverse problem by choosing the set with optimal skewness
    properties.  Also a tensor that represents the singular values of the
    matrices formed by the gradient vectors of the optimal QoIs at each center
    is returned.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois,
        Lambda_dim) where num_centers is the number of points in :math:`\Lambda`
        we have approximated the gradient vectors and num_qois is the total
        number of possible QoIs to choose from.
    :param qoiIndices: Set of QoIs to consider from grad_tensor.  Default is
        range(0, grad_tensor.shape[1]).
    :type qoiIndices: :class:`np.ndarray` of size (1, num QoIs to consider)
    :param int max_qois_return: Maximum number of desired QoIs to use in the
        inverse problem.  Default is Lambda_dim.
    :param int num_optsets_return: Number of best sets to return.  Default is 10.
    :param float inner_prod_tol: Throw out one vectors from each pair of QoIs
        that has average inner product greater than this.  Default is 0.9.
    :param float cond_tol: Throw out all sets of QoIs with average condition
        number greater than this.  Default is max_float.


    :rtype: tuple
    :returns: (condnum_indices_mat, optsingvals) where condnum_indices_mat has
        shape (num_optsets_return, num_qois_return+1) and optsingvals
        has shape (num_centers, num_qois_return, num_optsets_return)

    """
    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])
    if max_qois_return is None:
        max_qois_return = Lambda_dim
    if num_optsets_return is None:
        num_optsets_return = 10
    if inner_prod_tol is None:
        inner_prod_tol = 0.9
    if cond_tol is None:
        cond_tol = sys.float_info[0]

    unique_indices = find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices)
    good_sets_curr = util.fix_dimensions_vector_2darray(unique_indices)

    best_sets = []
    optsingvals_list = []
    for qois_return in range(2, max_qois_return + 1):
        (good_sets_curr, best_sets_curr, optsingvals_tensor_curr) = \
            find_good_sets(grad_tensor, good_sets_curr, unique_indices,
            num_optsets_return, inner_prod_tol, cond_tol)
        best_sets.append(best_sets_curr)
        optsingvals_list.append(optsingvals_tensor_curr)
        if comm.rank == 0:
            print best_sets_curr

    return (best_sets, optsingvals_list)
