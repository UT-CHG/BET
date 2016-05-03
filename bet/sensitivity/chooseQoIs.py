# Copyright (C) 2014-2015 The BET Development Team

"""
This module contains functions choosing optimal QoIs to use in the stochastic
inverse problem.
"""
import numpy as np
from itertools import combinations
from bet.Comm import comm
import bet.util as util
from scipy import stats


def calculate_avg_skewness(grad_tensor, qoi_set):
    r"""
    Given gradient vectors at some points (centers) in the parameter space and
    given a specific set of QoIs, caculate the average skewness of the matrices
    formed by the gradient vectors of each QoI map at each center.
    :param grad_tensor: Gradient vectors at each center in the parameter space
        :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois,
        Lambda_dim) where num_centers is the number of points in :math:`\Lambda`
        we have approximated the gradient vectors and num_qois is the number of
        QoIs we are given.
    :param list qoi_set: List of QoI indices
    :rtype: tuple
    :returns: (hmean_skewG, skewgi) where hmean_skewG is a float and skewgi
        has shape (num_centers, data_dim)
    """
    # Calculate the singular values of the matrix formed by the gradient
    # vectors of each QoI map.  This gives a set of singular values for each
    # center.
    G = grad_tensor[:, qoi_set, :]
    num_centers = G.shape[0]
    data_dim = G.shape[1]

    singvals = np.linalg.svd(G, compute_uv=False)
    # The measure of the parallelepipeds defined by the rows of of each Jacobian
    muG = np.tile(np.prod(singvals, axis=1), [data_dim, 1]).transpose()

    # Calcualte the measure of the parallelepipeds defined by the rows of each
    # Jacobian if we remove the ith row.
    muGi = np.zeros([num_centers, data_dim])
    for i in range(G.shape[1]):
        muGi[:, i] = np.prod(np.linalg.svd(np.delete(G, i, axis=1),
            compute_uv=False), axis=1)

    # Find the norm of each gradient vector
    normgi = np.linalg.norm(G, axis=2)

    # Find the norm of the new vector, giperp, that is perpendicular to the span
    # of the other vectors and defines a parallelepiped of the same measure.
    normgiperp = muG / muGi

    # We now calculate the local skewness
    skewgi = np.zeros([num_centers, data_dim])

    # The local skewness is calculate for nonzero giperp
    skewgi[normgiperp!=0] = normgi[normgiperp!=0] / normgiperp[normgiperp!=0]

    # If giperp is the zero vector, it is not GD from the rest of the gradient
    # vectors, so the skewness is infinity.
    skewgi[normgiperp==0] = np.inf

    # If the norm of giperp is infinity, then the rest of the vector were not GD
    # to begin with, so skewness is infinity.
    skewgi[normgiperp==np.inf] = np.inf

    # The local skewness is the max skewness of each vector relative the rest
    skewG = np.max(skewgi, axis=1)
    skewG[np.isnan(skewG)]=np.inf

    # We have may have values equal to infinity, so we consider the harmonic
    # mean.
    hmean_skewG = stats.hmean(skewG)

    return hmean_skewG, skewgi


def calculate_avg_condnum(grad_tensor, qoi_set):
    r"""
    Given gradient vectors at some points (centers) in the parameter space and
    given a specific set of QoIs, caculate the average condition number of the
    matrices formed by the gradient vectors of each QoI map at each center.

    :param grad_tensor: Gradient vectors at each center in the parameter space
        :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois,
        Lambda_dim) where num_centers is the number of points in :math:`\Lambda`
        we have approximated the gradient vectors and num_qois is the number of
        QoIs we are given.
    :param list qoi_set: List of QoI indices

    :rtype: tuple
    :returns: (condnum, singvals) where condnum is a float and singvals
        has shape (num_centers, Data_dim)

    """
    # Calculate the singular values of the matrix formed by the gradient
    # vectors of each QoI map.  This gives a set of singular values for each
    # center.
    singvals = np.linalg.svd(grad_tensor[:, qoi_set, :], compute_uv=False)
    indz = singvals[:, -1] == 0
    if np.sum(indz) == singvals.shape[0]:
        hmean_condnum = np.inf
    else:
        singvals[indz, 0] = np.inf
        singvals[indz, -1] = 1
        condnums = singvals[:, 0] / singvals[:, -1]
        hmean_condnum = stats.hmean(condnums)

    return hmean_condnum, singvals

def calculate_avg_volume(grad_tensor, qoi_set, bin_volume=None):
    r"""
    If you are using ``bin_ratio`` to define the hyperrectangle in the Data
    space you must must give this method gradient vectors normalized with
    respect to the 1-norm.  If you are using ``bin_size`` to define the
    hyperrectangle in the Data space you must give this method the original
    gradient vectors. If you also give a ``bin_volume``, this method will
    approximate the volume of the region of non-zero probability in the inverse
    solution.
    Given gradient vectors at some points (centers) in the parameter space
    and given a specific set of QoIs, calculate the average volume of the
    inverse image of a box in the data space assuming the mapping is linear near
    each center.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois,
        Lambda_dim) where num_centers is the number of points in :math:`\Lambda`
        we have approximated the gradient vectors and num_qois is the number of
        QoIs we are given.
    :param list qoi_set: List of QoI indices
    :param float bin_volume: The volume of the Data_dim hyperrectangle to
        invert into :math:`\Lambda`

    :rtype: tuple
    :returns: (avg_volume, singvals) where avg_volume is a float and singvals
        has shape (num_centers, Data_dim)

    """
    # If no volume is given, we consider how this set of QoIs we change the
    # volume of the unit hypercube.
    if bin_volume is None:
        bin_volume = 1.0

    # Calculate the singular values of the matrix formed by the gradient
    # vectors of each QoI map.  This gives a set of singular values for each
    # center.
    singvals = np.linalg.svd(grad_tensor[:, qoi_set, :], compute_uv=False)

    # Find the average produt of the singular values over each center, then use
    # this to compute the average volume of the inverse solution.
    avg_prod_singvals = np.mean(np.prod(singvals, axis=1))
    if avg_prod_singvals == 0:
        avg_volume = np.inf
    else:
        avg_volume = bin_volume / avg_prod_singvals

    return avg_volume, singvals

def chooseOptQoIs(grad_tensor, qoiIndices=None, num_qois_return=None,
        num_optsets_return=None, inner_prod_tol=1.0, volume=False,
        remove_zeros=True):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    set of QoIs to choose from, and the number of desired QoIs to return, this
    method returns the ``num_optsets_return`` best sets of QoIs with with
    repsect to either the average condition number of the matrix formed by the
    gradient vectors of each QoI map, or the average volume of the inverse
    problem us this set of QoIs, computed as the product of the singular values
    of the same matrix.  This method is brute force, i.e., if the method is
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
    :param boolean volume: If measure is True, use ``calculate_avg_volume``
        to determine optimal QoIs
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient vector at atleast one point in
        :math:`\Lambda`.

    :rtype: `np.ndarray` of shape (num_optsets_returned, num_qois_returned + 1)
    :returns: condnum_indices_mat

    """
    (condnum_indices_mat, _) = chooseOptQoIs_verbose(grad_tensor,
        qoiIndices, num_qois_return, num_optsets_return, inner_prod_tol, volume,
        remove_zeros)

    return condnum_indices_mat

def chooseOptQoIs_verbose(grad_tensor, qoiIndices=None, num_qois_return=None,
            num_optsets_return=None, inner_prod_tol=1.0, volume=False,
            remove_zeros=True):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    set of QoIs to choose from, and the number of desired QoIs to return, this
    method returns the ``num_optsets_return`` best sets of QoIs with with
    repsect to either the average condition number of the matrix formed by the
    gradient vectors of each QoI map, or the average volume of the inverse
    problem us this set of QoIs, computed as the product of the singular values
    of the same matrix.  This method is brute force, i.e., if the method is
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
    :param boolean volume: If volume is True, use ``calculate_avg_volume``
        to determine optimal QoIs
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient vector at atleast one point in
        :math:`\Lambda`.

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

    qoiIndices = find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices,
        remove_zeros)

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
    condnum_indices_mat[:, 0] = np.inf
    optsingvals_tensor = np.zeros([num_centers, num_qois_return,
        num_optsets_return])
    for qoi_set in range(len(qoi_combs)):
        if volume == False:
            (current_condnum, singvals) = calculate_avg_condnum(grad_tensor,
                qoi_combs[qoi_set])
        else:
            (current_condnum, singvals) = calculate_avg_volume(grad_tensor,
                qoi_combs[qoi_set])

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

def find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices=None,
        remove_zeros=True):
    r"""
    Given gradient vectors at each center in the parameter space, sort throught
    them and remove any QoI that has a zero vector at any center, then remove
    one from any pair of QoIs that have an average inner product greater than
    some tolerance, i.e., an average angle between the two vectors smaller than
    some tolerance.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param float inner_prod_tol: Maximum acceptable average inner product
        between two QoI maps.
    :param qoiIndices: Set of QoIs to consider.
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient vector at atleast one point in
        :math:`\Lambda`.

    :rtype: `np.ndarray` of shape (num_unique_vecs, 1)
    :returns: unique_vecs

    """

    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])

    # Normalize the gradient vectors with respect to the 2-norm so the inner
    # product tells us about the angle between the two vectors.
    norm_grad_tensor = np.linalg.norm(grad_tensor, ord=2, axis=2)

    # Remove any QoI that has a zero vector at atleast one of the centers.
    if remove_zeros:
        indz = np.array([])
        for i in range(norm_grad_tensor.shape[1]):
            if np.sum(norm_grad_tensor[:, i] == 0) > 0:
                indz = np.append(indz, i)
    else:
        indz = []

    # If it is a zero vector (has 0 norm), set norm=1, avoid divide by zero
    norm_grad_tensor[norm_grad_tensor == 0] = 1.0

    # Normalize each gradient vector
    grad_tensor = grad_tensor/np.tile(norm_grad_tensor, (Lambda_dim, 1,
        1)).transpose(1, 2, 0)

    if comm.rank == 0:
        print '*** find_unique_vecs ***'
        print 'num_zerovec : ', len(indz), 'of (', grad_tensor.shape[1],\
            ') original QoIs'
        print 'Possible QoIs : ', len(qoiIndices) - len(indz)
    qoiIndices = list(set(qoiIndices) - set(indz))

    # Find all num_qois choose 2 pairs of QoIs
    qoi_combs = np.array(list(combinations(list(qoiIndices), 2)))

    # For each pair, check the angle between the vectors and throw out the
    # second QoI if the angle is below some tolerance.  At this point all the
    # vectors are normalized, so the inner product will be between -1 and 1.
    repeat_vec = np.array([])
    for qoi_set in range(len(qoi_combs)):
        curr_set = qoi_combs[qoi_set]

        # If neither of the current QoIs are in the repeat_vec, test them
        if curr_set[0] not in repeat_vec and curr_set[1] not in repeat_vec:
            curr_inner_prod = np.sum(grad_tensor[:, curr_set[0], :] * \
                grad_tensor[:, curr_set[1], :]) / grad_tensor.shape[0]

            # If the innerprod>tol, throw out the second QoI
            if np.abs(curr_inner_prod) > inner_prod_tol:
                repeat_vec = np.append(repeat_vec, qoi_combs[qoi_set, -1])

    unique_vecs = np.array(list(set(qoiIndices) - set(repeat_vec)))
    if comm.rank == 0:
        print 'Unique QoIs : ', unique_vecs.shape[0]

    return unique_vecs

def find_good_sets(grad_tensor, good_sets_prev, unique_indices,
        num_optsets_return, cond_tol, volume):
    r"""

    .. todo::  Use the idea we only know vectors are with 10% accuracy to guide
        inner_prod tol and condnum_tol.

    Given gradient vectors at each center in the parameter space and given
    good sets of size n - 1, return good sets of size n.  That is, return
    sets of size n that have average condition number less than some tolerance.

    :param grad_tensor: Gradient vectors at each centers in the parameter
        space :math:`\Lambda` for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param good_sets_prev: Good sets of QoIs of size n - 1.
    :type good_sets_prev: :class:`np.ndarray` of size (num_good_sets_prev, n -
        1)
    :param unique_indices: Unique QoIs to consider.
    :type unique_indices: :class:`np.ndarray` of size (num_unique_qois, 1)
    :param int num_optsets_return: Number of best sets to return
    :param float cond_tol: Throw out all sets of QoIs with average condition
        number greater than this.
    :param boolean volume: If volume is True, use ``calculate_avg_volume``
        to determine optimal QoIs

    :rtype: tuple
    :returns: (good_sets, best_sets, optsingvals_tensor) where good sets has
        size (num_good_sets, n), best sets has size (num_optsets_return,
        n + 1) and optsingvals_tensor has size (num_centers, n, Lambda_dim)

    """
    num_centers = grad_tensor.shape[0]
    num_qois_return = good_sets_prev.shape[1] + 1
    comm.Barrier()

    # Initialize best sets and set all condition numbers large
    best_sets = np.zeros([num_optsets_return, num_qois_return + 1])
    best_sets[:, 0] = np.inf
    good_sets = np.zeros([1, num_qois_return])
    count_qois = 0
    optsingvals_tensor = np.zeros([num_centers, num_qois_return,
        num_optsets_return])

    # For each good set of size n - 1, find the possible sets of size n and
    # compute the average condition number of each
    count_qois = 0
    for i in range(good_sets_prev.shape[0]):
        min_ind = np.max(good_sets_prev[i, :])
        # Find all possible combinations of QoIs that include this set of n - 1
        if comm.rank == 0:
            inds_notin_set = util.fix_dimensions_vector_2darray(list(set(\
                unique_indices) - set(good_sets_prev[i, :])))

            # Choose only the QoI indices > min_ind so we do not repeat sets
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

        # For each combination, compute the average condition number and add the
        # set to good_sets if it is less than cond_tol
        for qoi_set in range(len(qoi_combs)):
            count_qois += 1
            curr_set = util.fix_dimensions_vector_2darray(qoi_combs[qoi_set])\
                .transpose()
            if volume == False:
                (current_condnum, singvals) = calculate_avg_skewness(grad_tensor,
                    qoi_combs[qoi_set])
            else:
                (current_condnum, singvals) = calculate_avg_volume(grad_tensor,
                    qoi_combs[qoi_set])

            # If its a good set, add it to good_sets
            if current_condnum < cond_tol:
                good_sets = np.append(good_sets, curr_set, axis=0)

                # If the average condition number is less than the max condition
                # number in our best_sets, add it to best_sets
                if current_condnum < best_sets[-1, 0]:
                    best_sets[-1, :] = np.append(np.array([current_condnum]),
                        qoi_combs[qoi_set])
                    order = best_sets[:, 0].argsort()
                    best_sets = best_sets[order]

                    # Store the corresponding singular values
                    optsingvals_tensor[:, :, -1] = singvals
                    optsingvals_tensor = optsingvals_tensor[:, :, order]

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    good_sets = comm.gather(good_sets, root=0)
    best_sets = np.array(comm.gather(best_sets, root=0))
    count_qois = np.array(comm.gather(count_qois, root=0))

    # Find the num_optsets_return smallest condition numbers from all processors
    if comm.rank == 0:

        # Organize the best sets
        best_sets = best_sets.reshape(num_optsets_return * \
            comm.size, num_qois_return + 1)
        [_, uniq_inds_best] = np.unique(best_sets[:, 0], return_index=True)
        best_sets = best_sets[uniq_inds_best, :]
        best_sets = best_sets[best_sets[:, 0].argsort()]
        best_sets = best_sets[:num_optsets_return, :]

        # Organize the good sets
        good_sets_new = np.zeros([1, num_qois_return])
        for each in good_sets:
            good_sets_new = np.append(good_sets_new, each[1:], axis=0)
        good_sets = good_sets_new

        print 'Possible sets of QoIs of size %i : '%good_sets.shape[1],\
            np.sum(count_qois)
        print 'Good sets of QoIs of size %i : '%good_sets.shape[1],\
            good_sets.shape[0] - 1

    comm.Barrier()
    best_sets = comm.bcast(best_sets, root=0)
    good_sets = comm.bcast(good_sets, root=0)

    return (good_sets[1:].astype(int), best_sets, optsingvals_tensor)

def chooseOptQoIs_large(grad_tensor, qoiIndices=None, max_qois_return=None,
        num_optsets_return=None, inner_prod_tol=None, cond_tol=None,
        volume=False, remove_zeros=True):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    large set of QoIs to choose from, and the number of desired QoIs to return,
    this method return the set of optimal QoIs of size 2, 3, ... max_qois_return
    to use in the inverse problem by choosing the sets with the smallext average
    condition number or volume.

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
    :param float inner_prod_tol: Maximum acceptable average inner product
        between two QoI maps.
    :param float cond_tol: Throw out all sets of QoIs with average condition
        number greater than this.
    :param boolean volume: If volume is True, use ``calculate_avg_volume``
        to determine optimal QoIs
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient vector at atleast one point in
        :math:`\Lambda`.

    :rtype: tuple
    :returns: (condnum_indices_mat, optsingvals) where condnum_indices_mat has
        shape (num_optsets_return, num_qois_return+1) and optsingvals
        has shape (num_centers, num_qois_return, num_optsets_return)

    """
    (best_sets, _) = chooseOptQoIs_large_verbose(grad_tensor, qoiIndices,
        max_qois_return, num_optsets_return, inner_prod_tol, cond_tol, volume,
        remove_zeros)

    return best_sets

def chooseOptQoIs_large_verbose(grad_tensor, qoiIndices=None,
        max_qois_return=None, num_optsets_return=None, inner_prod_tol=None,
        cond_tol=None, volume=False, remove_zeros=True):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    large set of QoIs to choose from, and the number of desired QoIs to return,
    this method return the set of optimal QoIs of size 1, 2, ... max_qois_return
    to use in the inverse problem by choosing the set with smallext average
    condition number.  Also a tensor that represents the singular values of the
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
    :param int num_optsets_return: Number of best sets to return.  Default is
        10.
    :param float inner_prod_tol: Throw out one vectors from each pair of
        QoIs that has average inner product greater than this.  Default is 0.9.
    :param float cond_tol: Throw out all sets of QoIs with average condition
        number greater than this.  Default is max_float.
    :param boolean volume: If volume is True, use ``calculate_avg_volume``
        to determine optimal QoIs
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient vector at atleast one point in
        :math:`\Lambda`.

    :rtype: tuple
    :returns: (condnum_indices_mat, optsingvals) where condnum_indices_mat has
        shape (num_optsets_return, num_qois_return+1) and optsingvals is a list
        where each element has shape (num_centers, num_qois_return,
        num_optsets_return).  num_qois_return will change for each element of
        the list.

    """
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])
    if max_qois_return is None:
        max_qois_return = Lambda_dim
    if num_optsets_return is None:
        num_optsets_return = 10
    if inner_prod_tol is None:
        inner_prod_tol = 1.0
    if cond_tol is None:
        cond_tol = np.inf

    # Find the unique QoIs to consider
    unique_indices = find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices,
        remove_zeros)
    if comm.rank == 0:
        print 'Unique Indices are : ', unique_indices

    good_sets_curr = util.fix_dimensions_vector_2darray(unique_indices)
    best_sets = []
    optsingvals_list = []

    # Given good sets of QoIs of size n - 1, find the good sets of size n
    for qois_return in range(2, max_qois_return + 1):
        (good_sets_curr, best_sets_curr, optsingvals_tensor_curr) = \
            find_good_sets(grad_tensor, good_sets_curr, unique_indices,
            num_optsets_return, cond_tol, volume)
        best_sets.append(best_sets_curr)
        optsingvals_list.append(optsingvals_tensor_curr)
        if comm.rank == 0:
            print best_sets_curr

    return (best_sets, optsingvals_list)
