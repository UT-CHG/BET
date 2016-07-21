# Copyright (C) 2014-2016 The BET Development Team

"""
This module contains functions for choosing optimal sets of QoIs to use in the
stochastic inverse problem.  
"""
import logging
from itertools import combinations
import numpy as np
from scipy import stats
from bet.Comm import comm
import bet.util as util

def calculate_avg_measure(input_set, qoi_set=None, bin_measure=None):
    r"""
    If you are using ``bin_ratio`` to define the hyperrectangle in the output
    space you must give this method gradient vectors normalized with
    respect to the 1-norm.  If you are using ``bin_size`` to define the
    hyperrectangle in the output space you must give this method the original
    gradient vectors. If you also give a ``bin_measure``, this method will
    approximate the measure of the region of non-zero probability in the inverse
    solution.
    Given gradient vectors at some points (centers) in the input space and
    given a specific set of QoIs, calculate the expected measure of the
    inverse image of a box in the data space using local linear approximations
    of the map Q.
    
    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None
    :type input_set: :class:`~bet.sample.sample_set`
    :param list qoi_set: List of QoI indices
    :param float bin_measure: The measure of the output_dim hyperrectangle to
        invert into the input space
    
    :rtype: tuple
    :returns: (avg_measure, singvals) where avg_measure is a float and singvals
        has shape (num_centers, output_dim)
    
    """

    if input_set._jacobians is None:
        raise ValueError("You must have jacobians to use this method.")
    if qoi_set is None:
        G = input_set._jacobians
    else:
        G = input_set._jacobians[:, qoi_set, :]
    if G.shape[1] > G.shape[2]:
        raise ValueError("Measure is not defined for more outputs than inputs.\
            Try adding a qoi_set to evaluate the measure of.")

    # If no measure is given, we consider how this set of QoIs will change the
    # measure of the unit hypercube.
    if bin_measure is None:
        bin_measure = 1.0

    # Calculate the singular values of the matrix formed by the gradient
    # vectors of each QoI map.  This gives a set of singular values for each
    # center.
    singvals = np.linalg.svd(G, compute_uv=False)

    # Find the average product of the singular values over each center, then use
    # this to compute the average measure of the inverse solution.
    avg_prod_singvals = np.mean(np.prod(singvals, axis=1))
    if avg_prod_singvals == 0:
        avg_measure = np.inf
    else:
        avg_measure = bin_measure / avg_prod_singvals

    return avg_measure, singvals

def calculate_avg_skewness(input_set, qoi_set=None):
    r"""
    Given gradient vectors at some points (centers) in the input space and
    given a specific set of QoIs, caculate the average skewness of the arrays
    formed by the gradient vectors of each QoI map at each center.

    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None
    :type input_set: :class:`~bet.sample.sample_set`
    :param list qoi_set: List of QoI indices
    :rtype: tuple
    :returns: (hmean_skewG, skewgi) where hmean_skewG is the harmonic mean of
        skewness at each center in the input space (float) and skewgi
        has shape (num_centers, output_dim)
    """

    if input_set._jacobians is None:
        raise ValueError("You must have jacobians to use this method.")
    if qoi_set is None:
        G = input_set._jacobians
    else:
        G = input_set._jacobians[:, qoi_set, :]
    if G.shape[1] > G.shape[2]:
        msg = "Skewness is not defined for more outputs than inputs."
        msg += " Try adding a qoi_set to evaluate the skewness of."
        raise ValueError(msg)

    num_centers = G.shape[0]
    output_dim = G.shape[1]

    # Calculate the singular values of the matrix formed by the gradient
    # vectors of each QoI map.  This gives a set of singular values for each
    # center.
    singvals = np.linalg.svd(G, compute_uv=False)

    # The measure of the parallelepipeds defined by the rows of each Jacobian
    muG = np.tile(np.prod(singvals, axis=1), [output_dim, 1]).transpose()

    # Calculate the measure of the parallelepipeds defined by the rows of each
    # Jacobian if we remove the i'th row.
    muGi = np.zeros([num_centers, output_dim])
    for i in xrange(G.shape[1]):
        muGi[:, i] = np.prod(np.linalg.svd(np.delete(G, i, axis=1),
            compute_uv=False), axis=1)

    # Find the norm of each gradient vector
    normgi = np.linalg.norm(G, axis=2)

    # Find the norm of the new vector, giperp, that is perpendicular to the span
    # of the other vectors and defines a parallelepiped of the same measure.
    normgiperp = muG / muGi

    # We now calculate the local skewness
    skewgi = np.zeros([num_centers, output_dim])

    # The local skewness is calculated for nonzero giperp
    skewgi[normgiperp != 0] = normgi[normgiperp != 0] / \
            normgiperp[normgiperp != 0]

    # If giperp is the zero vector, it is not GD from the rest of the gradient
    # vectors, so the skewness is infinity.
    skewgi[normgiperp == 0] = np.inf

    # If the norm of giperp is infinity, then the rest of the vector were not GD
    # to begin with, so skewness is infinity.
    skewgi[normgiperp == np.inf] = np.inf

    # The local skewness is the max skewness of each vector relative the rest
    skewG = np.max(skewgi, axis=1)
    skewG[np.isnan(skewG)] = np.inf

    # We may have values equal to infinity, so we consider the harmonic mean.
    hmean_skewG = stats.hmean(skewG)

    return hmean_skewG, skewgi

def calculate_avg_condnum(input_set, qoi_set=None):
    r"""
    Given gradient vectors at some points (centers) in the input space and
    given a specific set of QoIs, caculate the average condition number of the
    matrices formed by the gradient vectors of each QoI map at each center.

    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None
    :type input_set: :class:`~bet.sample.sample_set`
    :param list qoi_set: List of QoI indices
    
    :rtype: tuple
    :returns: (condnum, singvals) where condnum is a float and singvals
        has shape (num_centers, output_dim)
    
    """

    if input_set._jacobians is None:
        raise ValueError("You must have jacobians to use this method.")
    if qoi_set is None:
        G = input_set._jacobians
    else:
        G = input_set._jacobians[:, qoi_set, :]
    if G.shape[1] > G.shape[2]:
        msg = "Condition number is not defined for more outputs than inputs."  
        msg += " Try adding a qoi_set to evaluate the condition number of."
        raise ValueError(msg)

    # Calculate the singular values of the matrix formed by the gradient
    # vectors of each QoI map.  This gives a set of singular values for each
    # center.
    singvals = np.linalg.svd(G, compute_uv=False)
    indz = singvals[:, -1] == 0
    if np.sum(indz) == singvals.shape[0]:
        hmean_condnum = np.inf
    else:
        singvals[indz, 0] = np.inf
        singvals[indz, -1] = 1
        condnums = singvals[:, 0] / singvals[:, -1]
        hmean_condnum = stats.hmean(condnums)

    return hmean_condnum, singvals

def chooseOptQoIs(input_set, qoiIndices=None, num_qois_return=None,
        num_optsets_return=None, inner_prod_tol=1.0, measure=False,
        remove_zeros=True):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    set of QoIs to choose from, and the number of desired QoIs to return, this
    method returns the ``num_optsets_return`` best sets of QoIs with with
    repsect to either the average measure of the matrix formed by the
    gradient vectors of each QoI map, OR the average skewness of the inverse
    image of this set of QoIs, computed as the product of the singular values
    of the same matrix.  This method is brute force, i.e., if the method is
    given 10,000 QoIs and told to return the N best sets of 3, it will check all
    10,000 choose 3 possible sets.  See chooseOptQoIs_large for a less
    computationally expensive approach.
    
    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None
    :type input_set: :class:`~bet.sample.sample_set`
    :param qoiIndices: Set of QoIs to consider.  Default is
        xrange(0, input_set._jacobians.shape[1])
    :type qoiIndices: :class:`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.  Default is input_dim
    :param int num_optsets_return: Number of best sets to return
        Default is 10
    :param boolean measure: If measure is True, use ``calculate_avg_measure``
        to determine optimal QoIs, else use ``calculate_avg_skewness``
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient
    
    :rtype: `np.ndarray` of shape (num_optsets_returned, num_qois_returned + 1)
    :returns: measure_skewness_indices_mat
    
    """

    (measure_skewness_indices_mat, _) = chooseOptQoIs_verbose(input_set,
        qoiIndices, num_qois_return, num_optsets_return, inner_prod_tol,
        measure, remove_zeros)

    return measure_skewness_indices_mat

def chooseOptQoIs_verbose(input_set, qoiIndices=None, num_qois_return=None,
            num_optsets_return=None, inner_prod_tol=1.0, measure=False,
            remove_zeros=True):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    set of QoIs to choose from, and the number of desired QoIs to return, this
    method returns the ``num_optsets_return`` best sets of QoIs with with
    repsect to either the average measure of the matrix formed by the
    gradient vectors of each QoI map, OR the average skewness of the inverse
    image of this set of QoIs, computed as the product of the singular values
    of the same matrix.  This method is brute force, i.e., if the method is
    given 10,000 QoIs and told to return the N best sets of 3, it will check all
    10,000 choose 3 possible sets.  See chooseOptQoIs_large for a less
    computationally expensive approach.
    
    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None
    :type input_set: :class:`~bet.sample.sample_set`
    :param qoiIndices: Set of QoIs to consider.  Default is
        xrange(0, input_set._jacobians.shape[1])
    :type qoiIndices: :class:`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.  Default is input_dim
    :param int num_optsets_return: Number of best sets to return
        Default is 10
    :param boolean measure: If measure is True, use ``calculate_avg_measure``
        to determine optimal QoIs, else use ``calculate_avg_skewness``
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient
    
    :rtype: `np.ndarray` of shape (num_optsets_returned, num_qois_returned + 1)
    :returns: measure_skewness_indices_mat
    
    """

    G = input_set._jacobians
    if G is None:
        raise ValueError("You must have jacobians to use this method.")
    input_dim = input_set._dim
    num_centers = G.shape[0]

    if qoiIndices is None:
        qoiIndices = xrange(0, G.shape[1])
    if num_qois_return is None:
        num_qois_return = input_dim
    if num_optsets_return is None:
        num_optsets_return = 10

    # Remove QoIs that have zero gradients at any of the centers
    qoiIndices = find_unique_vecs(input_set, inner_prod_tol, qoiIndices,
        remove_zeros)

    # Find all posible combinations of QoIs
    if comm.rank == 0:
        qoi_combs = np.array(list(combinations(list(qoiIndices),
                        num_qois_return)))
        logging.info('Possible sets of QoIs : {}'.format(qoi_combs.shape[0]))
        qoi_combs = np.array_split(qoi_combs, comm.size)
    else:
        qoi_combs = None

    # Scatter them throughout the processors
    qoi_combs = comm.scatter(qoi_combs, root=0)

    # For each combination, check the skewness and keep the sets
    # that have the smallest skewness
    measure_skewness_indices_mat = np.zeros([num_optsets_return,
        num_qois_return + 1])
    measure_skewness_indices_mat[:, 0] = np.inf
    optsingvals_tensor = np.zeros([num_centers, num_qois_return,
        num_optsets_return])
    for qoi_set in xrange(len(qoi_combs)):
        if measure == False:
            (current_measskew, singvals) = calculate_avg_skewness(input_set,
                qoi_combs[qoi_set])
        else:
            (current_measskew, singvals) = calculate_avg_measure(input_set,
                qoi_combs[qoi_set])

        if current_measskew < measure_skewness_indices_mat[-1, 0]:
            measure_skewness_indices_mat[-1, :] = np.append(np.array(\
                    [current_measskew]), qoi_combs[qoi_set])
            order = measure_skewness_indices_mat[:, 0].argsort()
            measure_skewness_indices_mat = measure_skewness_indices_mat[order]

            optsingvals_tensor[:, :, -1] = singvals
            optsingvals_tensor = optsingvals_tensor[:, :, order]

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and skewness values from each processor
    measure_skewness_indices_mat = np.array(comm.gather(\
            measure_skewness_indices_mat, root=0))
    optsingvals_tensor = np.array(comm.gather(optsingvals_tensor, root=0))

    # Find the num_optsets_return smallest skewness values from all processors
    if comm.rank == 0:
        measure_skewness_indices_mat = measure_skewness_indices_mat.reshape(\
                num_optsets_return * comm.size, num_qois_return + 1)
        optsingvals_tensor = optsingvals_tensor.reshape(num_centers,
            num_qois_return, num_optsets_return * comm.size)
        order = measure_skewness_indices_mat[:, 0].argsort()

        measure_skewness_indices_mat = measure_skewness_indices_mat[order]
        measure_skewness_indices_mat = measure_skewness_indices_mat[\
                :num_optsets_return, :]

        optsingvals_tensor = optsingvals_tensor[:, :, order]
        optsingvals_tensor = optsingvals_tensor[:, :, :num_optsets_return]

    measure_skewness_indices_mat = comm.bcast(measure_skewness_indices_mat, 
            root=0)
    optsingvals_tensor = comm.bcast(optsingvals_tensor, root=0)

    return (measure_skewness_indices_mat, optsingvals_tensor)

def find_unique_vecs(input_set, inner_prod_tol, qoiIndices=None,
        remove_zeros=True):
    r"""
    Given gradient vectors at each center in the parameter space, sort throught
    them and remove any QoI that has a zero vector at any center, then remove
    one from any pair of QoIs that have an average inner product greater than
    some tolerance, i.e., an average angle between the two vectors smaller than
    some tolerance.
    
    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None
    :type input_set: :class:`~bet.sample.sample_set`
    :param float inner_prod_tol: Maximum acceptable average inner product
        between two QoI maps
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient vector at atleast one point in
        :math:`\Lambda`
    
    :rtype: `np.ndarray` of shape (num_unique_vecs, 1)
    :returns: unique_vecs
    
    """

    if input_set._jacobians is None:
        raise ValueError("You must have jacobians to use this method.")
    input_dim = input_set._dim
    G = input_set._jacobians
    if qoiIndices is None:
        qoiIndices = xrange(0, G.shape[1])

    # Normalize the gradient vectors with respect to the 2-norm so the inner
    # product tells us about the angle between the two vectors.
    norm_G = np.linalg.norm(G, ord=2, axis=2)

    # Remove any QoI that has a zero vector at atleast one of the centers.
    if remove_zeros:
        indz = np.array([])
        for i in xrange(norm_G.shape[1]):
            if np.sum(norm_G[:, i] == 0) > 0:
                indz = np.append(indz, i)
    else:
        indz = []

    # If it is a zero vector (has 0 norm), set norm=1, avoid divide by zero
    norm_G[norm_G == 0] = 1.0

    # Normalize each gradient vector
    G = G/np.tile(norm_G, (input_dim, 1, 1)).transpose(1, 2, 0)

    if comm.rank == 0:
        logging.info('*** find_unique_vecs ***')
        logging.info('num_zerovec : {} of ({}) original QoIs'.\
                format(len(indz), G.shape[1]))
        logging.info('Possible QoIs : {}'.format(len(qoiIndices)-len(indz)))
    qoiIndices = list(set(qoiIndices) - set(indz))

    # Find all num_qois choose 2 pairs of QoIs
    qoi_combs = np.array(list(combinations(list(qoiIndices), 2)))

    # For each pair, check the angle between the vectors and throw out the
    # second QoI if the angle is below some tolerance.  At this point all the
    # vectors are normalized, so the inner product will be between -1 and 1.
    repeat_vec = np.array([])
    for qoi_set in xrange(len(qoi_combs)):
        curr_set = qoi_combs[qoi_set]

        # If neither of the current QoIs are in the repeat_vec, test them
        if curr_set[0] not in repeat_vec and curr_set[1] not in repeat_vec:
            curr_inner_prod = np.sum(G[:, curr_set[0], :] * \
                G[:, curr_set[1], :]) / G.shape[0]

            # If the innerprod>tol, throw out the second QoI
            if np.abs(curr_inner_prod) > inner_prod_tol:
                repeat_vec = np.append(repeat_vec, qoi_combs[qoi_set, -1])

    unique_vecs = np.array(list(set(qoiIndices) - set(repeat_vec)))
    if comm.rank == 0:
        logging.info('Unique QoIs : {}'.format(unique_vecs.shape[0]))

    return unique_vecs

def find_good_sets(input_set, good_sets_prev, unique_indices,
        num_optsets_return, measskew_tol, measure):
    r"""

    .. todo::  Use the idea we only know vectors are with 10% accuracy to guide
        inner_prod tol and skewness_tol.
    
    Given gradient vectors at each center in the parameter space and given
    good sets of size (n - 1), return good sets of size n.  That is, return
    sets of size n that have average measure(skewness) less than some tolerance.
    
    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param good_sets_prev: Good sets of QoIs of size n - 1.
    :type good_sets_prev: :class:`np.ndarray` of size (num_good_sets_prev, n -
        1) 
    :param unique_indices: Unique QoIs to consider.
    :type unique_indices: :class:`np.ndarray` of size (num_unique_qois, 1)
    :param int num_optsets_return: Number of best sets to return
    :param float measskew_tol: Throw out all sets of QoIs with average
        measure(skewness) number greater than this.
    :param boolean measure: If measure is True, use ``calculate_avg_measure``
        to determine optimal QoIs, else use ``calculate_avg_skewness``
    
    :rtype: tuple
    :returns: (good_sets, best_sets, optsingvals_tensor) where good sets has
        size (num_good_sets, n), best sets has size (num_optsets_return,
        n + 1) and optsingvals_tensor has size (num_centers, n, input_dim)
    
    """

    if input_set._jacobians is None:
        raise ValueError("You must have jacobians to use this method.")

    num_centers = input_set._jacobians.shape[0]
    num_qois_return = good_sets_prev.shape[1] + 1
    comm.Barrier()

    # Initialize best sets and set all skewness values large
    best_sets = np.zeros([num_optsets_return, num_qois_return + 1])
    best_sets[:, 0] = np.inf
    good_sets = np.zeros([1, num_qois_return])
    count_qois = 0
    optsingvals_tensor = np.zeros([num_centers, num_qois_return,
        num_optsets_return])

    # For each good set of size (n - 1), find the possible sets of size n and
    # compute the average skewness of each
    count_qois = 0
    for i in xrange(good_sets_prev.shape[0]):
        min_ind = np.max(good_sets_prev[i, :])
        # Find all possible combinations of QoIs that include this set of
        # (n - 1)
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

        # For each combination, compute the average measure(skewness) and add
        # the set to good_sets if it is less than measskew_tol
        for qoi_set in xrange(len(qoi_combs)):
            count_qois += 1
            curr_set = util.fix_dimensions_vector_2darray(qoi_combs[qoi_set])\
                .transpose()
            if measure is False:
                (current_measskew, singvals) = calculate_avg_skewness(input_set,
                    qoi_combs[qoi_set])
            else:
                (current_measskew, singvals) = calculate_avg_measure(input_set,
                    qoi_combs[qoi_set])

            # If its a good set, add it to good_sets
            if current_measskew < measskew_tol:
                good_sets = np.append(good_sets, curr_set, axis=0)

                # If the average skewness is less than the maxskewness
                # in our best_sets, add it to best_sets
                if current_measskew < best_sets[-1, 0]:
                    best_sets[-1, :] = np.append(np.array([current_measskew]),
                        qoi_combs[qoi_set])
                    order = best_sets[:, 0].argsort()
                    best_sets = best_sets[order]

                    # Store the corresponding singular values
                    optsingvals_tensor[:, :, -1] = singvals
                    optsingvals_tensor = optsingvals_tensor[:, :, order]

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and skewness values from each processor
    good_sets = comm.gather(good_sets, root=0)
    best_sets = np.array(comm.gather(best_sets, root=0))
    count_qois = np.array(comm.gather(count_qois, root=0))

    # Find the num_optsets_return smallest skewness from all processors
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

        logging.info('Possible sets of QoIs of size {} : {}'.format(\
                good_sets.shape[1], np.sum(count_qois)))
        logging.info('Good sets of QoIs of size {} : {}'.format(\
                good_sets.shape[1], good_sets.shape[0] - 1))

    comm.Barrier()
    best_sets = comm.bcast(best_sets, root=0)
    good_sets = comm.bcast(good_sets, root=0)

    return (good_sets[1:].astype(int), best_sets, optsingvals_tensor)

def chooseOptQoIs_large(input_set, qoiIndices=None, max_qois_return=None,
        num_optsets_return=None, inner_prod_tol=None, measskew_tol=None,
        measure=False, remove_zeros=True):
    r"""
    Given gradient vectors at some points (centers) in the input space, a large
    set of QoIs to choose from, and the number of desired QoIs to return, this
    method returns the set of optimal QoIs of size 2, 3, ...  max_qois_return
    to use in the inverse problem by choosing the sets with the smallest
    average measure(skewness).
    
    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param qoiIndices: Set of QoIs to consider from input_set._jacobians.
        Default is xrange(0, input_set._jacobians.shape[1])
    :type qoiIndices: :class:`np.ndarray` of size (1, num QoIs to consider)
    :param int max_qois_return: Maximum number of desired QoIs to use in the
        inverse problem.  Default is input_dim
    :param int num_optsets_return: Number of best sets to return
        Default is 10
    :param float inner_prod_tol: Maximum acceptable average inner product
        between two QoI maps.
    :param float measskew_tol: Throw out all sets of QoIs with average
        measure(skewness) number greater than this.
    :param boolean measure: If measure is True, use ``calculate_avg_measure``
        to determine optimal QoIs, else use ``calculate_avg_skewness``
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient vector at atleast one point in
        :math:`\Lambda`.
    
    :rtype: tuple
    :returns: (measure_skewness_indices_mat, optsingvals) where
        measure_skewness_indices_mat has shape (num_optsets_return,
        num_qois_return+1) and optsingvals has shape (num_centers,
        num_qois_return, num_optsets_return)
    
    """
    (best_sets, _) = chooseOptQoIs_large_verbose(input_set, qoiIndices,
        max_qois_return, num_optsets_return, inner_prod_tol, measskew_tol, measure,
        remove_zeros)

    return best_sets

def chooseOptQoIs_large_verbose(input_set, qoiIndices=None,
        max_qois_return=None, num_optsets_return=None, inner_prod_tol=None,
        measskew_tol=None, measure=False, remove_zeros=True):
    r"""
    Given gradient vectors at some points (centers) in the parameter space, a
    large set of QoIs to choose from, and the number of desired QoIs to return,
    this method return the set of optimal QoIs of size 1, 2, ... max_qois_return
    to use in the inverse problem by choosing the set with smallext average
    skewness.  Also a tensor that represents the singular values of the
    matrices formed by the gradient vectors of the optimal QoIs at each center
    is returned.
    
    :param input_set: The input sample set.  Make sure the attribute _jacobians
        is not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param qoiIndices: Set of QoIs to consider from G.  Default is
        xrange(0, G.shape[1]).
    :type qoiIndices: :class:`np.ndarray` of size (1, num QoIs to consider)
    :param int max_qois_return: Maximum number of desired QoIs to use in the
        inverse problem.  Default is input_dim.
    :param int num_optsets_return: Number of best sets to return.  Default is
        10.  
    :param float inner_prod_tol: Throw out one vectors from each pair of
        QoIs that has average inner product greater than this.  Default is 0.9.
    :param float measskew_tol: Throw out all sets of QoIs with average
        measure(skewness) number greater than this.  Default is max_float.
    :param boolean measure: If measure is True, use ``calculate_avg_measure``
        to determine optimal QoIs, else use ``calculate_avg_skewness``
    :param boolean remove_zeros: If True, ``find_unique_vecs`` will remove any
        QoIs that have a zero gradient vector at atleast one point in
        :math:`\Lambda`.
    
    :rtype: tuple
    :returns: (measure_skewness_indices_mat, optsingvals) where
        measure_skewness_indices_mat has shape (num_optsets_return,
        num_qois_return+1) and optsingvals is a list where each element has
        shape (num_centers, num_qois_return, num_optsets_return).
        num_qois_return will change for each element of the list.
    
    """
    input_dim = input_set._dim
    if input_set._jacobians is None:
        raise ValueError("You must have jacobians to use this method.")
    if qoiIndices is None:
        qoiIndices = xrange(0, input_set._jacobians.shape[1])
    if max_qois_return is None:
        max_qois_return = input_dim
    if num_optsets_return is None:
        num_optsets_return = 10
    if inner_prod_tol is None:
        inner_prod_tol = 1.0
    if measskew_tol is None:
        measskew_tol = np.inf

    # Find the unique QoIs to consider
    unique_indices = find_unique_vecs(input_set, inner_prod_tol, qoiIndices,
        remove_zeros)
    if comm.rank == 0:
        logging.info('Unique Indices are : {}'.format(unique_indices))

    good_sets_curr = util.fix_dimensions_vector_2darray(unique_indices)
    best_sets = []
    optsingvals_list = []

    # Given good sets of QoIs of size (n - 1), find the good sets of size n
    for qois_return in xrange(2, max_qois_return + 1):
        (good_sets_curr, best_sets_curr, optsingvals_tensor_curr) = \
            find_good_sets(input_set, good_sets_curr, unique_indices,
            num_optsets_return, measskew_tol, measure)
        best_sets.append(best_sets_curr)
        optsingvals_list.append(optsingvals_tensor_curr)
        if comm.rank == 0:
            logging.info(best_sets_curr)

    return (best_sets, optsingvals_list)
