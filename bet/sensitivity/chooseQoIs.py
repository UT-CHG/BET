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
    """

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

    singvals = np.linalg.svd(grad_tensor[:, qoi_set, :], compute_uv=False)

    # Find the centers that have atleast one zero sinular value
    indz = singvals[:, -1] == 0
    indnz = singvals[:, -1] != 0

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
    10,000 choose 3 possible sets.  This can be expensive, methods currently
    being developed will take a more careful approach and reduce computational
    cost.

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
            num_optsets_return=None):
    r"""
    Given gradient vectors at some points(centers) in the parameter space, a set
    of QoIs to choose from, and the number of desired QoIs to return, this
    method return the set of optimal QoIs to use in the inverse problem by
    choosing the set with optimal skewness properties.  Also a tensor that
    represents the singualre values of the matrices formed by the gradient
    vectors of the optimal QoIs at each center is returned.

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

########################################
########################################
########################################
# WORK IN PROGRESS BELOW!

def find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices=None):
    """
    # DO NOT USE, WORK IN PROGRESS!!

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, optqoiIndices, optsingvals)

    """

    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])

    # Remove and QoI that has a zero vector at atleast one of the centers
    # Up for discussion if this is the best move.  Possible the QoI has
    # a zero vector at one centers, and is perfectly orthogonal to another
    # QoI at all the other centers.  For now we simply remove, since during
    # the condition number calculation it would get a high cond_num any way.
    normG = np.linalg.norm(grad_tensor, axis=2)
    indz = np.array([])
    for i in range(normG.shape[1]):
        if np.sum(normG[:,i]==0) > 0:
            indz = np.append(indz, i)
    qoiIndices = list(set(qoiIndices) - set(indz))

    # Find all n choose 2 pairs of QoIs
    qoi_combs = np.array(list(combinations(list(qoiIndices), 2)))
    if comm.rank==0:
        print '*** find_unique_vecs ***'
        print 'num_zerovec : ', len(indz)
        print 'Possible pairs of QoIs : ', qoi_combs.shape

    # For each pair, check the angle between the vectors and throw out the
    # second if the angle is below some tolerance.  For pairs of vectors we
    # want to consider using the idea that we only know the vectors to within
    # 10% error.
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
    if comm.rank==0:
        print 'Unique QoIs : ', unique_vecs.shape[0]

    return unique_vecs

def find_good_triplets(grad_tensor, cond_tol, qoiIndices=None):
    """
    # DO NOT USE, WORK IN PROGRESS!!

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
        approximated the gradient vectors, num_qois is the total number of
        possible QoIs to choose from, Ldim is the dimension of :math:`\Lambda`.
    :param qoiIndices: Set of QoIs to consider
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.

    :rtype: tuple
    :returns: (min_condum, optqoiIndices, optsingvals)

    """
    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])

    # Find all n choose 2 pairs of QoIs
    if comm.rank == 0:
        qoi_combs = np.array(list(combinations(list(qoiIndices), 3)))
        print '*** find_good_triplets ***'
        print 'Possible triplets of QoIs : ', qoi_combs.shape
        qoi_combs = np.array_split(qoi_combs, comm.size)
    else:
        qoi_combs = None

    # Scatter them throughout the processors
    qoi_combs = comm.scatter(qoi_combs, root=0)

    # For each combination, check the skewness and throw out one from each pair
    # that has global skewess>cond_tol.  For pairs of vectors we want to
    # consider using the idea that we only know the vectors to within 10% error.
    good_mat = np.zeros([1, 3])

    for qoi_set in range(len(qoi_combs)):
        curr_set = util.fix_dimensions_vector_2darray(qoi_combs[qoi_set]).transpose()
        (current_condnum, _) = calculate_avg_condnum(grad_tensor, qoi_combs[qoi_set])

        if current_condnum < cond_tol:
            good_mat = np.append(good_mat, curr_set, axis=0)

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    good_mat = np.array(comm.gather(good_mat, root=0))

    # Find the minimum of the minimums
    if comm.rank == 0:
        good_mat_new = np.zeros([1,3])
        for each in good_mat:
            good_mat_new = np.append(good_mat_new, each[1:], axis=0)
            
        good_mat = good_mat_new
        print 'Good triplets of QoIs : ', good_mat.shape[0]-1

    good_mat = comm.bcast(good_mat, root=0)

    return good_mat[1:]

def find_good_quartets(grad_tensor, qoiIndices=None, num_optsets_return=None, inner_prod_tol=None, cond_tol=None, unique_inds=None):
    """

    :param grad_tensor: Gradient vectors at each centers in the parameter 
        space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
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
        has shape (num_centers, num_qois_return, num_optsets_return)

    """
    num_qois_return = 4
    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])
    if num_optsets_return is None:
        num_optsets_return = 10
    if inner_prod_tol is None:
        inner_prod_tol = 0.8
    if cond_tol is None:
        cond_tol = 2.0

    if unique_inds is None:
        unique_inds = find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices)

    comm.Barrier()
    good_triplets = find_good_triplets(grad_tensor, cond_tol=cond_tol, qoiIndices=unique_inds)
    comm.Barrier()

    condnum_indices_mat = np.zeros([num_optsets_return, num_qois_return + 1])
    condnum_indices_mat[:,0] = 1E11
    for i in range(good_triplets.shape[0]):
        # Find all posible combinations of QoIs
        if comm.rank == 0:
            l = util.fix_dimensions_vector_2darray(list(set(unique_inds) - \
                set(good_triplets[i,:])))
            qoi_combs = np.append(np.tile(good_triplets[i,:],
                [len(unique_inds) - 3, 1]), l, axis=1)
            if i==0:
                print 'Possible quartets of QoIs : ', qoi_combs.shape[0] * \
                    good_triplets.shape[0]
            qoi_combs = np.array_split(qoi_combs, comm.size)
        else:
            qoi_combs = None

        # Scatter them throughout the processors
        qoi_combs = comm.scatter(qoi_combs, root=0)

        for qoi_set in range(len(qoi_combs)):
            lq = list(qoi_combs)
            (current_condnum, _) = calculate_avg_condnum(grad_tensor, list(lq[qoi_set]))

            if current_condnum < condnum_indices_mat[-1, 0]:
                repeat_set = False
                for j in range(condnum_indices_mat.shape[0]):
                    if len(set(lq[qoi_set]) - set(condnum_indices_mat[j, 1:])) == 0:
                        repeat_set = True
                if not repeat_set:
                    condnum_indices_mat[-1, :] = np.append(np.array([current_condnum]),
                        qoi_combs[qoi_set])
                    condnum_indices_mat = condnum_indices_mat[condnum_indices_mat[:, 
                        0].argsort()]
                    #optsingvals_tensor = singvals

        # Wait for all processes to get to this point
        comm.Barrier()

        # Gather the best sets and condition numbers from each processor
        condnum_indices_mat = np.array(comm.gather(condnum_indices_mat, root=0))

        # Find the num_optsets_return smallest condition numbers from all processors
        if comm.rank == 0:
            condnum_indices_mat = condnum_indices_mat.reshape(num_optsets_return * \
                comm.size, num_qois_return + 1)

            [temp, uniq_inds] =  np.unique(condnum_indices_mat[:, 0], return_index=True)
            condnum_indices_mat = condnum_indices_mat[uniq_inds, :]
            condnum_indices_mat = condnum_indices_mat[condnum_indices_mat[:, 
                0].argsort()]
            condnum_indices_mat = condnum_indices_mat[:num_optsets_return, :]

            temp = np.zeros([num_optsets_return, num_qois_return + 1])
            temp[:,0] = 1E11

            temp[:condnum_indices_mat.shape[0], :] = condnum_indices_mat
            condnum_indices_mat = temp



        condnum_indices_mat = comm.bcast(condnum_indices_mat, root=0)
        comm.Barrier()

    return (unique_inds, condnum_indices_mat)

def find_good_sets(grad_tensor, good_sets_prev, qoiIndices=None, num_optsets_return=None, inner_prod_tol=None, cond_tol=None, unique_inds=None):
    """

    :param grad_tensor: Gradient vectors at each centers in the parameter 
        space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
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
        has shape (num_centers, num_qois_return, num_optsets_return)

    """
    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    num_qois_return = good_sets_prev.shape[1] + 1
    if qoiIndices is None:
        qoiIndices = range(0, grad_tensor.shape[1])
    if num_optsets_return is None:
        num_optsets_return = 10
    if inner_prod_tol is None:
        inner_prod_tol = 0.99
    if cond_tol is None:
        cond_tol = 20000000.0 #sys.float_info[0]

    if unique_inds is None:
        unique_inds = find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices)

    comm.Barrier()

    best_sets = np.zeros([num_optsets_return, num_qois_return + 1])
    best_sets[:,0] = 1E11
    good_sets = np.zeros([1, num_qois_return])
    for i in range(good_sets_prev.shape[0]):
        # Find all posible combinations of QoIs
        if comm.rank == 0:
            l = util.fix_dimensions_vector_2darray(list(set(unique_inds) - \
                set(good_sets_prev[i,:])))
            qoi_combs = np.append(np.tile(good_sets_prev[i,:],
                [len(unique_inds) - good_sets_prev.shape[1], 1]), l, axis=1)
            if i==0:
                print 'Possible sets of QoIs of size() : ', qoi_combs.shape[0] * \
                    good_sets_prev.shape[0]
            qoi_combs = np.array_split(qoi_combs, comm.size)
        else:
            qoi_combs = None

        # Scatter them throughout the processors
        qoi_combs = comm.scatter(qoi_combs, root=0)

        # For each combination, check the skewness and throw out one from each pair
        # that has global skewess>cond_tol.  For pairs of vectors we want to
        # consider using the idea that we only know the vectors to within 10% error.
        

        for qoi_set in range(len(qoi_combs)):
            curr_set = util.fix_dimensions_vector_2darray(qoi_combs[qoi_set]).transpose()
            (current_condnum, _) = calculate_avg_condnum(grad_tensor, qoi_combs[qoi_set])
            if current_condnum < cond_tol:
                good_sets = np.append(good_sets, curr_set, axis=0)

            if current_condnum < best_sets[-1, 0]:
                repeat_set = False
                for j in range(best_sets.shape[0]):
                    if len(set(qoi_combs[qoi_set]) - set(best_sets[j, 1:])) == 0:
                        repeat_set = True
                if not repeat_set:
                    best_sets[-1, :] = np.append(np.array([current_condnum]),
                        qoi_combs[qoi_set])
                    best_sets = best_sets[best_sets[:, 
                        0].argsort()]

    # Wait for all processes to get to this point
    comm.Barrier()

    # Gather the best sets and condition numbers from each processor
    good_sets = np.array(comm.gather(good_sets, root=0))
    best_sets = np.array(comm.gather(best_sets, root=0))

    # Find the num_optsets_return smallest condition numbers from all processors
    if comm.rank == 0:
        best_sets = best_sets.reshape(num_optsets_return * \
            comm.size, num_qois_return + 1)

        [temp, uniq_inds] =  np.unique(best_sets[:, 0], return_index=True)
        best_sets = best_sets[uniq_inds, :]
        best_sets = best_sets[best_sets[:, 
            0].argsort()]
        best_sets = best_sets[:num_optsets_return, :]

        temp = np.zeros([num_optsets_return, num_qois_return + 1])
        temp[:,0] = 1E11

        temp[:best_sets.shape[0], :] = best_sets
        best_sets = temp

        # good sets
        good_sets_new = np.zeros([1, num_qois_return])
        for each in good_sets:
            good_sets_new = np.append(good_sets_new, each[1:], axis=0)
            
        good_sets = good_sets_new
        print 'Good sets of QoIs of size() : ', good_sets.shape[0] - 1


        comm.Barrier()
    best_sets = comm.bcast(best_sets, root=0)
    good_sets = comm.bcast(good_sets, root=0)

    return (good_sets[1:], best_sets)


def chooseOptQoIs_upto_Lambda_dim(grad_tensor, num_optsets_return=None, inner_prod_tol=None, cond_tol=None):
    """

    :param grad_tensor: Gradient vectors at each centers in the parameter 
        space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers,num_qois,Ldim)
        where num_centers is the number of points in :math:'\Lambda' we have
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
        has shape (num_centers, num_qois_return, num_optsets_return)

    """

    num_centers = grad_tensor.shape[0]
    Lambda_dim = grad_tensor.shape[2]
    qoiIndices = range(0, grad_tensor.shape[1])

    if num_optsets_return is None:
        num_optsets_return = 10
    if inner_prod_tol is None:
        inner_prod_tol = 0.8
    if cond_tol is None:
        cond_tol = 20.0

    unique_inds = find_unique_vecs(grad_tensor, inner_prod_tol, qoiIndices)

    for Ldim in range(3, Lambda_dim):
        good_sets = find_good_sets(grad_tensor, cond_tol=cond_tol, qoiIndices=unique_inds)
        








