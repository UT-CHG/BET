# Copyright (C) 2014-2015 The BET Development Team

"""
This module contains functions choosing optimal QoIs to use in the stochastic
inverse problem.
"""
import numpy as np
from itertools import combinations
from bet.Comm import comm
import bet.util as util
import scipy.io as sio

def chooseOptQoIs(grad_tensor, qoiIndices=None, num_qois_return=None,
        num_optsets_return=None):
    """
    Given gradient vectors at some points (centers) in the parameter space, a
    set of QoIs to choose from, and the number of desired QoIs to return, this
    method returns the num_optsets_return best sets of QoIs with with repsect
    to skewness properties.  This method is brute force, i.e., if the method is
    given 10,000 QoIs and told to return the N best sets of 3, it will check all
    10,000 choose 3 possible sets.  This can be expensive, methods currently
    being developed will take a more careful approach and reduce computational
    cost.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois, 
        Lambda_dim) where num_centers is the number of points in :math:'\Lambda'
        we have approximated the gradient vectors and num_qois is the total
        number of possible QoIs to choose from
    :param qoiIndices: Set of QoIs to consider from grad_tensor
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.
    :param int num_optsets_return: Number of best sets to return

    :rtype: 'np.ndarray' of shape (num_optsets_returned, num_qois_returned + 1)
    :returns: condnum_indices_mat

    """
    (condnum_indices_mat, optsingvals) = chooseOptQoIs_verbose(grad_tensor,
        qoiIndices, num_qois_return, num_optsets_return)

    return condnum_indices_mat

def chooseOptQoIs_verbose(grad_tensor, qoiIndices=None, num_qois_return=None,
            num_optsets_return=None):
    """
    Given gradient vectors at some points(centers) in the parameter space, a set
    of QoIs to choose from, and the number of desired QoIs to return, this
    method return the set of optimal QoIs to use in the inverse problem by
    choosing the set with optimal skewness properties.  Also a tensor that
    represents the singualre values of the matrices formed by the gradient
    vectors of the optimal QoIs at each center is returned.

    :param grad_tensor: Gradient vectors at each point of interest in the
        parameter space :math:'\Lambda' for each QoI map.
    :type grad_tensor: :class:`np.ndarray` of shape (num_centers, num_qois, 
        Lambda_dim) where num_centers is the number of points in :math:'\Lambda'
        we have approximated the gradient vectors and num_qois is the total
        number of possible QoIs to choose from
    :param qoiIndices: Set of QoIs to consider from grad_tensor
    :type qoiIndices: :class:'`np.ndarray` of size (1, num QoIs to consider)
    :param int num_qois_return: Number of desired QoIs to use in the
        inverse problem.
    :param int num_optsets_return: Number of best sets to return

    :rtype: tuple
    :returns: (condnum_indices_mat, optsingvals) where condnum_indices_mat has
        shape (num_optsets_return, num_qois_return+1) and optsingvals
        has shape (num_centers, num_qois_return, num_optsets_return)

    """
    print '*** chooseOptQoIs_verbose ***'
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

    # For each combination, check the skewness and keep the set
    # that has the best skewness, i.e., smallest condition number
    condnum_indices_mat = np.zeros([num_optsets_return, num_qois_return + 1])
    condnum_indices_mat[:,0] = 1E11
    optsingvals_tensor = np.zeros([num_centers, num_qois_return,
        num_optsets_return])
    for qoi_set in range(len(qoi_combs)):
        singvals = np.linalg.svd(
            grad_tensor[:, qoi_combs[qoi_set], :], compute_uv=False)

        # Find the centers that have atleast one zero sinular value
        indz = singvals[:,-1]==0
        indnz = singvals[:,-1]!=0

        current_condnum = (np.sum(singvals[indnz, 0] / singvals[indnz, -1], \
                          axis=0) + 1E9 * np.sum(indz)) / singvals.shape[0]

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
