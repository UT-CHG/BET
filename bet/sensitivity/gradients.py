# Copyright (C) 2014-2015 Lindley Graham and Steven Mattis

"""
This module contains functions for approximating gradient vectors
of QoI maps.
"""
import numpy as np
import scipy.spatial as spatial
from bet.Comm import *


def sample_linf_ball(lam_domain, centers, num_close, r):
    """

    Pick num_close points in a the l-infinity box of length
    2*r around a point in :math:`\Lambda`, do this for each point
    in centers.  If this box extends outside of :math:`\Lambda`, we
    sample the intersection.

    :param int num_close: Number of points in each cluster
    :param centers: Points in :math:`\Lambda` to cluster points around
    :type centers: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Each side of the box will have length 2*r
    :rtype: :class:`np.ndarray` of shape (num_close*num_centers, Ldim)
    :returns: Clusters of samples near each point in centers

    """

    Lambda_dim = centers.shape[1]
    num_centers = centers.shape[0]

    # Define the size of each box
    left = np.maximum(
        centers - r, np.ones([num_centers, Lambda_dim]) * lam_domain[:, 0])
    right = np.minimum(
        centers + r, np.ones([num_centers, Lambda_dim]) * lam_domain[:, 1])

    translate = r * np.ones(right.shape)
    indz = np.where(left == 0)
    translate[indz] = right[indz] - r

    translate = np.tile(translate, [num_close, 1])

    # Translate each box accordingly so no samples are outside :math:`\Lambda`
    samples = np.tile(right - left, [num_close, 1]) * np.random.random(
        [num_centers * num_close, Lambda_dim]) + np.tile(centers, [num_close, 1]) - translate

    return np.concatenate([centers, samples])


def sample_l1_ball(centers, num_close, r=None):
    """
    TODO: Come up with better variable names. Vectorize the for loops.
          Split this into two methods, sample_simplex, simlex_to_diamond(?).
          Take in 'hard' and 'soft' lam_domain boundaries and either
          allow for sampes out side lam_domain (soft) or restrict them to
          be inside (hard).

    Uniformly sample the l1-ball (defined by 2^dim simplices).  Then scale
    each dimension according to rvec and translate the center to centers.
    Do this for each point in centers.  *** This method currently allows
    samples to be placed outside of lam_domain.  Please place your
    centers accordingly.***

    :param centers: Points in :math:`\Lambda` to cluster samples around
    :type centers: :class:`np.ndarray` of shape (num_centers, Ldim)
    :param int num_close: Number of samples in each diamond
    :param float r: The radius of the diamond, along each axis
    :rtype: :class:`np.ndarray` of shape (num_samples*num_centers, Ldim)
    :returns: Uniform random samples from a diamond around each center point


    """
    Lambda_dim = centers.shape[1]
    if r is None:
        rvec = np.ones([Lambda_dim, 1])
    else:
        rvec = r * np.ones([Lambda_dim, 1])

    x = np.zeros([1, centers.shape[1]])

    u = np.random.random([num_close, 1])
    b = u**(1. / Lambda_dim)

    for j in range(centers.shape[0]):
        temp = np.random.random(
            [num_close, Lambda_dim - 1]) * np.tile(b, (1, Lambda_dim - 1))
        temp = np.sort(temp, 1)
        xtemp = np.zeros([num_close, Lambda_dim])
        temp1 = np.zeros([num_close, Lambda_dim + 1])
        temp1[:, 1:Lambda_dim] = temp
        temp1[:, Lambda_dim] = np.array(b).transpose()
        for i in range(1, Lambda_dim + 1):
            xtemp[:, i - 1] = temp1[:, i] - temp1[:, i - 1]

        u_sign = 2 * np.round(np.random.random([num_close, Lambda_dim])) - 1
        xtemp = xtemp * u_sign

        for i in range(Lambda_dim):
            xtemp[:, i] = rvec[i] * xtemp[:, i]
        xtemp = xtemp + centers[j, :]
        x = np.append(x, xtemp, axis=0)

    return x[1:]


def pick_cfd_points(centers, r):
    """

    Pick 2*Lambda_dim points, for each center, for centered
    finite difference gradient approximation.

    :param centers: Points in :math:`\Lambda` to cluster points around
    :type centers: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Each side of the box will have length 2*r
    :rtype: :class:`np.ndarray` of shape (num_close*num_centers, Ldim)
    :returns: Samples for centered finite difference stencil for
        each point in centers.

    """
    Lambda_dim = centers.shape[1]
    num_centers = centers.shape[0]
    samples = np.tile(centers, [Lambda_dim * 2, 1])

    translate = r * np.kron(np.eye(Lambda_dim), np.ones([num_centers, 1]))
    translate = np.append(translate, -translate, axis=0)
    samples += translate

    return samples


def pick_ffd_points(centers, r):
    """

    Pick Lambda_dim points, for each centers, for a forward finite
    difference gradient approximation.  The ordering of these samples
    is important.

    :param centers: Points in :math:`\Lambda` to cluster points around
    :type centers: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Each side of the box will have length 2*r
    :rtype: :class:`np.ndarray` of shape (num_close*num_centers, Ldim)
    :returns: Samples for centered finite difference stencil for
        each point in centers.

    """
    Lambda_dim = centers.shape[1]
    num_centers = centers.shape[0]
    samples = np.tile(centers, [Lambda_dim, 1])

    translate = r * np.kron(np.eye(Lambda_dim), np.ones([num_centers, 1]))
    samples += translate

    return np.concatenate([centers, samples])


def radial_basis_function(r, kernel=None, ep=None):
    """

    Evaluate a chosen radial basis function.  Allow for the
    choice of several radial basis functions to use in
    the calculate_gradients_rbf.

    :param r: Distances from the reference point
    :type r: :class:`np.ndarray`
    :param string kernel: Choice of radial basis funtion
    :param float ep: Shape parameter for the radial basis function
    :rtype: :class:`np.ndarray` of shape (r.shape)
    :returns: Radial basis function evaluated for each element of r

    """
    if ep is None:
        ep = 1.0

    if kernel is None or kernel is 'C4Matern':
        rbf = (1 + (ep * r) + (ep * r)**2 / 3) * np.exp(-ep * r)
    elif kernel is 'Gaussian':
        rbf = np.exp(-(ep * r)**2)
    elif kernel is 'Multiquadric':
        rbf = (1 + (ep * r)**2)**(0.5)
    elif kernel is 'InverseMultiquadric':
        rbf = 1 / ((1 + (ep * r)**2)**(0.5))
    else:
        raise ValueError("The kernel chosen is not currently available.")

    return rbf


def radial_basis_function_dxi(r, xi, kernel=None, ep=None):
    """

    Evaluate a partial derivative of a chosen radial basis function.
    Allow for the choice of several radial basis functions to
    use in the calculate_gradients_rbf.

    :param r: Distances from the reference point
    :type r: :class:`np.ndarray`
    :param xi: Distances from the reference point in dimension i
    :type xi: :class:`np.ndarray`
    :param string kernel: Choice of radial basis funtion
    :param float ep: Shape parameter for the radial basis function
    :rtype: :class:`np.ndarray` of shape (r.shape)
    :returns: Radial basis function evaluated for each element of r

    """
    if ep is None:
        ep = 1.0

    if kernel is None or kernel is 'C4Matern':
        rbfdxi = -(ep**2 * xi * np.exp(-ep * r) * (ep * r + 1)) / 3
    elif kernel is 'Gaussian':
        rbfdxi = -2 * ep**2 * xi * np.exp(-(ep * r)**2)
    elif kernel is 'Multiquadric':
        rbfdxi = (ep**2 * xi) / ((1 + (ep * r)**2)**(0.5))
    elif kernel is 'InverseMultiquadric':
        rbfdxi = -(ep**2 * xi) / ((1 + (ep * r)**2)**(1.5))
    else:
        raise ValueError("The kernel chosen is not currently available")

    return rbfdxi


def calculate_gradients_rbf(
        samples, data, xeval, num_neighbors=None, RBF=None, ep=None):
    """

    TO DO: vectorize first for loop?

    Approximate gradient vectors at ``num_xeval, xeval.shape[0]`` points
    in the parameter space for each QoI map.

    :param samples: Samples for which the model has been solved.
    :type samples: :class:`np.ndarray` of shape (num_samples, Ldim) where Ldim
        is the dimension of the parameter space :math:`\Lambda`
    :param data: QoI values corresponding to each sample.
    :type data: :class:`np.ndarray` of shape (num_samples, Ddim) where Ddim is
        the number of QoI (i.e. the dimension of the data space
        :math:`\mathcal{D}`
    :param xeval: Points in :math:`\Lambda` at which to approximate gradient
        information.
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param int num_neighbors: Number of nearest neighbors to use in gradient
        approximation. Default value is 30.
    :param string RBF: Choice of radial basis function.
        Default is Gaussian
    :param float ep: Choice of shape parameter for radial basis function.
        Default value is 1.0
    :rtype: :class:`np.ndarray` of shape (num_samples, Ddim, Ldim)
    :returns: Tensor representation of the gradient vectors of each QoI map
        at each point in xeval

    """
    if num_neighbors is None:
        num_neighbors = 30
    if ep is None:
        ep = 1.0
    if RBF is None:
        RBF = 'Gaussian'

    Lambda_dim = samples.shape[1]
    num_model_samples = samples.shape[0]
    Data_dim = data.shape[1]
    num_xeval = xeval.shape[0]

    rbf_tensor = np.zeros([num_xeval, num_model_samples, Lambda_dim])
    gradient_tensor = np.zeros([num_xeval, Data_dim, Lambda_dim])
    tree = spatial.KDTree(samples)

    for xe in range(num_xeval):
        [r, nearest] = tree.query(xeval[xe, :], k=num_neighbors)
        r = np.tile(r, (Lambda_dim, 1))

        diffVec = (xeval[xe, :] - samples[nearest, :]).transpose()
        distMat = spatial.distance_matrix(
            samples[nearest, :], samples[nearest, :])
        rbf_mat_values = np.linalg.solve(radial_basis_function(distMat, RBF),
                                         radial_basis_function_dxi(r, diffVec, RBF, ep).transpose()).transpose()

        for ind in range(num_neighbors):
            rbf_tensor[xe, nearest[ind], :] = rbf_mat_values[
                :, ind].transpose()

    gradient_tensor = rbf_tensor.transpose(
        2, 0, 1).dot(data).transpose(1, 2, 0)

    return gradient_tensor


def calculate_gradients_cfd(samples, data, xeval, r):
    """
    TODO: Check to see if this works for multiple QoIs... see
          ffd for fix if it doesn't.  It probably doesn't.

    Approximate gradient vectors at ``num_xeval, xeval.shape[0]`` points
    in the parameter space for each QoI map.  THIS METHOD IS DEPENDENT
    ON USING pick_cfd_points TO CHOOSE SAMPLES FOR THE CFD STENCIL AROUND
    EACH XEVAL.  THE ORDERING MATTERS.

    :param samples: Samples for which the model has been solved.
    :type samples: :class:`np.ndarray` of shape (num_samples, Ldim) where Ldim
    is the dimension of the parameter space :math:`\Lambda`
    :param data: QoI values corresponding to each sample.
    :type data: :class:`np.ndarray` of shape (num_samples, Ddim) where Ddim is
        the number of QoI (i.e. the dimension of the data space
        :math:`\mathcal{D}`
    :param xeval: Points in :math:`\Lambda` at which to approximate gradient
        information.
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Distance from center to place samples
    :rtype: :class:`np.ndarray` of shape (num_samples, Ddim, Ldim)
    :returns: Tensor representation of the gradient vectors of each QoI map
        at each point in xeval

    """
    num_xeval = xeval.shape[0]
    Lambda_dim = samples.shape[1]
    num_qois = data.shape[1]
    gradient_tensor = np.zeros([num_xeval, num_qois, Lambda_dim])

    gradient_vec = (
        data[:Lambda_dim * num_xeval] - data[Lambda_dim * num_xeval:]) / (2 * r)
    gradient_tensor = gradient_vec.reshape(
        Lambda_dim, 1, num_xeval).transpose(2, 1, 0)

    return gradient_tensor


def calculate_gradients_ffd(samples, data, xeval, r):
    """

    Approximate gradient vectors at ``num_xeval, xeval.shape[0]`` points
    in the parameter space for each QoI map.
    THIS METHOD IS DEPENDENT ON USING pick_ffd_points TO CHOOSE
    SAMPLES FOR THE CFD STENCIL AROUND EACH XEVAL.  THE ORDERING MATTERS.

    :param samples: Samples for which the model has been solved.
    :type samples: :class:`np.ndarray` of shape (num_samples, Ldim) where Ldim
        is the dimension of the parameter space :math:`\Lambda`
    :param data: QoI values corresponding to each sample.
    :type data: :class:`np.ndarray` of shape (num_samples, Ddim) where Ddim is
        the number of QoI (i.e. the dimension of the data space
        :math:`\mathcal{D}`
    :param xeval: Points in :math:`\Lambda` at which to approximate gradient
        information.
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Distance from center to place samples
    :rtype: :class:`np.ndarray` of shape (num_samples, Ddim, Ldim)
    :returns: Tensor representation of the gradient vectors of each QoI map
        at each point in xeval

    """
    num_xeval = xeval.shape[0]
    Lambda_dim = samples.shape[1]
    num_qois = data.shape[1]
    gradient_tensor = np.zeros([num_xeval, num_qois, Lambda_dim])

    gradient_vec = (
        data[num_xeval:] - np.tile(data[0:num_xeval], [Lambda_dim, 1])) / r

    gradient_tensor = np.ravel(gradient_vec.transpose()).reshape(
        num_qois, Lambda_dim, num_xeval).transpose(2, 0, 1)

    return gradient_tensor
