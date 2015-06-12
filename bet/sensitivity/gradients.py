# Copyright (C) 2014-2015 The BET Development Team

"""
This module contains functions for approximating gradient vectors
of QoI maps.
"""
import numpy as np
import scipy.spatial as spatial

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

    # Define bounds for each box
    left = np.maximum(
        centers - r, np.ones([num_centers, Lambda_dim]) * lam_domain[:, 0])
    right = np.minimum(
        centers + r, np.ones([num_centers, Lambda_dim]) * lam_domain[:, 1])

    # Samples each box uniformly
    samples = np.tile(right - left, [num_close, 1])*np.random.random(
        [num_centers * num_close, Lambda_dim]) + np.tile(left, [num_close, 1])

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
    # rvec is the vector of radii of the l1_ball in each coordinate
    # direction
    if r is None:
        rvec = np.ones(Lambda_dim)
    else:
        rvec = r * np.ones(Lambda_dim)

    samples = np.zeros([1, centers.shape[1]])

    # We choose weighted random distance from the center for each new sample
    random_dist = np.random.random([num_close, 1])
    weight_vec = random_dist**(1. / Lambda_dim)

    # For each center, randomly sample the l1_ball
    for cen in range(centers.shape[0]):
        # Begin by uniformly sampling the unit simplex in the first quadrant
        # Choose Lambda_dim-1 reals uniformly between 0 and weight_vec for each
        # new sample
        random_mat = np.random.random(
            [num_close, Lambda_dim - 1]) * np.tile(weight_vec, (1, Lambda_dim - 1))

        # Sort the random_mat
        random_mat = np.sort(random_mat, 1)

        # Contrust weight_mat so that the first column is zeros, the next
        # Lambda_dim-1 columns are the sorted reals between 0 and weight_vec,
        # and the last column is weight_vec.
        weight_mat = np.zeros([num_close, Lambda_dim + 1])
        weight_mat[:, 1:Lambda_dim] = random_mat
        weight_mat[:, Lambda_dim] = np.array(weight_vec).transpose()

        # The differences between the Lambda_dim+1 columns will give us
        # random points in the unit simplex of dimension Lambda_dim.
        samples_cen = np.zeros([num_close, Lambda_dim])
        for Ldim in range(Lambda_dim):
            samples_cen[:, Ldim] = weight_mat[:, Ldim + 1] - weight_mat[:, Ldim]

        # Assign a random sign to each element of each new sample
        # Now we have samples in the l1_ball, not just the unit simplex in
        # the first quadrant
        rand_sign = 2 * np.round(np.random.random([num_close, Lambda_dim])) - 1
        samples_cen = samples_cen * rand_sign

        # Scale each dimension according to rvec and translate to center
        samples_cen = samples_cen * rvec + centers[cen, :]

        # Append newsamples to samples
        samples = np.append(samples, samples_cen, axis=0)

    return np.concatenate([centers, samples[1:]])

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

    # Contstruct a [num_centers*2*Lambda_dim, Lambda_dim] matrix that
    # translates the centers to the CFD points
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

    # Construct a [num_centers*(Lambda_dim+1), Lambda_dim] matrix that
    # translates the senters to the FFD points.
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
    :returns: Tensor representation of the normalized gradient vectors of each
        QoI map at each point in xeval

    """
    Lambda_dim = samples.shape[1]
    if num_neighbors is None:
        num_neighbors = Lambda_dim + 1
    if ep is None:
        ep = 1.0
    if RBF is None:
        RBF = 'Gaussian'

    num_model_samples = samples.shape[0]
    Data_dim = data.shape[1]
    num_xeval = xeval.shape[0]

    rbf_tensor = np.zeros([num_xeval, num_model_samples, Lambda_dim])
    gradient_tensor = np.zeros([num_xeval, Data_dim, Lambda_dim])
    tree = spatial.KDTree(samples)

    # For each xeval, interpolate the data using the rbf chosen and
    # then evaluate the partail derivative of that rbf at the desired point. 
    for xe in range(num_xeval):
        # Find the k nearest neightbors and their distances to xeval[xe,:]
        [r, nearest] = tree.query(xeval[xe, :], k=num_neighbors)
        r = np.tile(r, (Lambda_dim, 1))

        # Compute the linf distances to each of the nearest neighbors
        diffVec = (xeval[xe, :] - samples[nearest, :]).transpose()

        # Compute the l2 distances between pairs of nearest neighbors
        distMat = spatial.distance_matrix(
            samples[nearest, :], samples[nearest, :])

        # 
        rbf_mat_values = np.linalg.solve(radial_basis_function(distMat, RBF),
            radial_basis_function_dxi(r, diffVec, RBF, ep).transpose()).transpose()

        for ind in range(num_neighbors):
            rbf_tensor[xe, nearest[ind], :] = rbf_mat_values[
                :, ind].transpose()

    gradient_tensor = rbf_tensor.transpose(
        2, 0, 1).dot(data).transpose(1, 2, 0)

    # Normalize each gradient vector
    norm_gradient_tensor = np.linalg.norm(gradient_tensor, axis=2)
    gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
        (Lambda_dim, 1, 1)).transpose(1 ,2, 0)

    return gradient_tensor

def calculate_gradients_cfd(samples, data, xeval, r):
    """

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
    :returns: Tensor representation of the normalized gradient vectors of each
        QoI map at each point in xeval

    """
    num_xeval = xeval.shape[0]
    Lambda_dim = samples.shape[1]
    num_qois = data.shape[1]
    gradient_tensor = np.zeros([num_xeval, num_qois, Lambda_dim])

    # Compute the gradient vectors using the standard CFD stencil
    gradient_vec = (
        data[:Lambda_dim * num_xeval] - data[Lambda_dim * num_xeval:]) / (2 * r)

    # Reshape and organize
    gradient_tensor = np.ravel(gradient_vec.transpose()).reshape(
        num_qois, Lambda_dim, num_xeval).transpose(2, 0, 1)

    # Normalize each gradient vector
    norm_gradient_tensor = np.linalg.norm(gradient_tensor, axis=2)
    gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
        (Lambda_dim, 1, 1)).transpose(1, 2, 0)

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
    :returns: Tensor representation of the normalized gradient vectors of each
        QoI map at each point in xeval

    """
    num_xeval = xeval.shape[0]
    Lambda_dim = samples.shape[1]
    num_qois = data.shape[1]
    gradient_tensor = np.zeros([num_xeval, num_qois, Lambda_dim])

    # Compute the gradient vectors using the standard FFD stencil
    gradient_vec = (
        data[num_xeval:] - np.tile(data[0:num_xeval], [Lambda_dim, 1])) / r

    # Reshape and organize
    gradient_tensor = np.ravel(gradient_vec.transpose()).reshape(
        num_qois, Lambda_dim, num_xeval).transpose(2, 0, 1)

    # Normalize each gradient vector
    norm_gradient_tensor = np.linalg.norm(gradient_tensor, axis=2)
    gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
        (Lambda_dim, 1, 1)).transpose(1, 2, 0)

    return gradient_tensor
