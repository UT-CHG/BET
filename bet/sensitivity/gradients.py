# Copyright (C) 2014-2016 The BET Development Team

"""
This module contains functions for approximating jacobians of QoI maps.
All methods that cluster points around centers are written to return the
input_set._values in the following order : CENTERS, FOLLOWED BY THE CLUSTER
AROUND THE FIRST CENTER, THEN THE CLUSTER AROUND THE SECOND CENTER AND SO ON.
"""
import numpy as np
import scipy.spatial as spatial
import bet.util as util
import sys

def sample_linf_ball(input_set, num_close, rvec):
    r"""
    Pick num_close points in a the l-infinity ball of length 2*rvec around a
    point in the input space, do this for each point in centers.  If this box
    extends outside of the domain of the input space, we sample the
    intersection.

    :param input_set: The input sample set.  Make sure the attribute _values is
        not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param int num_close: Number of points in each cluster
    :param rvec: Each side of the box will have length 2*rvec[i]
    :type rvec: :class:`np.ndarray` of shape (input_dim,)
    
    :rtype: :class:`np.ndarray` of shape ((num_close+1)*num_centers, input_dim)
    :returns: Centers and clusters of samples near each center
    
    """
    if input_set._values is None:
        raise ValueError("You must have values to use this method.")
    input_dim = input_set._dim
    centers = input_set._values
    num_centers = centers.shape[0]
    input_domain = input_set._domain
    rvec = util.fix_dimensions_vector(rvec)

    #If no lam_domain, set domain large
    if input_domain is None:
        input_domain = np.zeros([input_dim, 2])
        input_domain[:, 0] = -sys.float_info[0]
        input_domain[:, 1] = sys.float_info[0]

    # Define bounds for each box
    left = np.maximum(
        centers - rvec, np.ones([num_centers, input_dim]) * input_domain[:, 0])
    right = np.minimum(
        centers + rvec, np.ones([num_centers, input_dim]) * input_domain[:, 1])

    # Samples each box uniformly
    samples = np.repeat(right - left, num_close, axis=0) * np.random.random(
        [num_centers * num_close, input_dim]) + np.repeat(left, num_close, \
        axis=0)

    return np.concatenate([centers, samples])

def sample_l1_ball(input_set, num_close, rvec):
    r"""
    Uniformly sample the l1-ball (defined by 2^dim simplices).  Then scale
    each dimension according to rvec and translate the center to centers.
    Do this for each point in centers.  *This method currently allows
    samples to be placed outside of lam_domain.  Please place your
    centers accordingly.*
    
    :param input_set: The input sample set.  Make sure the attribute _values is
        not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param int num_close: Number of samples in each l1 ball
    :param rvec: The radius of the l1 ball, along each axis
    :type rvec: :class:`np.ndarray` of shape (input_dim)
    
    :rtype: :class:`np.ndarray` of shape ((num_close+1)*num_centers, input_dim)
    :returns: Uniform random samples from an l1 ball around each center
    
    """
    if input_set._values is None:
        raise ValueError("You must have values to use this method.")
    input_dim = input_set._dim
    centers = input_set._values
    rvec = util.fix_dimensions_vector(rvec)

    samples = np.zeros([(num_close + 1) * centers.shape[0], centers.shape[1]])
    samples[0:centers.shape[0], :] = centers

    # We choose weighted random distance from the center for each new sample
    random_dist = np.random.random([num_close, 1])
    weight_vec = random_dist**(1. / input_dim)

    # For each center, randomly sample the l1_ball
    for cen in range(centers.shape[0]):
        # Begin by uniformly sampling the unit simplex in the first quadrant
        # Choose input_dim-1 reals uniformly between 0 and weight_vec for each
        # new sample
        random_mat = np.random.random([num_close, input_dim - 1]) * \
            np.tile(weight_vec, (1, input_dim - 1))

        # Sort the random_mat
        random_mat = np.sort(random_mat, 1)

        # Contrust weight_mat so that the first column is zeros, the next
        # input_dim-1 columns are the sorted reals between 0 and weight_vec,
        # and the last column is weight_vec.
        weight_mat = np.zeros([num_close, input_dim + 1])
        weight_mat[:, 1:input_dim] = random_mat
        weight_mat[:, input_dim] = np.array(weight_vec).transpose()

        # The differences between the input_dim+1 columns will give us
        # random points in the unit simplex of dimension input_dim.
        samples_cen = np.zeros([num_close, input_dim])
        for Ldim in range(input_dim):
            samples_cen[:, Ldim] = weight_mat[:, Ldim + 1] - weight_mat[:, Ldim]

        # Assign a random sign to each element of each new sample
        # This give us samples in the l1_ball, not just the unit simplex in
        # the first quadrant
        rand_sign = 2 * np.round(np.random.random([num_close, input_dim])) - 1
        samples_cen = samples_cen * rand_sign

        # Scale each dimension according to rvec and translate to center
        samples_cen = samples_cen * rvec + centers[cen, :]

        # Append newsamples to samples
        samples[centers.shape[0] + cen * num_close:centers.shape[0] + \
            (cen + 1) * num_close, :] = samples_cen

    return samples

def pick_ffd_points(input_set, rvec):
    r"""
    Pick input_dim points, for each centers, for a forward finite
    difference gradient approximation.  The points are returned in the order:
    centers, followed by the cluster around the first center, then the cluster
    around the second center and so on.
    
    :param input_set: The input sample set.  Make sure the attribute _values is
        not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param rvec: The radius of the stencil, along each axis
    :type rvec: :class:`np.ndarray` of shape (input_dim,)
    
    :rtype: :class:`np.ndarray` of shape ((input_dim+1)*num_centers,
        input_dim)
    :returns: Samples for centered finite difference stencil for
        each point in centers.
    
    """
    if input_set._values is None:
        raise ValueError("You must have values to use this method.")
    input_dim = input_set._dim
    centers = input_set._values
    num_centers = centers.shape[0]
    samples = np.repeat(centers, input_dim, axis=0)
    rvec = util.fix_dimensions_vector(rvec)

    # Construct a [num_centers*(input_dim+1), input_dim] matrix that
    # translates the centers to the FFD points.
    translate = np.tile(np.eye(input_dim) * rvec, (num_centers, 1))
    samples = samples + translate

    return np.concatenate([centers, samples])

def pick_cfd_points(input_set, rvec):
    r"""
    Pick 2*input_dim points, for each center, for centered finite difference
    gradient approximation.  The center are not needed for the CFD gradient
    approximation, they are returned for consistency with the other methods and
    because of the common need to have not just the gradient but also the QoI
    value at the centers in adaptive sampling algorithms.The points are returned 
    in the order: centers, followed by the cluster around the first center, then 
    the cluster around the second center and so on.
    
    :param input_set: The input sample set.  Make sure the attribute _values is
        not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param rvec: The radius of the stencil, along each axis
    :type rvec: :class:`np.ndarray` of shape (input_dim,)
    
    :rtype: :class:`np.ndarray` of shape ((2*input_dim+1)*num_centers,
        input_dim)
    :returns: Samples for centered finite difference stencil for
        each point in centers.
    
    """
    if input_set._values is None:
        raise ValueError("You must have values to use this method.")
    input_dim = input_set._dim
    centers = input_set._values
    num_centers = centers.shape[0]
    samples = np.repeat(centers, 2 * input_dim, axis=0)
    rvec = util.fix_dimensions_vector(rvec)

    # Contstruct a [num_centers*2*input_dim, input_dim] array that
    # translates the centers to the CFD points
    ident = np.eye(input_dim) * rvec
    translate = np.tile(np.append(ident, -ident, axis=0), (num_centers, 1))
    samples = samples + translate

    return np.concatenate([centers, samples])

def radial_basis_function(r, kernel=None, ep=None):
    """
    Evaluate a chosen radial basis function.  Allow for the choice of several
    radial basis functions to use in
    :meth:~bet.sensitivity.gradients.calculate_gradients_rbf
    
    :param r: Distances from the reference point
    :type r: :class:`np.ndarray`
    :param string kernel: Choice of radial basis funtion. Default is C4Matern
    :param float ep: Shape parameter for the radial basis function.
        Default is 1.0
    
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
    Evaluate a partial derivative of a chosen radial basis function.  Allow for
    the choice of several radial basis functions to use in the
    :meth:~bet.sensitivity.gradients.calculate_gradients_rbf.
    
    :param r: Distances from the reference point
    :type r: :class:`np.ndarray`
    :param xi: Distances from the reference point in dimension i
    :type xi: :class:`np.ndarray`
    :param string kernel: Choice of radial basis funtion. Default is C4Matern
    :param float ep: Shape parameter for the radial basis function.
        Default is 1.0
    
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

def calculate_gradients_rbf(input_set, output_set, input_set_centers=None,
        num_neighbors=None, RBF=None, ep=None, normalize=True): 
    r"""
    Approximate gradient vectors at ``num_centers, centers.shape[0]`` points
    in the parameter space for each QoI map using a radial basis function
    interpolation method.
    
    :param input_set: The input sample set.  Make sure the attribute _values is
        not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param output_set: The output sample set.  Make sure the attribute _values
        is not None.
    :type output_set: :class:`~bet.sample.sample_set`
    :param input_set_centers: The input centers sample set.  Make sure the
        attribute _values is not None.
    :type input_set_centers: :class:`~bet.sample.sample_set`
    :param int num_neighbors: Number of nearest neighbors to use in gradient
        approximation. Default value is input_dim + 2.
    :param string RBF: Choice of radial basis function. Default is Gaussian
    :param float ep: Choice of shape parameter for radial basis function.
        Default value is 1.0
    :param boolean normalize:  If normalize is True, normalize each gradient
        vector
    
    :rtype: :class:`np.ndarray` of shape (num_samples, output_dim, input_dim)
    :returns: Tensor representation of the gradient vectors of each
        QoI map at each point in centers
    
    """
    if input_set._values is None or output_set._values is None:
        raise ValueError("You must have values to use this method.")
    samples = input_set._values
    data = output_set._values

    data = util.fix_dimensions_vector_2darray(util.clean_data(data))
    input_dim = samples.shape[1]
    num_model_samples = samples.shape[0]
    output_dim = data.shape[1]

    if num_neighbors is None:
        num_neighbors = input_dim + 2
    if ep is None:
        ep = 1.0
    if RBF is None:
        RBF = 'Gaussian'

    # If centers is None we assume the user chose clusters of size
    # input_dim + 2
    if input_set_centers is None:
        num_centers = num_model_samples / (input_dim + 2)
        centers = samples[:num_centers]
    else:
        centers = input_set_centers._values
        num_centers = centers.shape[0]

    rbf_tensor = np.zeros([num_centers, num_model_samples, input_dim])
    gradient_tensor = np.zeros([num_centers, output_dim, input_dim])
    tree = spatial.KDTree(samples)

    # For each center, interpolate the data using the rbf chosen and
    # then evaluate the partial derivative of that interpolant at the desired
    # point.
    for c in range(num_centers):
        # Find the k nearest neighbors and their distances to centers[c,:]
        [r, nearest] = tree.query(centers[c, :], k=num_neighbors)
        r = np.tile(r, (input_dim, 1))

        # Compute the linf distances to each of the nearest neighbors
        diffVec = (centers[c, :] - samples[nearest, :]).transpose()

        # Compute the l2 distances between pairs of nearest neighbors
        distMat = spatial.distance_matrix(
            samples[nearest, :], samples[nearest, :])

        # Solve for the rbf weights using interpolation conditions and
        # evaluate the partial derivatives
        rbf_mat_values = \
            np.linalg.solve(radial_basis_function(distMat, RBF),
            radial_basis_function_dxi(r, diffVec, RBF, ep) \
            .transpose()).transpose()

        # Construct the finite difference matrices
        rbf_tensor[c, nearest, :] = rbf_mat_values.transpose()

    gradient_tensor = rbf_tensor.transpose(2, 0, 1).dot(data).transpose(1, 2, 0)

    if normalize:
        # Compute the norm of each vector
        norm_gradient_tensor = np.linalg.norm(gradient_tensor, ord=1, axis=2)

        # If it is a zero vector (has 0 norm), set norm=1, avoid divide by zero
        norm_gradient_tensor[norm_gradient_tensor == 0] = 1.0

        # Normalize each gradient vector
        gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
            (input_dim, 1, 1)).transpose(1, 2, 0)

    return gradient_tensor

def calculate_gradients_ffd(input_set, output_set, normalize=True):
    """
    Approximate gradient vectors at ``num_centers, centers.shape[0]`` points
    in the parameter space for each QoI map.  THIS METHOD IS DEPENDENT ON USING
    :meth:~bet.sensitivity.gradients.pick_ffd_points TO CHOOSE SAMPLES FOR THE
    FFD STENCIL AROUND EACH CENTER. THE ORDERING MATTERS.
    
    :param input_set: The input sample set.  Make sure the attribute _values is
        not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param output_set: The output sample set.  Make sure the attribute _values
        is not None.
    :type output_set: :class:`~bet.sample.sample_set`
    :param boolean normalize:  If normalize is True, normalize each gradient
        vector
    
    :rtype: :class:`np.ndarray` of shape (num_samples, output_dim, input_dim)
    :returns: Tensor representation of the gradient vectors of each
        QoI map at each point in centers
    
    """
    if input_set._values is None or output_set._values is None:
        raise ValueError("You must have values to use this method.")
    samples = input_set._values
    data = output_set._values

    num_model_samples = samples.shape[0]
    input_dim = samples.shape[1]
    num_centers = num_model_samples / (input_dim + 1)

    # Find rvec from the first cluster of samples
    rvec = samples[num_centers:num_centers + input_dim, :] - samples[0, :]
    rvec = util.fix_dimensions_vector_2darray(rvec.diagonal())

    # Clean the data
    data = util.fix_dimensions_vector_2darray(util.clean_data(data))
    num_qois = data.shape[1]
    gradient_tensor = np.zeros([num_centers, num_qois, input_dim])

    rvec = np.tile(np.repeat(rvec, num_qois, axis=1), [num_centers, 1])

    # Compute the gradient vectors using the standard FFD stencil
    gradient_mat = (data[num_centers:] - np.repeat(data[0:num_centers], \
                   input_dim, axis=0)) * (1. / rvec)

    # Reshape and organize
    gradient_tensor = np.reshape(gradient_mat.transpose(), [num_qois,
        input_dim, num_centers], order='F').transpose(2, 0, 1)

    if normalize:
        # Compute the norm of each vector
        norm_gradient_tensor = np.linalg.norm(gradient_tensor, ord=1, axis=2)

        # If it is a zero vector (has 0 norm), set norm=1, avoid divide by zero
        norm_gradient_tensor[norm_gradient_tensor == 0] = 1.0

        # Normalize each gradient vector
        gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
            (input_dim, 1, 1)).transpose(1, 2, 0)

    return gradient_tensor

def calculate_gradients_cfd(input_set, output_set, normalize=True):
    """
    Approximate gradient vectors at ``num_centers, centers.shape[0]`` points
    in the parameter space for each QoI map.  THIS METHOD IS DEPENDENT
    ON USING :meth:~bet.sensitivity.pick_cfd_points TO CHOOSE SAMPLES FOR THE 
    CFD STENCIL AROUND EACH CENTER.  THE ORDERING MATTERS.
    
    :param input_set: The input sample set.  Make sure the attribute _values is
        not None.
    :type input_set: :class:`~bet.sample.sample_set`
    :param output_set: The output sample set.  Make sure the attribute _values
        is not None.
    :type output_set: :class:`~bet.sample.sample_set`
    :param boolean normalize:  If normalize is True, normalize each gradient
        vector
    
    :rtype: :class:`np.ndarray` of shape (num_samples, output_dim, input_dim)
    :returns: Tensor representation of the gradient vectors of each
        QoI map at each point in centers
    
    """
    if input_set._values is None or output_set._values is None:
        raise ValueError("You must have values to use this method.")
    samples = input_set._values
    data = output_set._values

    num_model_samples = samples.shape[0]
    input_dim = samples.shape[1]
    num_centers = num_model_samples / (2*input_dim + 1)

    # Find rvec from the first cluster of samples
    rvec = samples[num_centers:num_centers + input_dim, :] - samples[0, :]
    rvec = util.fix_dimensions_vector_2darray(rvec.diagonal())

    # Clean the data
    data = util.fix_dimensions_vector_2darray(util.clean_data(
        data[num_centers:]))
    num_qois = data.shape[1]
    gradient_tensor = np.zeros([num_centers, num_qois, input_dim])

    rvec = np.tile(np.repeat(rvec, num_qois, axis=1), [num_centers, 1])

    # Construct indices for CFD gradient approxiation
    inds = np.repeat(range(0, 2 * input_dim * num_centers, 2 * input_dim),
        input_dim) + np.tile(range(0, input_dim), num_centers)
    inds = np.array([inds, inds+input_dim]).transpose()

    gradient_mat = (data[inds[:, 0]] - data[inds[:, 1]]) * (0.5 / rvec)

    # Reshape and organize
    gradient_tensor = np.reshape(gradient_mat.transpose(), [num_qois,
        input_dim, num_centers], order='F').transpose(2, 0, 1)

    if normalize:
        # Compute the norm of each vector
        norm_gradient_tensor = np.linalg.norm(gradient_tensor, ord=1, axis=2)

        # If it is a zero vector (has 0 norm), set norm=1, avoid divide by zero
        norm_gradient_tensor[norm_gradient_tensor == 0] = 1.0

        # Normalize each gradient vector
        gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
            (input_dim, 1, 1)).transpose(1, 2, 0)

    return gradient_tensor
