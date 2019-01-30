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
import bet.sample as sample
import bet.sampling.LpGeneralizedSamples as lpsam

def sample_lp_ball(input_set, num_close, radius, p_num=2):
    r"""
    Pick num_close points in a the Lp ball of length 2*``radii_vec`` around a
    point in the input space, do this for each point in centers.  If this box
    extends outside of the domain of the input space, we sample the
    intersection.

    :param input_set: The input sample set.  Make sure the attribute
        ``_values`` is not ``None``
    :type input_set: :class:`~bet.sample.sample_set`
    :param int num_close: Number of points in each cluster
    :param radii_vec: Each side of the box will have length ``2*radii_vec[i]``
    :type radii_vec: :class:`numpy.ndarray` of shape (``input_dim``,)
    :param float p_num: :math:`0 < p \leq \infty`, p for the lp norm where
        infinity is ``numpy.inf``
    
    :rtype: :class:`~bet.sample.sample_set`
    :returns: Centers and clusters of samples near each center (values are 
        :class:`numpy.ndarray` of shape ((``num_close+1``)*``num_centers``,
        ``input_dim``))
    
    """
    if input_set.get_values() is None:
        raise ValueError("You must have values to use this method.")
    input_dim = input_set.get_dim()
    centers = input_set.get_values()
    num_centers = input_set.check_num()
    input_domain = input_set.get_domain()

    cluster_set = sample.sample_set(input_dim)
    if input_domain is not None:
        cluster_set.set_domain(input_domain)
    cluster_set.set_values(centers)

    for i in xrange(num_centers):
        in_bounds = 0
        inflate = 1.0
        while in_bounds < num_close:
            # sample uniformly
            new_cluster = lpsam.Lp_generalized_uniform(input_dim,
                    num_close*inflate, p_num, radius, centers[i, :])
            # check bounds
            if input_domain is not None:
                cluster_set.update_bounds(num_close*inflate)
                left = np.all(np.greater_equal(new_cluster, cluster_set._left),
                    axis=1)
                right = np.all(np.less_equal(new_cluster, cluster_set._right),
                    axis=1)
                inside = np.logical_and(left, right)
                in_bounds = np.sum(inside)
                new_cluster = new_cluster[inside, :]
                # increase inflate
                inflate *= 10.0
            else:
                in_bounds = num_close

        if in_bounds > num_close:
            new_cluster = new_cluster[:num_close, :]
        cluster_set.append_values(new_cluster)

    # reset bounds
    cluster_set._left = None
    cluster_set._right = None
    cluster_set._width = None
    return cluster_set


def sample_linf_ball(input_set, num_close, radii_vec):
    r"""
    Pick num_close points in a the L-inifity ball of length 2*``radii_vec``
    around a point in the input space, do this for each point in centers.  If
    this box extends outside of the domain of the input space, we sample the
    intersection.

    :param input_set: The input sample set.  Make sure the attribute
        ``_values`` is not ``None``
    :type input_set: :class:`~bet.sample.sample_set`
    :param int num_close: Number of points in each cluster
    :param radii_vec: Each side of the box will have length ``2*radii_vec[i]``
    :type radii_vec: :class:`numpy.ndarray` of shape (``input_dim``,)
    
    :rtype: :class:`~bet.sample.sample_set`
    :returns: Centers and clusters of samples near each center (values are 
        :class:`numpy.ndarray` of shape ((``num_close+1``)*``num_centers``,
        ``input_dim``))
    
    """
    return sample_lp_ball(input_set, num_close, radii_vec, p_num=np.inf)

def sample_l1_ball(input_set, num_close, radii_vec):
    r"""
    Pick num_close points in a the L-1 ball of length 2*``radii_vec`` around a
    point in the input space, do this for each point in centers.  If this box
    extends outside of the domain of the input space, we sample the
    intersection.

    :param input_set: The input sample set.  Make sure the attribute
        ``_values`` is not ``None``
    :type input_set: :class:`~bet.sample.sample_set`
    :param int num_close: Number of points in each cluster
    :param radii_vec: Each side of the box will have length ``2*radii_vec[i]``
    :type radii_vec: :class:`numpy.ndarray` of shape (``input_dim``,)
    
    :rtype: :class:`~bet.sample.sample_set`
    :returns: Centers and clusters of samples near each center (values are 
        :class:`numpy.ndarray` of shape ((``num_close+1``)*``num_centers``,
        ``input_dim``))

    """
    return sample_lp_ball(input_set, num_close, radii_vec, p_num=1)

def pick_ffd_points(input_set, radii_vec):
    r"""
    Pick input_dim points, for each centers, for a forward finite
    difference gradient approximation.  The points are returned in the order:
    centers, followed by the cluster around the first center, then the cluster
    around the second center and so on.
    
    :param input_set: The input sample set.  Make sure the attribute _values is
        not None
    :type input_set: :class:`~bet.sample.sample_set`
    :param radii_vec: The radius of the stencil, along each axis
    :type radii_vec: :class:`numpy.ndarray` of shape (input_dim,)
    
    :rtype: :class:`~bet.sample.sample_set`
    :returns: Centers and clusters of samples near each center (values are 
        :class:`numpy.ndarray` of shape ((``num_close+1``)*``num_centers``,
        ``input_dim``))
    
    """
    if input_set._values is None:
        raise ValueError("You must have values to use this method.")
    input_dim = input_set.get_dim()
    centers = input_set.get_values()
    num_centers = centers.shape[0]
    samples = np.repeat(centers, input_dim, axis=0)
    radii_vec = util.fix_dimensions_vector(radii_vec)

    # Construct a [num_centers*(input_dim+1), input_dim] matrix that
    # translates the centers to the FFD points.
    translate = np.tile(np.eye(input_dim) * radii_vec, (num_centers, 1))
    samples = samples + translate

    cluster_set = sample.sample_set(input_dim)
    if input_set.get_domain() is not None:
        cluster_set.set_domain(input_set.get_domain())
    cluster_set.set_values(centers)
    cluster_set.append_values(samples)
    return cluster_set

def pick_cfd_points(input_set, radii_vec):
    r"""
    Pick 2*input_dim points, for each center, for centered finite difference
    gradient approximation.  The center are not needed for the CFD gradient
    approximation, they are returned for consistency with the other methods and
    because of the common need to have not just the gradient but also the QoI
    value at the centers in adaptive sampling algorithms.The points are returned 
    in the order: centers, followed by the cluster around the first center, then 
    the cluster around the second center and so on.
    
    :param input_set: The input sample set.  Make sure the attribute _values is
        not None
    :type input_set: :class:`~bet.sample.sample_set`
    :param radii_vec: The radius of the stencil, along each axis
    :type radii_vec: :class:`numpy.ndarray` of shape (input_dim,)
    
    :rtype: :class:`~bet.sample.sample_set`
    :returns: Centers and clusters of samples near each center (values are 
        :class:`numpy.ndarray` of shape ((``num_close+1``)*``num_centers``,
        ``input_dim``))
    
    """
    if input_set._values is None:
        raise ValueError("You must have values to use this method.")
    input_dim = input_set.get_dim()
    centers = input_set.get_values()
    num_centers = centers.shape[0]
    samples = np.repeat(centers, 2 * input_dim, axis=0)
    radii_vec = util.fix_dimensions_vector(radii_vec)

    # Contstruct a [num_centers*2*input_dim, input_dim] array that
    # translates the centers to the CFD points
    ident = np.eye(input_dim) * radii_vec
    translate = np.tile(np.append(ident, -ident, axis=0), (num_centers, 1))
    samples = samples + translate

    cluster_set = sample.sample_set(input_dim)
    if input_set.get_domain() is not None:
        cluster_set.set_domain(input_set.get_domain())
    cluster_set.set_values(centers)
    cluster_set.append_values(samples)
    return cluster_set

def radial_basis_function(r, kernel=None, ep=None):
    """
    Evaluate a chosen radial basis function.  Allow for the choice of several
    radial basis functions to use in
    :meth:~bet.sensitivity.gradients.calculate_gradients_rbf
    
    :param r: Distances from the reference point
    :type r: :class:`numpy.ndarray`
    :param string kernel: Choice of radial basis funtion. Default is C4Matern
    :param float ep: Shape parameter for the radial basis function.
        Default is 1.0
    
    :rtype: :class:`numpy.ndarray` of shape (r.shape)
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
    :type r: :class:`numpy.ndarray`
    :param xi: Distances from the reference point in dimension i
    :type xi: :class:`numpy.ndarray`
    :param string kernel: Choice of radial basis funtion. Default is C4Matern
    :param float ep: Shape parameter for the radial basis function.
        Default is 1.0
    
    :rtype: :class:`numpy.ndarray` of shape (r.shape)
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

def calculate_gradients_rbf(cluster_discretization, num_centers=None,
        num_neighbors=None, RBF=None, ep=None, normalize=True): 
    r"""
    Approximate gradient vectors at ``num_centers, centers.shape[0]`` points
    in the parameter space for each QoI map using a radial basis function
    interpolation method.
    
    :param cluster_discretization: Must contain input and output values for the
        sample clusters.
    :type cluster_discretization: :class:`~bet.sample.discretization`
    :param int num_centers: The number of cluster centers.
    :param int num_neighbors: Number of nearest neighbors to use in gradient
        approximation. Default value is ``input_dim + 2``
    :param string RBF: Choice of radial basis function. Default is Gaussian
    :param float ep: Choice of shape parameter for radial basis function.
        Default value is 1.0
    :param boolean normalize:  If normalize is True, normalize each gradient
        vector
    
    :rtype: :class:`~bet.sample.discretization`
    :returns: A new :class:`~bet.sample.discretization` that contains only the
        centers of the clusters and their associated ``_jacobians`` which are
        tensor representation of the gradient vectors of each QoI map at each
        point in centers :class:`numpy.ndarray` of shape (num_samples,
        output_dim, input_dim)
    
    """
    if cluster_discretization._input_sample_set.get_values() is None \
            or cluster_discretization._output_sample_set.get_values() is None:
        raise ValueError("You must have values to use this method.")
    samples = cluster_discretization._input_sample_set.get_values()
    data = cluster_discretization._output_sample_set.get_values()

    input_dim = cluster_discretization._input_sample_set.get_dim()
    num_model_samples = cluster_discretization.check_nums()
    output_dim = cluster_discretization._output_sample_set.get_dim()

    if num_neighbors is None:
        num_neighbors = input_dim + 2
    if ep is None:
        ep = 1.0
    if RBF is None:
        RBF = 'Gaussian'

    # If centers is None we assume the user chose clusters of size
    # input_dim + 2
    if num_centers is None:
        num_centers = num_model_samples / (input_dim + 2)
    centers = samples[:num_centers, :]

    rbf_tensor = np.zeros([num_centers, num_model_samples, input_dim])
    gradient_tensor = np.zeros([num_centers, output_dim, input_dim])

    # For each center, interpolate the data using the rbf chosen and
    # then evaluate the partial derivative of that interpolant at the desired
    # point.
    for c in range(num_centers):
        # Find the k nearest neighbors and their distances to centers[c,:]
        [r, nearest] = cluster_discretization._input_sample_set.query(\
                centers[c, :], k=num_neighbors)
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

    center_input_sample_set = sample.sample_set(input_dim)
    center_input_sample_set.set_values(samples[:num_centers, :])
    if cluster_discretization._input_sample_set.get_domain() is not None:
        center_input_sample_set.set_domain(cluster_discretization.\
                _input_sample_set.get_domain())
    center_input_sample_set.set_jacobians(gradient_tensor)
    center_output_sample_set = sample.sample_set(output_dim)
    center_output_sample_set.set_values(data[:num_centers, :])
    if cluster_discretization._output_sample_set.get_domain() is not None:
        center_output_sample_set.set_domain(cluster_discretization.\
                _output_sample_set.get_domain())
    #center_output_sample_set.set_jacobians(gradient_tensor.transpose())
    center_discretization = sample.discretization(center_input_sample_set,
            center_output_sample_set)
    return center_discretization

def calculate_gradients_ffd(cluster_discretization, normalize=True):
    """
    Approximate gradient vectors at ``num_centers, centers.shape[0]`` points
    in the parameter space for each QoI map.  THIS METHOD IS DEPENDENT ON USING
    :meth:~bet.sensitivity.gradients.pick_ffd_points TO CHOOSE SAMPLES FOR THE
    FFD STENCIL AROUND EACH CENTER. THE ORDERING MATTERS.
    
    :param cluster_discretization: Must contain input and output values for the
        sample clusters.
    :type cluster_discretization: :class:`~bet.sample.discretization`
    :param boolean normalize:  If normalize is True, normalize each gradient
        vector
    
    :rtype: :class:`~bet.sample.discretization`
    :returns: A new :class:`~bet.sample.discretization` that contains only the
        centers of the clusters and their associated ``_jacobians`` which are
        tensor representation of the gradient vectors of each QoI map at each
        point in centers :class:`numpy.ndarray` of shape (num_samples,
        output_dim, input_dim)
    
    """
    if cluster_discretization._input_sample_set.get_values() is None \
            or cluster_discretization._output_sample_set.get_values() is None:
        raise ValueError("You must have values to use this method.")
    samples = cluster_discretization._input_sample_set.get_values()
    data = cluster_discretization._output_sample_set.get_values()

    input_dim = cluster_discretization._input_sample_set.get_dim()
    num_model_samples = cluster_discretization.check_nums()
    output_dim = cluster_discretization._output_sample_set.get_dim()

    num_model_samples = cluster_discretization.check_nums()
    input_dim = cluster_discretization._input_sample_set.get_dim()
    num_centers = num_model_samples / (input_dim + 1)

    # Find radii_vec from the first cluster of samples
    radii_vec = samples[num_centers:num_centers + input_dim, :] - samples[0, :]
    radii_vec = util.fix_dimensions_vector_2darray(radii_vec.diagonal())

    # Clean the data
    data = util.clean_data(data)
    gradient_tensor = np.zeros([num_centers, output_dim, input_dim])

    radii_vec = np.tile(np.repeat(radii_vec, output_dim, axis=1), [num_centers,
        1])

    # Compute the gradient vectors using the standard FFD stencil
    gradient_mat = (data[num_centers:] - np.repeat(data[0:num_centers], \
                   input_dim, axis=0)) * (1. / radii_vec)

    # Reshape and organize
    gradient_tensor = np.reshape(gradient_mat.transpose(), [output_dim,
        input_dim, num_centers], order='F').transpose(2, 0, 1)

    if normalize:
        # Compute the norm of each vector
        norm_gradient_tensor = np.linalg.norm(gradient_tensor, ord=1, axis=2)

        # If it is a zero vector (has 0 norm), set norm=1, avoid divide by zero
        norm_gradient_tensor[norm_gradient_tensor == 0] = 1.0

        # Normalize each gradient vector
        gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
            (input_dim, 1, 1)).transpose(1, 2, 0)

    center_input_sample_set = sample.sample_set(input_dim)
    center_input_sample_set.set_values(samples[:num_centers, :])
    if cluster_discretization._input_sample_set.get_domain() is not None:
        center_input_sample_set.set_domain(cluster_discretization.\
                _input_sample_set.get_domain())
    center_input_sample_set.set_jacobians(gradient_tensor)
    center_output_sample_set = sample.sample_set(output_dim)
    center_output_sample_set.set_values(data[:num_centers, :])
    if cluster_discretization._output_sample_set.get_domain() is not None:
        center_output_sample_set.set_domain(cluster_discretization.\
                _output_sample_set.get_domain())
    #center_output_sample_set.set_jacobians(gradient_tensor.transpose())
    center_discretization = sample.discretization(center_input_sample_set,
            center_output_sample_set)
    return center_discretization

def calculate_gradients_cfd(cluster_discretization, normalize=True):
    """
    Approximate gradient vectors at ``num_centers, centers.shape[0]`` points
    in the parameter space for each QoI map.  THIS METHOD IS DEPENDENT
    ON USING :meth:~bet.sensitivity.pick_cfd_points TO CHOOSE SAMPLES FOR THE 
    CFD STENCIL AROUND EACH CENTER.  THE ORDERING MATTERS.
    
    :param cluster_discretization: Must contain input and output values for the
        sample clusters.
    :type cluster_discretization: :class:`~bet.sample.discretization`
    :param boolean normalize:  If normalize is True, normalize each gradient
        vector
    
    :rtype: :class:`~bet.sample.discretization`
    :returns: A new :class:`~bet.sample.discretization` that contains only the
        centers of the clusters and their associated ``_jacobians`` which are
        tensor representation of the gradient vectors of each QoI map at each
        point in centers :class:`numpy.ndarray` of shape (num_samples,
        output_dim, input_dim)
    
    """
    if cluster_discretization._input_sample_set.get_values() is None \
            or cluster_discretization._output_sample_set.get_values() is None:
        raise ValueError("You must have values to use this method.")
    samples = cluster_discretization._input_sample_set.get_values()
    data = cluster_discretization._output_sample_set.get_values()

    input_dim = cluster_discretization._input_sample_set.get_dim()
    num_model_samples = cluster_discretization.check_nums()
    output_dim = cluster_discretization._output_sample_set.get_dim()

    num_model_samples = cluster_discretization.check_nums()
    input_dim = cluster_discretization._input_sample_set.get_dim()

    num_centers = num_model_samples / (2*input_dim + 1)

    # Find radii_vec from the first cluster of samples
    radii_vec = samples[num_centers:num_centers + input_dim, :] - samples[0, :]
    radii_vec = util.fix_dimensions_vector_2darray(radii_vec.diagonal())

    # Clean the data
    data = util.clean_data(data[num_centers:])
    gradient_tensor = np.zeros([num_centers, output_dim, input_dim])

    radii_vec = np.tile(np.repeat(radii_vec, output_dim, axis=1), [num_centers,
        1])

    # Construct indices for CFD gradient approxiation
    inds = np.repeat(range(0, 2 * input_dim * num_centers, 2 * input_dim),
        input_dim) + np.tile(range(0, input_dim), num_centers)
    inds = np.array([inds, inds+input_dim]).transpose()

    gradient_mat = (data[inds[:, 0]] - data[inds[:, 1]]) * (0.5 / radii_vec)

    # Reshape and organize
    gradient_tensor = np.reshape(gradient_mat.transpose(), [output_dim,
        input_dim, num_centers], order='F').transpose(2, 0, 1)

    if normalize:
        # Compute the norm of each vector
        norm_gradient_tensor = np.linalg.norm(gradient_tensor, ord=1, axis=2)

        # If it is a zero vector (has 0 norm), set norm=1, avoid divide by zero
        norm_gradient_tensor[norm_gradient_tensor == 0] = 1.0

        # Normalize each gradient vector
        gradient_tensor = gradient_tensor/np.tile(norm_gradient_tensor,
            (input_dim, 1, 1)).transpose(1, 2, 0)

    center_input_sample_set = sample.sample_set(input_dim)
    center_input_sample_set.set_values(samples[:num_centers, :])
    if cluster_discretization._input_sample_set.get_domain() is not None:
        center_input_sample_set.set_domain(cluster_discretization.\
                _input_sample_set.get_domain())
    center_input_sample_set.set_jacobians(gradient_tensor)
    center_output_sample_set = sample.sample_set(output_dim)
    center_output_sample_set.set_values(data[:num_centers, :])
    if cluster_discretization._output_sample_set.get_domain() is not None:
        center_output_sample_set.set_domain(cluster_discretization.\
                _output_sample_set.get_domain())
    #center_output_sample_set.set_jacobians(gradient_tensor.transpose())
    center_discretization = sample.discretization(center_input_sample_set,
            center_output_sample_set)
    return center_discretization
