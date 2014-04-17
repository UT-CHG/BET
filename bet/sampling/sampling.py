def emulate_iid_lebesgue(lam_domain, num_l_emulate=1e7):
    """
    Generates ``num_l_emulate`` samples in ``lam_domain`` assuming a Lebesgue
    measure.

    Note: This function is designed only for generalized rectangles.

    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)
    :param int num_l_emulate: The number of iid samples to generate.
    :rtype lambda_emulate: :class:`~numpy.ndarray` of shape (ndim, num_l_emulate)
    :returns lambda_emulate: Samples used to partition the parameter space

    """
    # Parition the space using emulated samples into many voronoi cells
    # These samples are iid so that we can apply the standard MC
    # assumuption/approximation.
    lam_width = lam_domain[:,1]-lam_domain[:,0]
    lambda_left = np.repeat([lam_domain[:,0]], num_l_emulate,0).transpose()
    lambda_right = np.repeat([lam_domain[:,1]], num_l_emulate,0).transpose()
    l_center = (lambda_right+lambda_left)/2.0
    lambda_emulate = (lambda_right-lambda_left)
    lambda_emulate = lambda_emulate * np.random.random(lambda_emulate.shape)
    lambda_emulate = lambda_emulate + lambda_left
    return lambda_emulate

def regular_grid_lebesgue(lam_domain, num_per_dim):
    """
    Generates regular grid of samples assuming a Lebesgue measure. There are
    many ways of doing this using Numpy, so this is an unimplemented
    placeholder.

    Note: This function is designed only for generalized rectangles.

    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)
    :param num_per_dim: List of number of samples to generate per dimension for
        a total of prod(num_per_dim) samples
    :rtype lambda_emulate: :class:`~numpy.ndarray` of shape (ndim,
        prod(num_er_dim))
    :returns lambda_emulate: Samples used to partition the parameter space

    """
    pass
