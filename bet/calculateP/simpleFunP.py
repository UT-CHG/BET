"""
This module provides methods for creating simple funciton approximations to be
used by :mod:`~bet.calculateP.calculateP`.
"""
from bet.Comm import *
import numpy as np
import scipy.spatial as spatial
import bet.calculateP.voronoiHistogram as vHist

def unif_unif(data, Q_ref, M=50, bin_ratio=0.2, num_d_emulate=1E6):
    """
    Creates a simple function approximation of :math:`\\rho_{\\mathcal{D}}`
    where :math:`\\rho_{\\mathcal{D}}` is a uniform probability density on
    a generalized rectangle centered at Q_ref.
    The support of this density is defined by bin_ratio, which determines
    the size of the generalized rectangle by scaling the circumscribing 
    generalized rectangle of :math:`\\mathcal{D}`.
    The simple function approximation is then defined by determining M 
    Voronoi cells (i.e., "bins") partitioning :math:`\\mathcal{D}`. These
    bins are only implicitly defined by M samples in :math:`\\mathcal{D}`.
    Finally, the probabilities of each of these bins is computed by 
    sampling from :math:`\\rho{\\mathcal{D}}` and using nearest neighbor
    searches to bin these samples in the M implicitly defined bins. 
    The result is the simple function approximation denoted by
    :math:`\\rho_{\\mathcal{D},M}`.
    
    Note that all computations in the measure-theoretic framework that
    follow from this are for the fixed simple function approximation
    :math:`\\rho_{\\mathcal{D},M}`.

    :param int M: Defines number M samples in D used to define
        :math:`\\rho_{\\mathcal{D},M}` The choice of M is something of an "art" -
        play around with it and you can get reasonable results with a
        relatively small number here like 50.
    :param bin_ratio: The ratio used to determine the width of the
        uniform distributiion as ``bin_size = (data_max-data_min)*bin_ratio``
    :type bin_ratio: double or list()
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption 
    :param data: Array containing QoI data where the QoI is mdim
        diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param Q_ref: $Q(lambda_reference})$
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples
    """
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)   
    # Determine the appropriate bin size for this QoI
    data_max = np.max(data, 0)
    data_min = np.min(data, 0)
    bin_size = (data_max-data_min)*bin_ratio

    '''
    Create M samples defining M Voronoi cells (i.e., "bins") in D used to 
    define the simple function approximation :math:`\\rho_{\\mathcal{D},M}`.
    
    This does not have to be random, but here we assume this to be the case.
    We can choose these samples deterministically but that fails to scale 
    with dimension efficiently.
    
    Note that these M samples are chosen for the sole purpose of determining
    the bins used to create the approximation to :math:`\\rho_{\\mathcal{D}}`.
    
    We call these M samples "d_distr_samples" because they are samples
    on the data space and the distr implies these samples are chosen
    to create the approximation to the probability measure (distribution) 
    on D.
    
    Note that we create these samples in a set containing the hyperrectangle
    in order to get output cells with zero probability. If all of the
    d_dstr_samples were taken from within the support of :math:`\\rho_{\\mathcal{D}}`
    then each of the M bins would have positive probability. This would
    in turn imply that the support of :math:`\\rho_{\\Lambda}` is all of
    :math:`\\Lambda`.
    '''
    if rank == 0:
        d_distr_samples = 1.5*bin_size*(np.random.random((M,
            data.shape[1]))-0.5)+Q_ref 
    else:
        d_distr_samples = np.empty((M, data.shape[1]))
    comm.Bcast([d_distr_samples, MPI.DOUBLE], root=0)

    '''
    Compute probabilities in the M bins used to define
    :math:`\\rho_{\\mathcal{D},M}` by Monte Carlo approximations
    that in this context amount to binning with nearest neighbor
    approximations the num_d_emulate samples taken from
    :math:`\\rho_{\\mathcal{D}}`.
    '''
    # Generate the samples from :math:`\rho_{\mathcal{D}}`
    num_d_emulate = int(num_d_emulate/size)+1
    d_distr_emulate = bin_size*(np.random.random((num_d_emulate,
        data.shape[1]))-0.5) + Q_ref

    # Bin these samples using nearest neighbor searches
    d_Tree = spatial.KDTree(d_distr_samples)
    (_, k) = d_Tree.query(d_distr_emulate)
    count_neighbors = np.zeros((M,), dtype=np.int)
    for i in range(M):
        count_neighbors[i] = np.sum(np.equal(k, i))

    # Use the binning to define :math:`\\rho_{\\mathcal{D},M}`
    ccount_neighbors = np.copy(count_neighbors)
    comm.Allreduce([count_neighbors, MPI.INT], [ccount_neighbors, MPI.INT],
            op=MPI.SUM)
    count_neighbors = ccount_neighbors
    rho_D_M = count_neighbors.astype(np.float64) / float(num_d_emulate*size)

    '''
    NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    above, while possibly informed by the sampling of the map Q, do not require
    solving the model EVER! This can be done "offline" so to speak. The results
    can then be stored and accessed later by the algorithm using a completely
    different set of parameter samples and model solves.
    '''
    return (rho_D_M, d_distr_samples, d_Tree)

def hist_regular(data, distr_samples, nbins):
    """
    create nbins regulary spaced bins
    check to make sure  each bin has about 1 data sample per bin, if not
    recompute bins
    (hist, edges) = histdd(distr_samples, bins)
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogramdd.html#numpy.histogramdd
    determine d_distr_samples from edges
    """
    pass

def hist_gaussian(data, distr_samples, nbins):
    """
    determine mean, standard deviation of distr_samples
    partition D into nbins of equal probability for N(mean, sigma)
    check to make sure  each bin has about 1 data sample per bin, if not
    recompute bins
    (hist, edges) = histdd(distr_samples, bins)
    determine d_distr_samples from edges
    """
    pass

def hist_unif(data, distr_samples, nbins):
    """
    same as hist_regular bit with uniformly spaced bins
    unif_unif can and should call this function
    """
    pass

def gaussian_regular(data, Q_ref, std, nbins, num_d_emulate=1E6):
    pass
    #return (d_distr_prob, d_distr_samples, d_Tree)

def multivariate_gaussian(x, mean, std):
    dim = len(mean)
    detDiagCovMatrix = np.sqrt(np.prod(np.diag(std(std))))
    frac = (2.0*np.pi)**(-dim/2.0)  * (1.0/detDiagCovMatrix)
    fprime = x-mean
    return frac*np.exp(-0.5*np.dot(fprime, 1.0/np.diag(std*std)))

def normal_normal(Q_ref, M, std, num_d_emulate=1E6):
    """
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}` where :math:`\rho_{\mathcal{D},M}` is a
    multivariate normal probability density centered at Q_ref with 
    standard deviation std using M bins sampled from the given normal 
    distribution.

    :param int M: Defines number M samples in D used to define :math:`\rho_{\mathcal{D},M}`
        The choice of M is something of an "art" - play around with it
        and you can get reasonable results with a relatively small
        number here like 50.
 
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption 
    :param Q_ref: $Q(lambda_{reference})$
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :param std: The standard deviation of each QoI
    :type std: :class:`~numpy.ndarray` of size (mdim,)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples

    """
    import scipy.stats as stats
    # Create M smaples defining M bins in D used to define :math:`\rho_{\mathcal{D},M}`
    # rho_D is assumed to be a multi-variate normal distribution with mean
    # Q_ref and standard deviation std.

    covariance = np.diag(std*std)

    d_distr_samples = np.zeros((M, len(Q_ref)))
    if rank == 0:
        for i in range(len(Q_ref)):
            d_distr_samples[:, i] = np.random.normal(Q_ref[i], std[i], M) 
    comm.Bcast([d_distr_samples, MPI.DOUBLE], root=0)

 
    # Now compute probabilities for :math:`\rho_{\mathcal{D},M}` by sampling from rho_D
    # First generate samples of rho_D - I sometimes call this emulation  
    num_d_emulate = int(num_d_emulate/size)+1
    d_distr_emulate = np.zeros((num_d_emulate, len(Q_ref)))
    for i in range(len(Q_ref)):
        d_distr_emulate[:, i] = np.random.normal(Q_ref[i], std[i],
                num_d_emulate) 

    # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)

    d_Tree = spatial.KDTree(d_distr_samples)
    (_, k) = d_Tree.query(d_distr_emulate)
    count_neighbors = np.zeros((M,), dtype=np.int)
    volumes = np.zeros((M,))
    for i in range(M):
        Itemp = np.equal(k, i)
        count_neighbors[i] = np.sum(Itemp)
        volumes[i] = np.sum(1.0/stats.multivariate_normal.pdf(d_distr_emulate[Itemp, 
            :], Q_ref, covariance))
    # Now define probability of the d_distr_samples
    # This together with d_distr_samples defines :math:`\rho_{\mathcal{D},M}`
    ccount_neighbors = np.copy(count_neighbors)
    comm.Allreduce([count_neighbors, MPI.INT], [ccount_neighbors, MPI.INT],
            op=MPI.SUM)
    count_neighbors = ccount_neighbors
    cvolumes = np.copy(volumes)
    comm.Allreduce([volumes, MPI.DOUBLE], [cvolumes, MPI.DOUBLE], op=MPI.SUM)
    volumes = cvolumes
    rho_D_M = count_neighbors.astype(np.float64)*volumes 
    rho_D_M = rho_D_M/np.sum(rho_D_M)
    
    # NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    # above, while informed by the sampling of the map Q, do not require
    # solving the model EVER! This can be done "offline" so to speak.
    return (rho_D_M, d_distr_samples, d_Tree)

def unif_normal(Q_ref, M, std, num_d_emulate=1E6):
    """
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}` where :math:`\rho_{\mathcal{D},M}` is a
    multivariate normal probability density centered at Q_ref with 
    standard deviation std using M bins sampled from a uniform distribution
    with a size 4 standard deviations in each direction.

    :param int M: Defines number M samples in D used to define :math:`\rho_{\mathcal{D},M}`
        The choice of M is something of an "art" - play around with it
        and you can get reasonable results with a relatively small
        number here like 50.
 
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption 
    :param Q_ref: $Q(lambda_{reference})$
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :param std: The standard deviation of each QoI
    :type std: :class:`~numpy.ndarray` of size (mdim,)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples

    """
    import scipy.stats as stats
    # Create M smaples defining M bins in D used to define :math:`\rho_{\mathcal{D},M}`
    # rho_D is assumed to be a multi-variate normal distribution with mean
    # Q_ref and standard deviation std.

    bin_size = 4.0*std
    d_distr_samples = np.zeros((M, len(Q_ref)))
    if rank == 0:
        d_distr_samples = bin_size*(np.random.random((M, 
            len(Q_ref)))-0.5)+Q_ref
    comm.Bcast([d_distr_samples, MPI.DOUBLE], root=0)

 
    # Now compute probabilities for :math:`\rho_{\mathcal{D},M}` by sampling from rho_D
    # First generate samples of rho_D - I sometimes call this emulation  
    num_d_emulate = int(num_d_emulate/size)+1
    d_distr_emulate = np.zeros((num_d_emulate, len(Q_ref)))
    for i in range(len(Q_ref)):
        d_distr_emulate[:, i] = np.random.normal(Q_ref[i], std[i], 
                num_d_emulate) 

    # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)

    d_Tree = spatial.KDTree(d_distr_samples)
    (_, k) = d_Tree.query(d_distr_emulate)
    count_neighbors = np.zeros((M,), dtype=np.int)
    #volumes = np.zeros((M,))
    for i in range(M):
        Itemp = np.equal(k, i)
        count_neighbors[i] = np.sum(Itemp)
        
    # Now define probability of the d_distr_samples
    # This together with d_distr_samples defines :math:`\rho_{\mathcal{D},M}`
    ccount_neighbors = np.copy(count_neighbors)
    comm.Allreduce([count_neighbors, MPI.INT], [ccount_neighbors, MPI.INT],
            op=MPI.SUM) 
    count_neighbors = ccount_neighbors
    rho_D_M = count_neighbors.astype(np.float64)/float(size*num_d_emulate)
    
    # NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    # above, while informed by the sampling of the map Q, do not require
    # solving the model EVER! This can be done "offline" so to speak.
    return (rho_D_M, d_distr_samples, d_Tree)

def gaussian_unif(data, Q_ref, std, nbins, num_d_emulate=1E6):
    pass
    #return (d_distr_prob, d_distr_samples, d_Tree)

def uniform_hyperrectangle_user(data, domain, center_pts_per_edge=1):
    """
    Creates a simple funciton appoximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho{\mathcal{D}, M}` is a uniform probablity density over the
    hyperrectangular domain specified by domain.

    Since :math:`\rho_\mathcal{D}` is a uniform distribution on a
    hyperrectangle we should we able to represent it exactly with
    :math:`M=3^{m}` where m is the dimension of the data space or rather
    ``len(d_distr_samples) == 3**mdim`.

    :param data: Array containing QoI data where the QoI is mdim diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param domain: The domain overwhich :math:`\rho_\mathcal{D}` is
        uniform.
    :type domain: :class:`numpy.ndarray` of shape (2, mdim)
    :param list() center_pts_per_edge: number of center points per edge and
        additional two points will be added to create the bounding layer

    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
        ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree`
        is the :class:`~scipy.spatial.KDTree` for d_distr_samples
    """
    # determine the center of the domain
    if len(domain.shape) == 1:
        domain = np.expand_dims(domain, axis=1)
    domain_center = np.mean(domain, 0)
    domain_min = np.min(domain, 0)
    domain_max = np.max(domain, 0)
    domain_lengths = domain_max - domain_min
   
    # determine the ratio of the lengths of the domain to the lengths of the
    # hyperrectangle containing the data
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    data_max = np.max(data, 0)
    data_min = np.min(data, 0)
    data_lengths = data_max - data_min
    bin_ratios = domain_lengths/data_lengths

    return uniform_hyperrectangle(data, domain_center, bin_ratios,
            center_pts_per_edge)

def uniform_hyperrectangle_binsize(data, Q_ref, bin_size, center_pts_per_edge=1):
    """
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho_{\mathcal{D},M}` is a uniform probability density
    centered at Q_ref with bin_size of the width
    of D.

    Since rho_D is a uniform distribution on a hyperrectanlge we should be able
    to represent it exactly with ``M = 3^mdim`` or rather
    ``len(d_distr_samples) == 3^mdim``.

    :param bin_size: The size used to determine the width of the
        uniform distribution 
    :type bin_size: double or list()
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption 
    :param data: Array containing QoI data where the QoI is mdim diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param Q_ref: $Q(lambda_{reference})$
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :param list() center_pts_per_edge: number of center points per edge and
        additional two points will be added to create the bounding layer

    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
        ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree`
        is the :class:`~scipy.spatial.KDTree` for d_distr_samples

    """
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    data_max = np.max(data, 0)
    data_min = np.min(data, 0)

    sur_domain = np.zeros((data.shape[1], 2))
    sur_domain[:, 0] = data_min
    sur_domain[:, 1] = data_max
    points, _, rect_domain = vHist.center_and_layer1_points_binsize(center_pts_per_edge, 
            Q_ref, bin_size, sur_domain)
    edges = vHist.edges_regular_binsize(center_pts_per_edge, Q_ref, bin_size,
            sur_domain) 
    _, volumes, _ = vHist.histogramdd_volumes(edges, points)
    return vHist.simple_fun_uniform(points, volumes, rect_domain)

def uniform_hyperrectangle(data, Q_ref, bin_ratio, center_pts_per_edge=1):
    """
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho_{\mathcal{D},M}` is a uniform probability density
    centered at Q_ref with bin_ratio of the width
    of D.

    Since rho_D is a uniform distribution on a hyperrectanlge we should be able
    to represent it exactly with ``M = 3^mdim`` or rather
    ``len(d_distr_samples) == 3^mdim``.

    :param bin_ratio: The ratio used to determine the width of the
        uniform distributiion as ``bin_size = (data_max-data_min)*bin_ratio``
    :type bin_ratio: double or list()
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption 
    :param data: Array containing QoI data where the QoI is mdim diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param Q_ref: $Q(lambda_{reference})$
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :param list() center_pts_per_edge: number of center points per edge and
        additional two points will be added to create the bounding layer

    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
        ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree`
        is the :class:`~scipy.spatial.KDTree` for d_distr_samples

    """
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    data_max = np.max(data, 0)
    data_min = np.min(data, 0)

    # TO-DO: Check for inputted center_pts_per_edge in case given as list
    # or as numpy array to see if dimensions match data space dimensions and
    # that positive integer values are being used. Also, create this change
    # elsewhere since center_pts_per_edge is only a scalar if dim(D)=1.
    if not isinstance(center_pts_per_edge, np.ndarray):
        center_pts_per_edge = np.ones((data.shape[1])) * center_pts_per_edge
    else:
        if not len(center_pts_per_edge) == data.shape[1]:
            center_pts_per_edge = np.ones((data.shape[1]))
            print 'Warning: center_pts_per_edge dimension mismatch. Using 1 in each dimension.'

    sur_domain = np.zeros((data.shape[1], 2))
    sur_domain[:, 0] = data_min
    sur_domain[:, 1] = data_max
    points, _, rect_domain = vHist.center_and_layer1_points(center_pts_per_edge, 
            Q_ref, bin_ratio, sur_domain)
    edges = vHist.edges_regular(center_pts_per_edge, Q_ref, bin_ratio,
            sur_domain) 
    _, volumes, _ = vHist.histogramdd_volumes(edges, points)
    return vHist.simple_fun_uniform(points, volumes, rect_domain)

def uniform_data(data):
    """
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho_{\mathcal{D},M}` is a uniform probability density over
    the entire ``data_domain``. Here the ``data_domain`` is the union of
    voronoi cells defined by ``data``. In other words we assign each sample the
    same probability, so ``M = len(data)`` or rather ``len(d_distr_samples) ==
    len(data)``. The purpose of this method is to approximate uniform
    distributions over irregularly shaped domains.
    
    :param data: Array containing QoI data where the QoI is mdim diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param list() center_pts_per_edge: number of center points per edge and
        additional two points will be added to create the bounding layer

    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
        ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree`
        is the :class:`~scipy.spatial.KDTree` for d_distr_samples
    """
    d_distr_prob = np.ones((data.shape[1],))
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    d_Tree = spatial.KDTree(data)
    return (d_distr_prob, data, d_Tree)
