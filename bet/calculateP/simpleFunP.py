"""
This module provides methods for creating simple funciton approximations to be
used by :mod:`~bet.calculateP.calculateP`.
"""
from bet.vis.Comm import *
import numpy as np
import scipy.spatial as spatial
import bet.calculateP.voronoiHistogram as vHist

def unif_unif(data, Q_ref, M=50, bin_ratio=0.2, num_d_emulate=1E6):
    """
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}` where :math:`\rho_{\mathcal{D},M}` is a
    uniform probability density centered at Q_ref with bin_ratio of the width
    of D using M uniformly spaced bins.

    :param int M: Defines number M samples in D used to define :math:`\rho_{\mathcal{D},M}`
        The choice of M is something of an "art" - play around with it
        and you can get reasonable results with a relatively small
        number here like 50.
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

    # Create M samples defining M bins in D used to define :math:`\rho_{\mathcal{D},M}`
    # This choice of rho_D was based on looking at Q(Lambda) and getting a
    # sense of what a reasonable amount of error was for this problem. Notice I
    # use uniform distributions of various lengths depending on which
    # measurement I chose from the 4 I made above. Also, this does not have to
    # be random. I can choose these bins deterministically but that doesn't
    # scale well typically. These are also just chosen to determine bins and do
    # not necessarily have anything to do with probabilities. I use "smaller"
    # uniform densities below. This was just to setup a discretization of D in
    # some random way so that I put bins near where the output probability is
    # (why would I care about binning zero probability events?).
    if rank == 0:
        d_distr_samples = 1.5*bin_size*(np.random.random((M,
            data.shape[1]))-0.5)+Q_ref 
    else:
        d_distr_samples = np.empty((M, data.shape[1]))
    comm.Bcast([d_distr_samples, MPI.DOUBLE], root=0)

    # Now compute probabilities for :math:`\rho_{\mathcal{D},M}` by sampling from rho_D
    # First generate samples of rho_D - I sometimes call this emulation
    num_d_emulate = int(num_d_emulate/size)+1
    d_distr_emulate = bin_size*(np.random.random((num_d_emulate,
        data.shape[1]))-0.5) + Q_ref

    # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    #k = dsearchn(d_distr_samples, d_distr_emulate)
    d_Tree = spatial.KDTree(d_distr_samples)
    (_, k) = d_Tree.query(d_distr_emulate)
    count_neighbors = np.zeros((M,), dtype=np.float64)
    for i in range(M):
        count_neighbors[i] = np.sum(np.equal(k, i))

    # Now define probability of the d_distr_samples
    # This together with d_distr_samples defines :math:`\rho_{\mathcal{D},M}`
    ccount_neighbors = np.copy(count_neighbors)
    comm.Allreduce([count_neighbors, MPI.INT], [ccount_neighbors, MPI.INT],
            op=MPI.SUM)
    count_neighbors = ccount_neighbors
    rho_D_M = count_neighbors / (num_d_emulate*size)
    
    # NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    # above, while informed by the sampling of the map Q, do not require
    # solving the model EVER! This can be done "offline" so to speak.
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
    count_neighbors = np.zeros((M,))
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
    rho_D_M = count_neighbors*volumes 
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
    count_neighbors = np.zeros((M,))
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
    rho_D_M = count_neighbors/(size*num_d_emulate)
    
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
