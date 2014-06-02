import numpy as np
import scipy.spatial as spatial
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def unif_unif(data, true_Q, M=50, bin_ratio=0.2, num_d_emulate = 1E6):
    """
    Creates a simple function approximation of rho_{D,M} where rho_{D,M} is a
    uniform probability density centered at true_Q with bin_ratio of the width of D
    using M uniformly spaced bins.

    :param int M: Defines number M samples in D used to define rho_{D,M}
        The choice of M is something of an "art" - play around with it
        and you can get reasonable results with a relatively small
        number here like 50.
    :param double bin_ratio: The ratio used to determine the width of the
        uniform distributiion as ``bin_size = (data_max-data_min)*bin_ratio``
    :param int num_d_emulate: Number of samples used to emulate using an MC assumption
    :param data: Array containing QoI data where the QoI is mdim diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param true_Q: $Q(\lambda_{true})$
    :type true_Q: :class:`~numpy.ndarray` of size (mdim,)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples

    """
    # Determine the appropriate bin size for this QoI
    data_max = np.max(data, 0)
    data_min = np.min(data, 0)
    bin_size = (data_max-data_min)*bin_ratio

    # Create M samples defining M bins in D used to define rho_{D,M}
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
    if rank==0:
        d_distr_samples = 1.5*bin_size*(np.random.random((M,data.shape[1]))-0.5)+true_Q
    else:
        d_distr_samples = None
    d_distr_samples = comm.bcast(d_distr_samples, root=0)

    # Now compute probabilities for rho_{D,M} by sampling from rho_D
    # First generate samples of rho_D - I sometimes call this emulation
    num_d_emulate = int(num_d_emulate/size)+1
    d_distr_emulate = bin_size*(np.random.random((num_d_emulate,data.shape[1]))-0.5)+true_Q

    # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    #k = dsearchn(d_distr_samples, d_distr_emulate)
    d_Tree = spatial.KDTree(d_distr_samples)
    (length,k) = d_Tree.query(d_distr_emulate)
    count_neighbors = np.zeros((M,))
    for i in range(M):
        count_neighbors[i] = np.sum(np.equal(k,i))

    # Now define probability of the d_distr_samples
    # This together with d_distr_samples defines rho_{D,M}
    count_neighbors= comm.allreduce(count_neighbors, count_neighbors, op=MPI.SUM)
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

def gaussian_regular(data, true_Q, std, nbins, num_d_emulate = 1E6):
    pass
    return (d_distr_prob, d_distr_samples, d_Tree)

def multivariate_gaussian(x, mean, std):
    dim = len(mean)
    detDiagCovMatrix = np.sqrt(np.prod(np.diag(std(std))))
    frac = (2.0*np.pi)**(-dim/2.0)  * (1.0/detDiagCovMatrix)
    fprime = x-mean
    return frac*np.exp(-0.5*np.dot(fprime, 1.0/np.diag(std*std)))

def normal_normal(true_Q, M, std, num_d_emulate = 1E6):
    """
    Creates a simple function approximation of rho_{D,M} where rho_{D,M} is a
    multivariate normal probability density centered at true_Q with 
    standard deviation std using M bins sampled from the given normal 
    distribution.

    :param int M: Defines number M samples in D used to define rho_{D,M}
        The choice of M is something of an "art" - play around with it
        and you can get reasonable results with a relatively small
        number here like 50.
 
    :param int num_d_emulate: Number of samples used to emulate using an MC assumption
    :param true_Q: $Q(\lambda_{true})$
    :type true_Q: :class:`~numpy.ndarray` of size (mdim,)
    :param std: The standard deviation of each QoI
    :type std: :class:`~numpy.ndarray` of size (mdim,)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples

    """
    import scipy.stats as stats
    # Create M smaples defining M bins in D used to define rho_{D,M}
    # rho_D is assumed to be a multi-variate normal distribution with mean
    # true_Q and standard deviation std.

    covariance = np.diag(std*std)

    d_distr_samples = np.zeros((M, len(true_Q)))
    if rank ==0:
        for i in range(len(true_Q)):
            d_distr_samples[:,i] = np.random.normal(true_Q[i], std[i], M) 
    d_distr_samples = comm.bcast(d_distr_samples, root=0)

 
    # Now compute probabilities for rho_{D,M} by sampling from rho_D
    # First generate samples of rho_D - I sometimes call this emulation  
    num_d_emulate = int(num_d_emulate/size)+1
    d_distr_emulate = np.zeros((num_d_emulate, len(true_Q)))
    for i in range(len(true_Q)):
        d_distr_emulate[:,i] = np.random.normal(true_Q[i], std[i], num_d_emulate) 

    # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    d_Tree = spatial.KDTree(d_distr_samples)
    (length,k) = d_Tree.query(d_distr_emulate)
    count_neighbors = np.zeros((M,))
    volumes = np.zeros((M,))
    for i in range(M):
        Itemp = np.equal(k,i)
        count_neighbors[i] = np.sum(Itemp)
        volumes[i] = np.sum(1.0/stats.multivariate_normal.pdf(d_distr_emulate[Itemp,:], true_Q, covariance))
    # Now define probability of the d_distr_samples
    # This together with d_distr_samples defines rho_{D,M}
    count_neighbors= comm.allreduce(count_neighbors, count_neighbors, op=MPI.SUM)
    volumes = comm.allreduce(volumes, volumes, op=MPI.SUM)
    rho_D_M = count_neighbors*volumes 
    rho_D_M = rho_D_M/np.sum(rho_D_M)
    
    # NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    # above, while informed by the sampling of the map Q, do not require
    # solving the model EVER! This can be done "offline" so to speak.
    return (rho_D_M, d_distr_samples, d_Tree)

def unif_normal(true_Q, M, std, num_d_emulate = 1E6):
    """
    Creates a simple function approximation of rho_{D,M} where rho_{D,M} is a
    multivariate normal probability density centered at true_Q with 
    standard deviation std using M bins sampled from a uniform distribution
    with a size 4 standard deviations in each direction.

    :param int M: Defines number M samples in D used to define rho_{D,M}
        The choice of M is something of an "art" - play around with it
        and you can get reasonable results with a relatively small
        number here like 50.
 
    :param int num_d_emulate: Number of samples used to emulate using an MC assumption
    :param true_Q: $Q(\lambda_{true})$
    :type true_Q: :class:`~numpy.ndarray` of size (mdim,)
    :param std: The standard deviation of each QoI
    :type std: :class:`~numpy.ndarray` of size (mdim,)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples

    """
    import scipy.stats as stats
    # Create M smaples defining M bins in D used to define rho_{D,M}
    # rho_D is assumed to be a multi-variate normal distribution with mean
    # true_Q and standard deviation std.

    bin_size=4.0*std
    d_distr_samples = np.zeros((M, len(true_Q)))
    if rank ==0:
        d_distr_samples = bin_size*(np.random.random((M,len(true_Q)))-0.5)+true_Q
    d_distr_samples = comm.bcast(d_distr_samples, root=0)

 
    # Now compute probabilities for rho_{D,M} by sampling from rho_D
    # First generate samples of rho_D - I sometimes call this emulation  
    num_d_emulate = int(num_d_emulate/size)+1
    d_distr_emulate = np.zeros((num_d_emulate, len(true_Q)))
    for i in range(len(true_Q)):
        d_distr_emulate[:,i] = np.random.normal(true_Q[i], std[i], num_d_emulate) 

    # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    d_Tree = spatial.KDTree(d_distr_samples)
    (length,k) = d_Tree.query(d_distr_emulate)
    count_neighbors = np.zeros((M,))
    volumes = np.zeros((M,))
    for i in range(M):
        Itemp = np.equal(k,i)
        count_neighbors[i] = np.sum(Itemp)
        
    # Now define probability of the d_distr_samples
    # This together with d_distr_samples defines rho_{D,M}
    count_neighbors= comm.allreduce(count_neighbors, count_neighbors, op=MPI.SUM)
    rho_D_M = count_neighbors/(size*num_d_emulate)
    
    # NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    # above, while informed by the sampling of the map Q, do not require
    # solving the model EVER! This can be done "offline" so to speak.
    return (rho_D_M, d_distr_samples, d_Tree)

def gaussian_unif(data, true_Q, std, nbins, num_d_emulate = 1E6):
    pass
    return (d_distr_prob, d_distr_samples, d_Tree)

def uniform_hyperrectangle(data, Q_true, bin_ratio):
    """
    Creates a simple function approximation of rho_{D,M} where rho_{D,M} is a
    uniform probability density centered at true_Q with bin_ratio of the width
    of D.

    Since rho_D is a uniform distribution on a hyperrectanlge we should be able
    to represent it exactly with ``M = 3^{mdim}`` or rather
    ``len(d_distr_samples) == mdim``.

    :param double bin_ratio: The ratio used to determine the width of the
        uniform distributiion as ``bin_size = (data_max-data_min)*bin_ratio``
    :param int num_d_emulate: Number of samples used to emulate using an MC assumption
    :param data: Array containing QoI data where the QoI is mdim diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param true_Q: $Q(\lambda_{true})$
    :type true_Q: :class:`~numpy.ndarray` of size (mdim,)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples
    """
    pass
    return (d_distr_prob, d_distr_samples, d_Tree)

def uniform_datadomain(data_domain):
    """
    Creates a simple function approximation of rho_{D,M} where rho_{D,M} is a
    uniform probability density over the entire ``data_domain``. Since rho_D is
    a uniform distribution on a hyperrectanlge we should be able to represent
    it exactly with ``M = 1`` or rather ``len(d_distr_samples) == 1``.
    
    :param data_domain: The domain for each QoI of the model.
    :type data_domain: :class:`numpy.ndarray` of shape (2, mdim)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples
    """
    pass
    return (d_distr_prob, d_distr_samples, d_Tree)

def uniform_data_minmax(data):
    """
    Creates a simple function approximation of rho_{D,M} where rho_{D,M} is a
    uniform probability density over the entire ``data_domain``. Here the
    ``data_domain`` is the hyperrectangle defined by minima and maxima of the
    ``data`` in each dimension. Since rho_D is a uniform distribution on a
    hyperrectanlge we should be able to represent it exactly with ``M = 1`` or
    rather ``len(d_distr_samples) == 1``.
    
    :param data: Array containing QoI data where the QoI is mdim diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples
    """
    pass
    return uniform_datadomain(data_domain)

def uniform_data(data):
    """
    Creates a simple function approximation of rho_{D,M} where rho_{D,M} is a
    uniform probability density over the entire ``data_domain``. Here the
    ``data_domain`` is the union of voronoi cells defined by ``data``. In other
    words we assign each sample the same probability, so ``M = len(data)`` or
    rather ``len(d_distr_samples) == len(data)``. The purpose of this method is
    to approximate uniform distributions over irregularly shaped domains.
    
    :param data: Array containing QoI data where the QoI is mdim diminsional
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :rtype: tuple
    :returns: (rho_D_M, d_distr_samples, d_Tree) where ``rho_D_M`` and
    ``d_distr_samples`` are (mdim, M) :class:`~numpy.ndarray` and `d_Tree` is
    the :class:`~scipy.spatial.KDTree` for d_distr_samples
    """
    pass
    return (d_distr_prob, d_distr_samples, d_Tree)
