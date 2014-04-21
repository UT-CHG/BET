import numpy as np
import scipy.spatial as spatial

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
    distr_right = np.repeat([true_Q+bin_size*.75], M, 0)#.transpose()
    distr_left = np.repeat([true_Q-bin_size*.75], M, 0)#.transpose()
    distr_center = (distr_right+distr_left)/2.0
    d_distr_samples = (distr_right-distr_left)
    d_distr_samples = d_distr_samples * np.random.random(distr_right.shape)
    d_distr_samples = d_distr_samples + distr_left

    # Now compute probabilities for rho_{D,M} by sampling from rho_D
    # First generate samples of rho_D - I sometimes call this emulation
    distr_right = np.repeat([true_Q+bin_size*0.5], num_d_emulate, 0)#.transpose()
    distr_left = np.repeat([true_Q-bin_size*0.5], num_d_emulate, 0)#.transpose()
    distr_center = (distr_right+distr_left)/2.0
    d_distr_emulate = (distr_right-distr_left)
    d_distr_emulate = d_distr_emulate * np.random.random(distr_right.shape)
    d_distr_emulate = d_distr_emulate + distr_left

    # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    #k = dsearchn(d_distr_samples, d_distr_emulate)
    d_Tree = spatial.KDTree(d_distr_samples)
    k = d_Tree.query(d_distr_emulate)
    count_neighbors = np.zeros((M,))
    for i in range(M):
        count_neighbors[i] = np.sum(np.equal(k,i))

    # Now define probability of the d_distr_samples
    # This together with d_distr_samples defines rho_{D,M}
    d_distr_prob = count_neighbors / num_d_emulate
    
    # NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    # above, while informed by the sampling of the map Q, do not require
    # solving the model EVER! This can be done "offline" so to speak.
    return (d_distr_prob, d_distr_samples, d_Tree)

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

def hist_normal(data, distr_samples, nbins):
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
