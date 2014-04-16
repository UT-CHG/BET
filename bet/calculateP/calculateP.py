"""
This module provides methods for calulating the probability
measure $P_{\Lambda}$.

* :mod:`~bet.calculateP.prob_emulated` provides a skeleton class and calculates the
    probability for a set of emulation points.
* :mod:`~bet.calculateP.calculateP.prob_samples_ex` calculates the exact volumes of the
    interior voronoi cells and estimates the volumes of the exterior voronoi
    cells by using a set of bounding points
* :mod:`~bet.calculateP.calculateP.prob_samples_mc` estimates the volumes of the voronoi cells
    using MC integration

"""
import numpy as np
import scipy.spatial as spatial

def simple_fun_rho_D(data, true_Q, M=50, bin_ratio=0.2):
    """
    Creates a simple function approximation of rho_{D,M} where rho_{D,M} is a
    uniform probability density centered at true_Q with bin_ratio of the width of D
    using M bins.

    :param int M: Defines number M samples in D used to define rho_{D,M}
        The choice of M is something of an "art" - play around with it
        and you can get reasonable results with a relatively small
        number here like 50.
    :param double bin_ratio: The ratio used to determine the width of the
        uniform distributiion as ``bin_size = (data_max-data_min)*bin_ratio``
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
    distr_right = np.repeat([true_Q+bin_size*.75], M, 0).transpose()
    distr_left = np.repeat([true_Q-bin_size*.75], M, 0).transpose()
    distr_center = (distr_right+distr_left)/2.0
    d_distr_samples = (distr_right-distr_left)
    d_distr_samples = d_distr_samples * np.random.random(distr_right.shape)
    d_distr_samples = d_distr_samples + distr_left

    # Now compute probabilities for rho_{D,M} by sampling from rho_D
    # First generate samples of rho_D - I sometimes call this emulation
    num_d_emulate = 1E6
    distr_right = np.repeat([true_Q+bin_size*0.5], num_d_emulate, 0).transpose()
    distr_left = np.repeat([true_Q-bin_size*0.5], num_d_emulate, 0).transpose()
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


def prob_emulated(samples, data, rho_D_M, d_distr_samples,
        lam_domain, num_l_emulate=1e7, d_Tree=None): 
    """
    Calculates P_{\Lambda}(\mathcal{V}_{\lambda_{emulate}}), the probability
    assoicated with a set of voronoi cells defined by ``num_l_emulate`` iid
    samples (\lambda_{emulate}).

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (ndim, num_samples)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (mdim, M) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (mdim, M) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)
    :param int num_l_emulate: The number of iid samples used to parition the
        parameter space
    :rtype: tuple
    :returns: (P, lambda_emulate, io_ptr, emulate_ptr)

    """

    if d_Tree == None:
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    #io_ptr = dsearchn(d_distr_samples, data);
    io_ptr = d_Tree.query(data)

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
    
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    emulate_ptr = l_Tree.query(lambda_emulate)

    # Apply the standard MC approximation to determine the number of emulated
    # samples per model run sample. This is for approximating 
    # \mu_\Lambda(A_i \intersect b_j)
    lam_vol = np.zeros((num_l_emulate,))
    for i in range(samples.shape[-1]):
        lam_vol[i] = np.sum(np.equal(emulate_ptr, i))
    lam_vol = lam_vol/num_l_emulate

    P = np.zeros((num_l_emulate,))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr, i)
        IItemp = np.equal(emulate_ptr, Itemp)
        P[IItemp] = rho_D_M[i]*lam_vol(IItemp)/sum(lam_vol(IItemp))

    return (P, lambda_emulate, io_ptr, emulate_ptr)








    

    
    
