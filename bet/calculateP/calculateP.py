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

def prob_emulated(samples,
                  data,
                  rho_D,
                  lam_domain):
    """
    :param samples: The samples in parameter space for which the model was run.
    :param data: The data from running the model given the samples with dimensions nSamples x nQoI.
    :param rho_D:
    :param lam_domain: The domain for each parameter for the model.

    """
    # Determine the appropriate bin size for this QoI
    bin_ratio = 0.2
    data_max = np.max(data, 0)
    data_min = np.min(data, 0)
    bin_size = (data_max-data_min)*bin_ratio

   M = 50 # Defines number M samples in D used to define rho_{D,M}
   # The choice of M is something of an "art" - play around with it
   # and you can get reasonable results with a relatively small
   # number here like 50.


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
    distr_right = np.repeat([true_Q+bin_size*0.5], M, 0).transpose()
    distr_left = np.repeat([true_Q-bin_size*0.5], M, 0).transpose()
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

    # Determine which inputs go to which M bins using the QoI
    #io_ptr = dsearchn(d_distr_samples, data);
    io_ptr = d_Tree.query(data)

    num_l_emulate = 1e7
    P_emulated = np.zeros(num_l_emulate,1)
    lam_width = lam_domain[:,1]-lam_domain[:,0]
    l_emulate = np.repeat([lam_domain[:,0], num_l_emulate,0).transpose()


    
    
