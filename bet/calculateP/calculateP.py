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
    # Determine the appropriate bin zie for this QoI
    bin_size = np.zeros(data.shape[1],)
    for i in range(len(data)):
        bin_size[i] = np.max(data[:,i]) - np.min(data[:,i])

    bin_size *= 1.0/bin_ratio ### fix


    
