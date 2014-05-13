"""
This module provides methods for plotting probabilities. 
"""

import matploblib.pyplot as plt

def plot_voronoi_probs(P_samples,
                       samples,
                       lam_domain,
                       nbins=20,
                       post_process=False):
    """
    This makes plots of every pair of marginals (or joint in 2d case) of
    input probability measure defined by P_samples post_process - 
    is an input that only applies to the 2d case if you want
    to see marginals on a regular grid instead of w.r.t. the Voronoi cells.

    :param P_samples: Probabilities.
    :type P_samples: :class:`~numpy.ndarray` of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)
    :param nbins: Number of bins in each direction.
    :type nbins: :int

    """
    lam_dim=lam_domain.shape[0]
    
    if lam_dim == 2: # Plot Voronoi tesselations, otherwise plot 2d 
        #projections/marginals of the joint inverse measure
        num_samples = samples.shape[0]
        #Add fake samples outside of lam_domain to close Voronoi 
        #tesselations at infinity
        midpt = 
        
    
