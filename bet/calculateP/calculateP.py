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
import pyhull

def prob_emulated(samples, data, rho_D_M, d_distr_samples,
        lambda_emulate=None, d_Tree=None): 
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
    :param lambda_emulate: Samples used to partition the parameter space
    :type lambda_emulate: :class:`~numpy.ndarray` of shape (ndim, num_l_emulate)
    :rtype: tuple
    :returns: (P, lambda_emulate, io_ptr, emulate_ptr, lam_vol)

    """
    
    if lambda_emulate == None:
        lambda_emulate = samples
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

    # Calculate Probabilties
    P = np.zeros((num_l_emulate,))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr, i)
        if np.any(Itemp):
            IItemp = np.equal(emulate_ptr, Itemp)
            if np.any(IItemp):
                P[IItemp] = rho_D_M[i]/np.sum(IItemp)

    return (P, lambda_emulate, io_ptr, emulate_ptr)

def prob(samples, data, rho_D_M, d_distr_samples, lam_domain, d_Tree=None): 
    """
    Calculates P_{\Lambda}(\mathcal{V}_{\lambda_{samples}}), the probability
    assoicated with a set of voronoi cells defined by the model solves at
    $\lambda_{samples}$ where the volumes of these voronoi cells are assumed to
    be equal under the MC assumption.

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
    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr) where P is the
        probability associated with samples, lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins, and emulate_ptr
        a pointer from emulated samples to samples (in parameter space)

    """
    # Calculate pointers and volumes
    # DO NOT CALL HERE WITH NUM_L_EMULATE(?)
    (P, lambda_emulate, io_ptr, emulate_ptr) = prob_emulated(samples,
            data, rho_D_M, d_distr_samples, lam_domain, num_l_emulate,
            d_Tree)
    
    # Apply the standard MC approximation 
    lam_vol = np.ones((samples.shape[-1],))

    # Calculate Probabilities
    P = np.zeros((samples.shape[-1],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr, i)
        if np.any(Itemp):
            P[Itemp] = rho_D_M[i]*lam_vol(Itemp)/np.sum(lam_vol(Itemp))

    return (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr)

def prob_samples_qhull(samples, data, rho_D_M, d_distr_samples,
        lam_domain, d_Tree=None): 
    """
    Calculates P_{\Lambda}(\mathcal{V}_{\lambda_{emulate}}), the probability
    assoicated with a set of voronoi cells defined by ``num_l_emulate`` iid
    samples (\lambda_{emulate}).

    This method is only intended when ``lam_domain`` is a generalized rectangle.

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
    :returns: (P, io_ptr, lam_vol)

    """

    if d_Tree == None:
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    #io_ptr = dsearchn(d_distr_samples, data);
    io_ptr = d_Tree.query(data)

    # Calcuate the bounding region for the parameters
    lam_bound = np.copy(samples)
    lam_width = lam_domain[:,1]-lam_domain[:,0]
    nbins = d_distr_samples.shape[1]
    # Add fake samples outside of lam_domain to close Voronoi tesselations.
    pts_per_edge = nbins
    sides = np.zeros((2,pts_per_edge))
    for i in range(lam_domain.shape[0]):
        sides[i,:] = np.linspace(lam_domain[i,0], lam_domain[i,1], pts_per_edge)
    # add midpoints
    for i in range(lam_domain.shape[0]):
        new_pt = sides
        new_pt[i,:] = np.repeat(lam_domain[i,0]-lam_width[i]/pts_per_edge,pts_per_edge,0).transpose() 
        lam_bound = np.vstack((lam_bound,new_pt))
        new_pt = sides
        new_pt[i,:] = np.repeat(lam_domain[i,1]-lam_width[i]/pts_per_edge,pts_per_edge,0).transpose() 
        lam_bound = np.vstack((lam_bound,new_pt))
        
    # add corners
    corners = np.zeros((2**lam_domain.shape[0], lam_domain.shape[0]))
    for i in range(lam_domain.shape[0]):
        corners[i,:] = lam_domain[i,np.repeat(np.hstack((np.ones((1,2**(i-1))),2*np.ones((1,2**(i-1))))), 2**(lam_domain.shape[0]-i),0).tranpose()]
        corners[i,:] +=lam_width[i]*np.repeat(np.hstack((np.ones((1,2**(i-1))),-np.ones((1,2**(i-1))))), 2**(lam_domain.shape[0]-i)/pts_per_edge,0).transpose()

    lam_bound = np.vstack((lam_bound, corners))
    
    # Calculate the Voronoi diagram for samples. Calculate the volumes of 
    # the convex hulls of the corresponding Voronoi regions.
    lam_vol = np.zeros((samples.shape[-1],))
    for i in range((samples.shape[-1])):
        vornoi = spatial.Voronoi(lam_bound)
        lam_vol[i] = float(pyhull.qconvex('Qt FA', vornoi.vertices).split()[-1])
    
    # Calculate probabilities.
    P = np.zeros((samples.shape[-1],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr, i)
        P[Itemp] = rho_D_M[i]*lam_vol(Itemp)/np.sum(lam_vol(Itemp))
    P = P/np.sum(P)

    return (P, io_ptr, lam_vol)

def prob_samples_mc(samples, data, rho_D_M, d_distr_samples,
        lam_domain, num_l_emulate=1e7, d_Tree=None): 
    """
    Calculates P_{\Lambda}(\mathcal{V}_{\lambda_{samples}}), the probability
    assoicated with a set of voronoi cells defined by the model solves at
    $\lambda_{samples}$ where the volumes of these voronoi cells are
    approximated using MC integration.

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
    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr) where P is the
        probability associated with samples, lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins, and emulate_ptr
        a pointer from emulated samples to samples (in parameter space)

    """
    # Calculate pointers and volumes
    (P, lambda_emulate, io_ptr, emulate_ptr) = prob_emulated(samples,
            data, rho_D_M, d_distr_samples, lam_domain, num_l_emulate,
            d_Tree)
    
    # Apply the standard MC approximation to determine the number of emulated
    # samples per model run sample. This is for approximating 
    # \mu_\Lambda(A_i \intersect b_j)
    lam_vol = np.zeros((num_l_emulate,))
    for i in range(samples.shape[-1]):
        lam_vol[i] = np.sum(np.equal(emulate_ptr, i))
    lam_vol = lam_vol/num_l_emulate

    # Calculate Probabilities
    P = np.zeros((samples.shape[-1],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr, i)
        if np.any(Itemp):
            P[Itemp] = rho_D_M[i]*lam_vol(Itemp)/np.sum(lam_vol(Itemp))

    return (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr)







    

    
    
