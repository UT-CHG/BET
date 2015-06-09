# Copyright (C) 2014-2015 The BET Development Team

r""" 
This module provides methods for calulating the probability measure
:math:`P_{\Lambda}`.

* :mod:`~bet.calculateP.prob_emulated` provides a skeleton class and calculates
    the probability for a set of emulation points.
* :mod:`~bet.calculateP.calculateP.prob_samples_ex` calculates the exact
    volumes of the interior voronoi cells and estimates the volumes of the
    exterior voronoi cells by using a set of bounding points
* :mod:`~bet.calculateP.calculateP.prob_samples_mc` estimates the volumes of
    the voronoi cells using MC integration
"""
from bet.Comm import comm, MPI 
import numpy as np
import scipy.spatial as spatial
import bet.util as util

def emulate_iid_lebesgue(lam_domain, num_l_emulate):
    """
    Parition the parameter space using emulated samples into many voronoi
    cells. These samples are iid so that we can apply the standard MC                                       
    assumuption/approximation

    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)  
    :param num_l_emulate: The number of emulated samples.
    :type num_l_emulate: int

    :rtype: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :returns: a set of samples for emulation

    """
    num_l_emulate = int(num_l_emulate/comm.size)+1
    lam_width = lam_domain[:, 1] - lam_domain[:, 0]
    lambda_emulate = lam_width*np.random.random((num_l_emulate,
        lam_domain.shape[0]))+lam_domain[:, 0] 
    return lambda_emulate 

def prob_emulated(samples, data, rho_D_M, d_distr_samples,
        lambda_emulate=None, d_Tree=None): 
    r"""

    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{emulate}})`, the
    probability assoicated with a set of voronoi cells defined by
    ``num_l_emulate`` iid samples :math:`(\lambda_{emulate})`.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :param lambda_emulate: Samples used to partition the parameter space
    :type lambda_emulate: :class:`~numpy.ndarray` of shape (num_l_emulate, ndim)
    :rtype: tuple
    :returns: (P, lambda_emulate, io_ptr, emulate_ptr, lam_vol)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if type(lambda_emulate) == type(None):
        lambda_emulate = samples
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if type(d_Tree) == type(None):
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data)
    
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate)
    
    # Calculate Probabilties
    P = np.zeros((lambda_emulate.shape[0],))
    d_distr_emu_ptr = np.zeros(emulate_ptr.shape)
    d_distr_emu_ptr = io_ptr[emulate_ptr]
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(d_distr_emu_ptr, i)
        Itemp_sum = np.sum(Itemp)
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P[Itemp] = rho_D_M[i]/Itemp_sum

    return (P, lambda_emulate, io_ptr, emulate_ptr)

def prob(samples, data, rho_D_M, d_distr_samples, d_Tree=None): 
    r"""
    
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of voronoi cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these voronoi
    cells are assumed to be equal under the MC assumption.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, lambda_emulate, io_ptr) where P is the
        probability associated with samples, and lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins.

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if type(d_Tree) == type(None):
        d_Tree = spatial.KDTree(d_distr_samples)

    # Set up local arrays for parallelism
    local_index = range(0+comm.rank, samples.shape[0], comm.size)
    samples_local = samples[local_index, :]
    data_local = data[local_index, :]
    local_array = np.array(local_index)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data_local)

    # Apply the standard MC approximation and
    # calculate probabilities
    P_local = np.zeros((samples_local.shape[0],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr, i)
        Itemp_sum = np.sum(Itemp)
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P_local[Itemp] = rho_D_M[i]/Itemp_sum 
    P_global = util.get_global_values(P_local)
    global_index = util.get_global_values(local_array)
    P = np.zeros(P_global.shape)
    P[global_index] = P_global[:]

    lam_vol = (1.0/float(samples.shape[0]))*np.ones((samples.shape[0],))

    return (P, lam_vol, io_ptr)

def prob_mc(samples, data, rho_D_M, d_distr_samples,
            lambda_emulate=None, d_Tree=None): 
    r"""
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of voronoi cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these voronoi
    cells are approximated using MC integration.

    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param data: The data from running the model given the samples.
    :type data: :class:`~numpy.ndarray` of size (num_samples, mdim)
    :param rho_D_M: The simple function approximation of rho_D
    :type rho_D_M: :class:`~numpy.ndarray` of shape  (M,) 
    :param d_distr_samples: The samples in the data space that define a
        parition of D to for the simple function approximation
    :type d_distr_samples: :class:`~numpy.ndarray` of shape  (M, mdim) 
    :param d_Tree: :class:`~scipy.spatial.KDTree` for d_distr_samples
    :param lambda_emulate: Samples used to partition the parameter space

    :rtype: tuple of :class:`~numpy.ndarray` of sizes (num_samples,),
        (num_samples,), (ndim, num_l_emulate), (num_samples,), (num_l_emulate,)
    :returns: (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr) where P is the
        probability associated with samples, lam_vol the volumes associated
        with the samples, io_ptr a pointer from data to M bins, and emulate_ptr
        a pointer from emulated samples to samples (in parameter space)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1) 
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1) 
    if type(lambda_emulate) == type(None):
        lambda_emulate = samples
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)
    if type(d_Tree) == type(None):
        d_Tree = spatial.KDTree(d_distr_samples)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr) = d_Tree.query(data)
    
    # Determine which emulated samples match with which model run samples
    l_Tree = spatial.KDTree(samples)
    (_, emulate_ptr) = l_Tree.query(lambda_emulate)

    # Apply the standard MC approximation to determine the number of emulated
    # samples per model run sample. This is for approximating 
    # \mu_Lambda(A_i \intersect b_j)
    lam_vol = np.zeros((samples.shape[0],)) 
    for i in range(samples.shape[0]):
        lam_vol[i] = np.sum(np.equal(emulate_ptr, i))
    clam_vol = np.copy(lam_vol) 
    comm.Allreduce([lam_vol, MPI.DOUBLE], [clam_vol, MPI.DOUBLE], op=MPI.SUM)
    lam_vol = clam_vol
    lam_vol = lam_vol/(len(lambda_emulate)*comm.size)

    # Set up local arrays for parallelism
    local_index = range(0+comm.rank, samples.shape[0], comm.size)
    samples_local = samples[local_index, :]
    data_local = data[local_index, :]
    lam_vol_local = lam_vol[local_index]
    local_array = np.array(local_index)
        
    # Determine which inputs go to which M bins using the QoI
    (_, io_ptr_local) = d_Tree.query(data_local)

    # Calculate Probabilities
    P_local = np.zeros((samples_local.shape[0],))
    for i in range(rho_D_M.shape[0]):
        Itemp = np.equal(io_ptr_local, i)
        Itemp_sum = np.sum(lam_vol_local[Itemp])
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        if Itemp_sum > 0:
            P_local[Itemp] = rho_D_M[i]*lam_vol_local[Itemp]/Itemp_sum 
    P_global = util.get_global_values(P_local)
    global_index = util.get_global_values(local_array)
    P = np.zeros(P_global.shape)
    P[global_index] = P_global[:]
    return (P, lam_vol, lambda_emulate, io_ptr, emulate_ptr)







    

    
    
