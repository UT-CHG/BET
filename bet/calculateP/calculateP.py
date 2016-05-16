# Copyright (C) 2014-2015 The BET Development Team

r""" 
This module provides methods for calulating the probability measure
:math:`P_{\Lambda}`.

* :mod:`~bet.calculateP.prob_emulated` provides a skeleton class and calculates
    the probability for a set of emulation points.
* :mod:`~bet.calculateP.calculateP.prob_samples_mc` estimates the probability based on pre-defined volumes.
"""
from bet.Comm import comm, MPI 
import numpy as np
import scipy.spatial as spatial
import bet.util as util
import bet.sample as samp

def emulate_iid_lebesgue(domain, num_l_emulate, globalize=False):
    """
    Parition the parameter space using emulated samples into many voronoi
    cells. These samples are iid so that we can apply the standard MC                                       
    assumuption/approximation

    :param domain: The domain for each parameter for the model.
    :type domain: :class:`~numpy.ndarray` of shape (ndim, 2)  
    :param num_l_emulate: The number of emulated samples.
    :type num_l_emulate: int

    :rtype: :class:`~bet.sample.voronoi_sample_set`
    :returns: a set of samples for emulation

    """
    num_l_emulate = (num_l_emulate/comm.size) + \
            (comm.rank < num_l_emulate%comm.size)
    lam_width = domain[:, 1] - domain[:, 0]
    lambda_emulate = lam_width*np.random.random((num_l_emulate,
        domain.shape[0]))+domain[:, 0]

    set_emulated = samp.voronoi_sample_set(dim=domain.shape[0])
    set_emulated._domain = domain
    set_emulated._values_local = lambda_emulate
    if globalize:
        set_emulated.local_to_global()
    return set_emulated

def prob_emulated(discretization, globalize=True): 
    r"""

    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{emulate}})`, the
    probability assoicated with a set of voronoi cells defined by
    ``num_l_emulate`` iid samples :math:`(\lambda_{emulate})`.
    This is added to the emulated input sample set object.

    :param discretization: An object containing the discretization information.
    :type class:`bet.sample.discretization`
    :param bool globalize: Makes local variables global.

    """

    # Check dimensions
    discretization.check_nums()
    op_num = discretization._output_probability_set.check_num()
    emi_num = discretization._emulated_input_sample_set.check_num()

    # Check for necessary properties
    if discretization._io_ptr_local is None:
        discretization.set_io_ptr(globalize=True)
    if discretization._emulated_ii_ptr_local is None:
        discretization.set_emulated_ii_ptr(globalize=False)

    # Calculate Probabilties
    P = np.zeros((discretization._emulated_input_sample_set._values_local.shape[0],))
    d_distr_emu_ptr = discretization._io_ptr[discretization._emulated_ii_ptr_local]
    for i in range(op_num):
        if discretization._output_probability_set._probabilities[i] > 0.0:
            Itemp = np.equal(d_distr_emu_ptr, i)
            Itemp_sum = np.sum(Itemp)
            Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
            if Itemp_sum > 0:
                P[Itemp] = discretization._output_probability_set._probabilities[i]/Itemp_sum
    
    discretization._emulated_input_sample_set._probabilities_local = P
    
    pass


def prob(discretization): 
    r"""
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of  cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these 
    cells are provided.

    :param discretization: An object containing the discretization information.
    :type class:`bet.sample.discretization`
    :param bool globalize: Makes local variables global.

    """

    # Check Dimensions
    discretization.check_nums()
    op_num = discretization._output_probability_set.check_num()

    # Check for necessary attributes
    if discretization._io_ptr_local is None:
        discretization.set_io_ptr(globalize=False)

    # Calculate Probabilities
    if discretization._input_sample_set._values_local is None:
            discretization._input_sample_set.global_to_local()
    P_local = np.zeros((len(discretization._io_ptr_local),))
    for i in range(op_num):
        if discretization._output_probability_set._probabilities[i] > 0.0:
            Itemp = np.equal(discretization._io_ptr_local, i)
            Itemp_sum = np.sum(discretization._input_sample_set._volumes_local[Itemp])
            Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
            if Itemp_sum > 0:            
                P_local[Itemp] = discretization._output_probability_set._probabilities[i]*discretization._input_sample_set._volumes_local[Itemp]/Itemp_sum
        
    discretization._input_sample_set._probabilities = util.get_global_values(P_local)
    discretization._input_sample_set._probabilities_local = P_local


def prob_mc(discretization): 
    r"""
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability associated with a set of  cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes are calculated
    with the given emulated input points.

    :param discretization: An object containing the discretization information.
    :type class:`bet.sample.discretization`
    :param globalize: Makes local variables global.
    :type bool

    """

    # Check Dimensions
    num = discretization.check_nums()
    op_num = discretization._output_probability_set.check_num()
    if discretization._output_probability_set._values_local is None:
        discretization._output_probability_set.global_to_local()
    if discretization._emulated_input_sample_set._values_local is None:
        discretization._emulated_input_sample_set.global_to_local()

    # Calculate Volumes
    (_, emulate_ptr) = discretization._input_sample_set.query(discretization._emulated_input_sample_set._values_local)
    vol = np.zeros((num,))
    for i in range(num):
        vol[i] = np.sum(np.equal(emulate_ptr,i))
    cvol = np.copy(vol)
    comm.Allreduce([vol, MPI.DOUBLE], [cvol, MPI.DOUBLE], op=MPI.SUM)
    vol = cvol
    vol = vol/float(discretization._emulated_input_sample_set._values.shape[0])
    discretization._input_sample_set._volumes = vol
    discretization._input_sample_set.global_to_local()

    return prob(discretization)


    

    
