# Copyright (C) 2014-2016 The BET Development Team

r""" 
This module provides methods for calulating the probability measure
:math:`P_{\Lambda}`.

* :mod:`~bet.calculateP.prob_on_emulated_samples` provides a skeleton class and calculates
    the probability for a set of emulation points.
* :mod:`~bet.calculateP.calculateP.prob` estimates the 
    probability based on pre-defined volumes.
* :mod:`~bet.calculateP.calculateP.prob_with_emulated` estimates the 
    probability using volume emulation.
* :mod:`~bet.calculateP.calculateP.prob_from_sample_set` estimates the 
    probability based on probabilities from another sample set on the same space.

"""
import numpy as np
from bet.Comm import comm, MPI 
import bet.util as util
import bet.sampling.basicSampling as bsam

def prob_on_emulated_samples(discretization, globalize=True): 
    r"""

    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{emulate}})`, the
    probability assoicated with a set of voronoi cells defined by
    ``num_l_emulate`` iid samples :math:`(\lambda_{emulate})`.
    This is added to the emulated input sample set object.

    :param discretization: An object containing the discretization information.
    :type discretization: class:`bet.sample.discretization`
    :param bool globalize: Makes local variables global.

    """

    # Check dimensions
    discretization.check_nums()
    op_num = discretization._output_probability_set.check_num()
    discretization._emulated_input_sample_set.check_num()

    # Check for necessary properties
    if discretization._io_ptr_local is None:
        discretization.set_io_ptr(globalize=True)
    if discretization._emulated_ii_ptr_local is None:
        discretization.set_emulated_ii_ptr(globalize=False)

    # Calculate Probabilties
    P = np.zeros((discretization._emulated_input_sample_set.\
            _values_local.shape[0],))
    d_distr_emu_ptr = discretization._io_ptr[discretization.\
            _emulated_ii_ptr_local]
    for i in range(op_num):
        if discretization._output_probability_set._probabilities[i] > 0.0:
            Itemp = np.equal(d_distr_emu_ptr, i)
            Itemp_sum = np.sum(Itemp)
            Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
            if Itemp_sum > 0:
                P[Itemp] = discretization._output_probability_set.\
                        _probabilities[i]/Itemp_sum
    
    discretization._emulated_input_sample_set._probabilities_local = P
    if globalize:
        discretization._emulated_input_sample_set.local_to_global()
    pass

def prob(discretization): 
    r"""
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability assoicated with a set of  cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes of these 
    cells are provided.

    :param discretization: An object containing the discretization information.
    :type discretization: class:`bet.sample.discretization`
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
            Itemp_sum = np.sum(discretization._input_sample_set.\
                    _volumes_local[Itemp])
            Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
            if Itemp_sum > 0:            
                P_local[Itemp] = discretization._output_probability_set.\
                        _probabilities[i]*discretization._input_sample_set.\
                        _volumes_local[Itemp]/Itemp_sum
        
    discretization._input_sample_set._probabilities = util.\
            get_global_values(P_local)
    discretization._input_sample_set._probabilities_local = P_local

def prob_with_emulated_volumes(discretization): 
    r"""
    
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples}})`, the
    probability associated with a set of  cells defined by the model
    solves at :math:`(\lambda_{samples})` where the volumes are calculated
    with the given emulated input points.

    :param discretization: An object containing the discretization information.
    :type discretization: class:`bet.sample.discretization`
    :param globalize: Makes local variables global.

    """

    # Check Dimensions
    num = discretization.check_nums()
    discretization._output_probability_set.check_num()
    if discretization._output_probability_set._values_local is None:
        discretization._output_probability_set.global_to_local()

    # Calculate Volumes
    discretization.estimate_input_volume_emulated()
    return prob(discretization)


    
def prob_from_sample_set(set_old, set_new, set_emulate):
    r"""
    
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples_new}})`
    from :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples_old}}) using
    a set of emulated points are distributed with respect to the 
    volume measure.

    :param set_old: Sample set on which probabilities have already been
    calculated
    :type set_old: :class:`~bet.sample.sample_set_base` 
    :param set_new: Sample set for which probabilities will be calculated.
    :type set_new: :class:`~bet.sample.sample_set_base` 
    :param set_emulate: Sample set for volume emulation
    :type set_emulate: :class:`~bet.sample.sample_set_base`

    """
    # Check dimensions
    num_old = set_old.check_num()
    num_new = set_new.check_num()
    set_emulate.check_num()
    if (set_old._dim != set_new._dim) or (set_old._dim != set_emulate._dim):
        raise samp.dim_not_matching("Dimensions of sets are not equal.")
    # Localize emulated points
    if set_emulate._values_local is None:
        set_emulate.global_to_local()

    # Map emulated points to old and new sets
    (_, ptr1) = set_old.query(set_emulate._values_local)
    (_, ptr2) = set_new.query(set_emulate._values_local)
    ptr1 = ptr1.flat[:]
    ptr2 = ptr2.flat[:]

    # Set up probability vectors
    prob_new = np.zeros((num_new,))
    prob_em = np.zeros((len(ptr1), ))
    
    # Loop over old cells and divide probability over emulated cells
    for i in range(num_old):
        if set_old._probabilities[i] > 0.0:
            Itemp = np.equal(ptr1, i)
            Itemp_sum = np.sum(Itemp)
            Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
            if Itemp_sum > 0:
                prob_em[Itemp] += set_old._probabilities[i]/float(Itemp_sum)

    # Loop over new cells and distribute probability from emulated cells
    for i in range(num_new):
        Itemp = np.equal(ptr2, i)
        Itemp_sum = np.sum(prob_em[Itemp])
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        prob_new[i] = Itemp_sum
    
    # Set probabilities
    set_new.set_probabilities(prob_new)
    return prob_new
