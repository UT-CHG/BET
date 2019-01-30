# Copyright (C) 2014-2016 The BET Development Team

r""" 
This module provides methods for calulating the probability measure
:math:`P_{\Lambda}`.

* :mod:`~bet.calculateP.prob_on_emulated_samples` provides a skeleton class and
    calculates the probability for a set of emulation points.
* :mod:`~bet.calculateP.calculateP.prob` estimates the 
    probability based on pre-defined volumes.
* :mod:`~bet.calculateP.calculateP.prob_with_emulated` estimates the 
    probability using volume emulation.
* :mod:`~bet.calculateP.calculateP.prob_from_sample_set` estimates the 
    probability based on probabilities from another sample set on the same
    space.

"""
import logging
import numpy as np
from bet.Comm import comm, MPI 
import bet.util as util
import bet.sample as samp

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
    for i in xrange(op_num):
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

def prob(discretization, globalize=True): 
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
    for i in xrange(op_num):
        if discretization._output_probability_set._probabilities[i] > 0.0:
            Itemp = np.equal(discretization._io_ptr_local, i)
            Itemp_sum = np.sum(discretization._input_sample_set.\
                    _volumes_local[Itemp])
            Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
            if Itemp_sum > 0:            
                P_local[Itemp] = discretization._output_probability_set.\
                        _probabilities[i]*discretization._input_sample_set.\
                        _volumes_local[Itemp]/Itemp_sum
    if globalize:
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
    discretization.check_nums()
    discretization._output_probability_set.check_num()
    if discretization._output_probability_set._values_local is None:
        discretization._output_probability_set.global_to_local()

    # Calculate Volumes
    discretization.estimate_input_volume_emulated()
    return prob(discretization)

def prob_from_sample_set_with_emulated_volumes(set_old, set_new, 
                                               set_emulate=None):
    r"""
    
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples_new}})`
    from :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples_old}})` using
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
    if set_emulate is None:
        logging.warning("Using MC assumption because no emulated points given")
        return prob_from_sample_set(set_old, set_new)

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
    warn = False
    for i in xrange(num_old):
        if set_old._probabilities[i] > 0.0:
            Itemp = np.equal(ptr1, i)
            Itemp_sum = np.sum(Itemp)
            Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
            if Itemp_sum > 0:
                prob_em[Itemp] += set_old._probabilities[i]/float(Itemp_sum)
            else:
                warn = True
    # Warn that some cells have no emulated points in them
    if warn:
        msg = "Some old cells have no emulated points in them. "
        msg += "Renormalizing probability."
        logging.warning(msg)
        total_prob = np.sum(prob_em)
        total_prob = comm.allreduce(total_prob, op=MPI.SUM)
        prob_em = prob_em/total_prob
    # Loop over new cells and distribute probability from emulated cells
    for i in xrange(num_new):
        Itemp = np.equal(ptr2, i)
        Itemp_sum = np.sum(prob_em[Itemp])
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        prob_new[i] = Itemp_sum
    
    # Set probabilities
    set_new.set_probabilities(prob_new)
    return prob_new

def prob_from_sample_set(set_old, set_new):
    r"""
    
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples_new}})`
    from :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples_old}})` using
    the MC assumption with respect to set_old.

    :param set_old: Sample set on which probabilities have already been
        calculated
    :type set_old: :class:`~bet.sample.sample_set_base` 
    :param set_new: Sample set for which probabilities will be calculated.
    :type set_new: :class:`~bet.sample.sample_set_base` 
    
    """
    # Check dimensions
    set_old.check_num()
    num_new = set_new.check_num()

    if (set_old._dim != set_new._dim):
        raise samp.dim_not_matching("Dimensions of sets are not equal.")

    # Map old points new sets
    if set_old._values_local is None:
        set_old.global_to_local()
    (_, ptr) = set_new.query(set_old._values_local)
    ptr = ptr.flat[:]

    # Set up probability vector
    prob_new = np.zeros((num_new,))

    # Loop over new cells and distribute probability from old
    for i in xrange(num_new):
        Itemp = np.equal(ptr, i)
        Itemp_sum = np.sum(set_old._probabilities_local[Itemp])
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        prob_new[i] = Itemp_sum
    
    # Set probabilities
    set_new.set_probabilities(prob_new)
    return prob_new

def prob_from_discretization_input(disc, set_new):
    r"""
    
    Calculates :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples_new}})`
    from :math:`P_{\Lambda}(\mathcal{V}_{\lambda_{samples_old}})` where
    :math:`\lambda_{samples_old}` come from an input discretization.

    :param disc: Discretiztion on which probabilities have already been
        calculated
    :type disc: :class:`~bet.sample.discretization` 
    :param set_new: Sample set for which probabilities will be calculated.
    :type set_new: :class:`~bet.sample.sample_set_base` 

    """
    if disc._emulated_input_sample_set is None:
        logging.warning("Using MC assumption because no emulated points given")
        em_set = disc._input_sample_set
    else:
        em_set = disc._emulated_input_sample_set
    
    if em_set._values_local is None:
        em_set.global_to_local()
    if em_set._probabilities_local is None:
        raise AttributeError("Probabilities must be pre-calculated.")

    # Check dimensions
    disc.check_nums()
    num_new = set_new.check_num()

    if (disc._input_sample_set._dim != set_new._dim):
        raise samp.dim_not_matching("Dimensions of sets are not equal.")

    (_, ptr) = set_new.query(em_set._values_local)
    ptr = ptr.flat[:]

    # Set up probability vectors
    prob_new = np.zeros((num_new,))
    prob_em = em_set._probabilities_local

    for i in xrange(num_new):
        Itemp = np.equal(ptr, i)
        Itemp_sum = np.sum(prob_em[Itemp])
        Itemp_sum = comm.allreduce(Itemp_sum, op=MPI.SUM)
        prob_new[i] = Itemp_sum
    
    # Set probabilities
    set_new.set_probabilities(prob_new)
    return prob_new
