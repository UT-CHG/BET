# Copyright (C) 2014-2016 The BET Development Team

"""
This module provides methods for generating and using surrogate models. 
"""
import numpy as np
import logging
import bet.sample as sample
import bet.calculateP.calculateError as calculateError
from bet.Comm import comm

class piecewise_polynomial_surrogate(object):
    def __init__(self, input_disc):
        """
        Initializes a piecewise polynomial surrogate based on 
        existing input discretization.

        :param discretization: An object containing the discretization 
        information.
        :type discretization: class:`bet.sample.discretization`

        """
        if not isinstance(input_disc, sample.discretization):
            msg = "The argument must be of type bet.sample.discretization."
            raise calculateError.wrong_argument_type(msg)
        input_disc.check_nums()
        self.input_disc = input_disc
        self.input_disc._input_sample_set.local_to_global()
        self.input_disc._output_sample_set.local_to_global()
        
    def generate_for_input_set(self, input_sample_set, order=0):
        """
        Generates a surrogate discretization based on the input discretization,
        for a user-defined input sample set. The output sample set values
        and error estimates are piecewise polynomially defined over input sample
        set cells from the input discretization. For order 0, both are piecewise
        constant. For order 1, values are piecewise linear (assuming Jacobians)
        exist, and error estimates are piecewise constant.

        :param input_sample_set: input sample set for surrogate discretization
        :type set_old: :class:`~bet.sample.sample_set_base`
        :param order: Polynomial order
        :type order: int
        :rtype: :class:`~bet.sample.discretization`
        :returns: discretization defining the surrogate model

        """
        # Check inputs
        if order not in [0, 1]:
            msg = "Order must be 0 or 1."
            raise calculateError.wrong_argument_type(msg)
        input_sample_set.check_num()
        if input_sample_set._dim != self.input_disc._input_sample_set._dim:
            raise sample.dim_not_matching("Dimensions of input sets are not equal.")
        
        # Give properties from input discretization.    
        if input_sample_set._domain is None:
            if self.input_disc._input_sample_set._domain is not None:
                input_sample_set.set_domain(self.input_disc._input_sample_set._domain)
        if input_sample_set._p_norm is None:
            if self.input_disc._input_sample_set._p_norm is not None:
                input_sample_set.set_p_norm(self.input_disc._input_sample_set._p_norm)

        # Setup dummy discretizion to get pointers
        # Assumes Voronoi sample set for now
        output_sample_set = sample.sample_set(self.input_disc._output_sample_set._dim)
        dummy_disc = self.input_disc.copy()
        dummy_disc.set_emulated_input_sample_set(input_sample_set)
        dummy_disc.set_emulated_ii_ptr(globalize=False)

        if order == 0:
            # define new values based on piecewise constants
            new_values_local = self.input_disc._output_sample_set._values[dummy_disc._emulated_ii_ptr_local]
            output_sample_set.set_values_local(new_values_local)
        elif order == 1:
            # define new values based on piecewise linears using Jacobians
            if self.input_disc._input_sample_set._jacobians is None:
                if self.input_disc._input_sample_set._jacobians_local is None:
                    msg = "The input discretization must have jacobians defined."
                    raise calculateError.wrong_argument_type(msg)
                else:
                   self.input_disc._input_sample_set.local_to_global()
                    
            jac_local = self.input_disc._input_sample_set._jacobians[dummy_disc._emulated_ii_ptr_local]
            diff_local = self.input_disc._input_sample_set._values[dummy_disc._emulated_ii_ptr_local] - input_sample_set._values_local
            new_values_local = self.input_disc._output_sample_set._values[dummy_disc._emulated_ii_ptr_local]
            new_values_local += np.einsum('ijk,ik->ij', jac_local, diff_local)
            output_sample_set.set_values_local(new_values_local)
        
        # if the exist, define error estimates with piecewise constants
        if self.input_disc._output_sample_set._error_estimates is not None:
            new_ee = self.input_disc._output_sample_set._error_estimates[dummy_disc._emulated_ii_ptr_local]
            output_sample_set.set_error_estimates_local(new_ee)
        # create discretization object for the surrogate
        disc = sample.discretization(input_sample_set=input_sample_set,
                                     output_sample_set=output_sample_set,
                                     output_probability_set=self.input_disc._output_probability_set)
        return disc
        
        
