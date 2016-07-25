# Copyright (C) 2014-2016 The BET Development Team

"""
This module provides methods for generating and using surrogate models. 
"""
import numpy as np
import bet.sample as sample
import bet.calculateP.calculateError as calculateError
import bet.calculateP.calculateP as calculateP
from bet.Comm import comm, MPI

class piecewise_polynomial_surrogate(object):
    """
    This class provides methods for generating a piecewise polynomial
    surrogate.
    """
    def __init__(self, input_disc):
        """
        Initializes a piecewise polynomial surrogate based on 
        existing input discretization.

        :param discretization: An object containing the discretization 
            information.
        :type discretization: :class:`bet.sample.discretization`

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
        constant. For order 1, values are piecewise linear (assuming Jacobians
        exist), and error estimates are piecewise constant.

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
            msg = "Dimensions of input sets are not equal."
            raise sample.dim_not_matching(msg)
        
        # Give properties from input discretization.    
        if input_sample_set._domain is None:
            if self.input_disc._input_sample_set._domain is not None:
                input_sample_set.set_domain(self.input_disc.\
                        _input_sample_set._domain)
        if input_sample_set._p_norm is None:
            if self.input_disc._input_sample_set._p_norm is not None:
                input_sample_set.set_p_norm(self.input_disc.\
                        _input_sample_set._p_norm)

        # Setup dummy discretizion to get pointers
        # Assumes Voronoi sample set for now
        output_sample_set = sample.sample_set(self.input_disc.\
                _output_sample_set._dim)
        self.dummy_disc = self.input_disc.copy()
        self.dummy_disc.set_emulated_input_sample_set(input_sample_set)
        self.dummy_disc.set_emulated_ii_ptr(globalize=False)

        if order == 0:
            # define new values based on piecewise constants
            new_values_local = self.input_disc._output_sample_set.\
                    _values[self.dummy_disc._emulated_ii_ptr_local]
            output_sample_set.set_values_local(new_values_local)
        elif order == 1:
            # define new values based on piecewise linears using Jacobians
            if self.input_disc._input_sample_set._jacobians is None:
                if self.input_disc._input_sample_set._jacobians_local is None:
                    msg = "The input discretization must" 
                    msg += " have jacobians defined."
                    raise calculateError.wrong_argument_type(msg)
                else:
                    self.input_disc._input_sample_set.local_to_global()
                    
            jac_local = self.input_disc._input_sample_set._jacobians[\
                    self.dummy_disc._emulated_ii_ptr_local]
            diff_local = self.input_disc._input_sample_set._values[\
                    self.dummy_disc._emulated_ii_ptr_local] - \
                    input_sample_set._values_local
            new_values_local = self.input_disc._output_sample_set._values[\
                    self.dummy_disc._emulated_ii_ptr_local]
            new_values_local += np.einsum('ijk,ik->ij', jac_local, diff_local)
            output_sample_set.set_values_local(new_values_local)
        
        # if they exist, define error estimates with piecewise constants
        if self.input_disc._output_sample_set._error_estimates is not None:
            new_ee = self.input_disc._output_sample_set._error_estimates[\
                    self.dummy_disc._emulated_ii_ptr_local]
            output_sample_set.set_error_estimates_local(new_ee)
        # create discretization object for the surrogate
        self.surrogate_discretization = sample.discretization(input_sample_set\
                =input_sample_set, output_sample_set=output_sample_set,
                output_probability_set=self.input_disc._output_probability_set)
        return self.surrogate_discretization
    
    def calculate_prob_for_sample_set_region(self, s_set, 
                                             regions, update_input=True):
        """
        Solves stochastic inverse problem based on surrogate points and the
        MC assumption. Calculates the probability of a regions of input space
        and error estimates for those probabilities.

        :param: s_set: sample set for which to calculate error
        :type s_set: :class:`bet.sample.sample_set_base`
        :param region: list of regions of s_set for which to calculate error
        :type region: list
        :param update_input: whether or not to update probabilities and
            errror identifiers for input discretization
        :type update_input: bool

        :rtype: tuple
        :returns: (probabilities, ``error_estimates``), the probability and
            error estimates for the region
        
        """
        if not hasattr(self, 'surrogate_discretization'):
            msg = "surrogate discretization has not been created"
            raise calculateError.wrong_argument_type(msg)
        if not isinstance(s_set, sample.sample_set_base):
            msg = "s_set must be of type bet.sample.sample_set_base"
            raise calculateError.wrong_argument_type(msg)
            
        # Calculate probability of region 
        if self.surrogate_discretization._input_sample_set._volumes_local\
                is None:
            self.surrogate_discretization._input_sample_set.\
                    estimate_volume_mc(globalize=False)
        calculateP.prob(self.surrogate_discretization, globalize=False)
        prob_new_values = calculateP.prob_from_sample_set(\
                self.surrogate_discretization._input_sample_set, s_set)
        
        # Calculate for each region
        probabilities = []
        error_estimates = []
        for region in regions:
            marker = np.equal(s_set._region, region)
            probability = np.sum(prob_new_values[marker])

            # Calculate error estimate for region
            model_error = calculateError.model_error(\
                    self.surrogate_discretization)
            error_estimate = model_error.calculate_for_sample_set_region_mc(\
                    s_set, region)
            probabilities.append(probability)
            error_estimates.append(error_estimate)
        # Update input only if 1 region is given
        if update_input:
            num = self.input_disc._input_sample_set.check_num()
            prob = np.zeros((num,))
            error_id = np.zeros((num,))
            for i in range(num):
                Itemp = np.equal(self.dummy_disc._emulated_ii_ptr_local, i)
                prob_sum = np.sum(self.surrogate_discretization.\
                        _input_sample_set._probabilities_local[Itemp])
                prob[i] = comm.allreduce(prob_sum, op=MPI.SUM)
                error_id_sum = np.sum(self.surrogate_discretization.\
                        _input_sample_set._error_id_local[Itemp])
                error_id[i] = comm.allreduce(error_id_sum, op=MPI.SUM)
            self.input_disc._input_sample_set.set_probabilities(prob)
            self.input_disc._input_sample_set.set_error_id(error_id)
                    
        return (probabilities, error_estimates)
        
        
