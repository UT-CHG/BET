import numpy as np
import bet.util as util
import scipy.spatial as spatial

class no_list(Exception):
	pass

class length_not_matching(Exception):
	pass

class dim_not_matching(Exception):
    pass


class sample_set(object):
    def __init__(self, dim):
        self._array_names = ['_values', '_volumes', '_probabilities', '_jacobians', '_error_estimates']
        self._dim = dim
        self.domain = None
        self._values = None
        self._volumes = None
        self._probabilities = None
        self._jacobians = None
        self._error_estimates = None
        pass

    def check_num(self):
        num = None
        for array_name in self._array_names:
            current_array = getattr(self, array_name)
            if current_array:
                if num is None:
                    num = current_array.shape[0]
                    first_array = array_name
                else:
                    if num != current_array.shape[0]:
                        raise length_not_matching("length of " + array_name + " inconsistent with " + first_array)

    def get_dim(self):
        return self._dim

    def set_domain(self, domain):
        self._domain = domain

    def get_domain(self):
        return self._domain

    def set_values(self, values):
        """
        Sets values. input is a list or 1D array
        """
        self._values = util.fix_dimensions_data(values)
        if self._values.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        pass

    def get_values(self):
        """
        Returns value
        """
        return self._values

    def append_values(self, new_values):
        new_values = util.fix_dimensions_data(new_values)
        self._values = np.concatenate((self._values, new_values), axis=0)
        pass

    def set_volumes(self, volumes):
        self._volumes = volumes
        pass

    def get_volumes(self):
        return self._volumes

    def set_probabilities(self, probabilities):
        self._probabilities = probabilities
        pass

    def get_probabilities(self):
        return self._probabilities

    def set_jacobians(self, jacobians):
        self._jacobians = jacobians
        pass

    def get_jacobians(self):
        return self._jacobians

    def append_jacobians(self, new_jacobians):
        self._jacobians = np.concatenate((self._jacobians, new_jacobians), axis=0)
        pass

    def set_error_estimates(self, error_estimates):
        self._error_estimates = error_estimates
        pass

    def get_error_estimates(self):
        return self._error_estimates

    def append_error_estimates(self, new_error_estimates):
        self._error_estimates = np.concatenate((self._error_estimates, new_error_estimates), axis=0)
        pass

    def set_kdtree(self):
        self._kdtree = spatial.KDTree(self._values)
        pass

    def get_kdtree(self):
        return self._kdtree

class discretization(object):
    def __init__(self, input_sample_set, output_sample_set, input_domain=None, output_domain=None,
                 emulated_input_sample_set=None, emulated_output_sample_set=None,
                 output_probability_set=None):
        self._input_sample_set = input_sample_set
        self._output_sample_set = output_sample_set
        self._input_domain = input_domain
        self._output_domain = output_domain
        self._emulated_input_sample_set = emulated_input_sample_set
        self._emulated_output_sample_set = emulated_output_sample_set
        self._output_probability_set = output_probability_set
        self._io_ptr = None
        self._emulated_ii_ptr = None
        self._emulated_oo_ptr = None
        self.check_nums()
        pass

    def check_nums(self):
        if self._input_sample_set._values.shape[0] != self._output_sample_set._values.shape[0]:
            raise length_not_matching("input and output lengths do not match")

    def set_io_ptr(self):
        (_, self._io_ptr) = self._output_probability_set.get_kdtree.query(self._output_sample_set.get_values())
        pass

    def get_io_ptr(self):
        return self._io_ptr
                
    def set_emulated_ii_ptr(self):
        (_, self._emulated_ii_ptr) = self._input_sample_set.get_kdtree.query(self._emulated_input_sample_set.get_values())
        pass

    def get_emulated_ii_ptr(self):
        return self._emulated_ii_ptr

    def set_emulated_oo_ptr(self):
        (_, self._emulated_oo_ptr) = self._output_probability_set.get_kdtree.query(self._emulated_output_sample_set.get_values())

    def get_emulated_oo_ptr(self):
        return self._emulated_oo_ptr
