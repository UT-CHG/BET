import numpy as np
import bet.util as util
from bet.Comm import comm
import scipy.spatial as spatial

class no_list(exception):
    pass

class length_not_matching(exception):
    pass

class dim_not_matching(exception):
    pass


class sample_set(object):
    def __init__(self, dim):
        self._array_names = ['_values', '_volumes', '_probabilities', '_jacobians', '_error_estimates']
        self._dim = dim 
        self._values = None
        self._volumes = None
        self._probabilities = None
        self._jacobians = None
        self._error_estimates = None
        #: Local values for parallelism
        self._values_local = None
        #: Local volumes for parallelism
        self._volumes_local = None
        #: Local probabilities for parallelism
        self._probabilities_local = None
        #: Local Jacobians for parallelism
        self._jacobians_local = None
        #: Local error_estimates for parallelism
        self._error_estimates_local = None
        #: Local indicies of global arrays
        self._local_index = None
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

    def set_values(self, values):
        """
        Sets values. input is a list or 1D array
        """
        self._values = util.fix_dimensions_data(values)
        if self._values.shape[0] != self._dim:
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
        return self._jacobians = jacobians

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

    def set_values_local(self, values_local):
        self._values_local = values_local
        pass
        
    def get_values_local(self):
        return self._values_local

    def set_volumes_local(self, volumes_local):
        self._volumes_local = volumes_local
        pass

    def get_volumes_local(self):
        return self._volumes_local

    def set_probabilities_local(self, probabilities_local):
        self._probabilities_local = probabilities_local
        pass

    def get_probabilities_local(self):
        return self._probabilities_local

    def set_jacobians_local(self, jacobians_local):
        self._jacobians_local = jacobians_local
        pass

    def get_jacobians_local(self):
        return self._jacobians_local

    def set_error_estimates_local(self, error_estimates_local):
        self._error_estimates_local = error_estimates_local
        pass

    def get_error_estimates_local(self):
        return self._error_estimates_local

    def local_to_global(self):
        num = check_num_local()
        local = comm.rank*np.ones((num,), dytpe='np.int')
        first = True
        for array_name in self._array_names:
            current_array_local = getattr(self, array_name + "_local")
            if current_array_local:
                setattr(self, array_name, util.get_global(current_array_local))
            if first:
                global_var = util.get_global(local)
                self._local_index = np.equal(global_var, comm.rank)
                first = False
        pass

    def global_to_local(self):
        num = check_num()
        local_num = num % comm.size
        local_val = min(local_num*(comm.rank + 1), num)
        self._local_index = range(local_num*comm.rank, local_val) 
        #self._local_index = range(0+comm.rank, self._values.shape[0], comm.size)
        for array_name in self._array_names:
            current_array = getattr(self, array_name)
            if current_array:
                setattr(self, array_name + "_local", current_array[local_index])
        pass
                
        
        

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
        #: local io pointer for parallelism
        self._io_ptr_local = None
        #: local emulated ii ptr for parallelsim
        self._emulated_ii_ptr_local = None
        #: local emulated oo ptr for parallelism
        self._emulated_oo_ptr_local = None
        self.check_nums()
        pass

    def check_nums(self):
        if self._input_sample_set._values.shape[0] != self._output_sample_set._values.shape[0]:
            raise length_not_matching("input and output lengths do not match")

    def set_io_ptr(self, localize = False):
        if not self._output_sample_set._values_local:
            self._output_sample_set.get_local_values()
        (_, self._io_ptr_local) = self._output_probability_set.get_kdtree.query(self._output_sample_set.values_local)
        if not localize:
            self.io_ptr = util.get_global_values(self._io_ptr_local)
        pass

    def get_io_ptr(self):
        return self._io_ptr
                
    def set_emulated_ii_ptr(self, localize = False):
        if not self._emulated_input_sample_set.values._local:
            self._output_sample_set.get_local_values()
        (_, self._emulated_ii_ptr_local) = self._input_sample_set.get_kdtree.query(self._emulated_input_sample_set._values_local)
        if not localize:
            self.emulate_ii_ptr_local = util.get_global_values(self._emulated_ii_ptr_local)
        pass

    def get_emulated_ii_ptr(self):
        return self._emulated_ii_ptr

    def set_emulated_oo_ptr(self, localize = False):
        if not self._emulated_output_sample_set.values._local:
            self._emulated_output_sampe_set.get_local_values()
        (_, self._emulated_oo_ptr_local) = self._output_probability_set.get_kdtree.query(self._emulated_output_sample_set._values_local)
        if not localize:
            self.emulate_oo_ptr = util.get_global_values(self._emulated_oo_ptr_local)

    def get_emulated_oo_ptr(self):
        return self._emulated_oo_ptr = self._emulated_oo_ptr
