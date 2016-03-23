# Copyright (C) 2016 The BET Development Team


"""
This module contains data structure/storage classes for BET. Notably:
    :class:`bet.sample.sample_set`
    :class:`bet.sample.discretization`
    :class:`bet.sample.length_not_matching`
    :class:`bet.sample.dim_not_matching`
"""

import numpy as np
import scipy.spatial as spatial
from bet.Comm import comm
import bet.util as util

class length_not_matching(Exception):
    """
    Exception for when the length of the array is inconsistent.
    """
    

class dim_not_matching(Exception):
    """
    Exception for when the dimension of the array is inconsistent.
    """
    
class sample_set(object):
    """

    A data structure containing arrays specific to a set of samples.

    """
    def __init__(self, dim):
        """

        Initialization
        
        :param int dim: Dimension of the space in which these samples reside.

        """
        #: List of attribute names for attributes that are
        #: :class:`numpy.ndarray`
        self._array_names = ['_values', '_volumes', '_probabilities',
                '_jacobians', '_error_estimates'] 
        #: Dimension of the sample space
        self._dim = dim 
        #: :class:`numpy.ndarray` of sample values of shape (num, dim)
        self._values = None
        #: :class:`numpy.ndarray` of sample Voronoi volumes of shape (num, dim)
        self._volumes = None
        #: :class:`numpy.ndarray` of sample probabilities of shape (num, dim)
        self._probabilities = None
        #: :class:`numpy.ndarray` of Jacobians at samples of shape (num,
        #: other_dim, dim)
        self._jacobians = None
        #: :class:`numpy.ndarray` of model error estimates at samples of shape
        #: (num, ??) 
        self._error_estimates = None
        #: The sample domain :class:`numpy.ndarray` of shape (dim, 2)
        self._domain = None
        #: :class:`scipy.spatial.KDTree`
        self._kdtree = None
        #: Local values for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num, dim)
        self._values_local = None
        #: Local volumes for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num,)
        self._volumes_local = None
        #: Local probabilities for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num,)
        self._probabilities_local = None
        #: Local Jacobians for parallelism, :class:`numpy.ndarray` of shape
        #: (local_num, other_dim, dim)
        self._jacobians_local = None
        #: Local error_estimates for parallelism, :class:`numpy.ndarray` of
        #: shape (local_num,)
        self._error_estimates_local = None
        #: Local indicies of global arrays, :class:`numpy.ndarray` of shape
        #: (local_num,)
        self._local_index = None

    def check_num(self):
        """
        
        Checks that the number of entries in ``self._values``,
        ``self._volumes``, ``self._probabilities``, ``self._jacobians``, and
        ``self._error_estimates`` all match (assuming the named array exists).
        
        :rtype: int
        :returns: num

        """
        num = None
        for array_name in self._array_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                if num is None:
                    num = current_array.shape[0]
                    first_array = array_name
                else:
                    if num != current_array.shape[0]:
                        raise length_not_matching("length of " + array_name +
                                                  " inconsistent with " + first_array) 
        return num

    def get_dim(self):
        """

        Return the dimension of the sample space.
        
        :rtype: int
        :returns: Dimension of the sample space.

        """
        return self._dim

    def set_values(self, values):
        """
        Sets the sample values. 
        
        :param values: sample values
        :type values: :class:`numpy.ndarray` of shape (num, dim)

        """
        self._values = util.fix_dimensions_data(values)
        if self._values.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        
    def get_values(self):
        """
        Returns sample values.

        :rtype: :class:`numpy.ndarray`
        :returns: sample values

        """
        return self._values

    def append_values(self, new_values):
        """
        Appends the ``new_values`` to ``self._values``. 

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_values: New values to append.
        :type new_values: :class:`numpy.ndarray` of shape (num, dim)

        """
        new_values = util.fix_dimensions_data(new_values)
        self._values = np.concatenate((self._values, new_values), axis=0)
        
    def set_domain(self, domain):
        """
        Sets the domain.

        :param domain: Sample domain
        :type domain: :class:`numpy.ndarray` of shape (dim, 2)
        
        """
        if domain.shape[0] != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        else:
            self._domain = domain
        
    def get_domain(self):
        """
        Returns the sample domain,

        :rtype: :class:`numpy.ndarray` of shape (dim, 2)
        :returns: Sample domain

        """
        return self._domain

    def set_volumes(self, volumes):
        """
        Sets sample Voronoi cell volumes.

        :type volumes: :class:`numpy.ndarray` of shape (num,)
        :param volumes: sample Voronoi cell volumes

        """
        self._volumes = volumes
        
    def get_volumes(self):
        """
        Returns sample Voronoi cell volumes.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample Voronoi cell volumes

        """
        return self._volumes

    def set_probabilities(self, probabilities):
        """
        Set sample probabilities.

        :type probabilities: :class:`numpy.ndarray` of shape (num,)
        :param probabilities: sample probabilities

        """
        self._probabilities = probabilities
        
    def get_probabilities(self):
        """
        Returns sample probabilities.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample probabilities

        """
        return self._probabilities

    def set_jacobians(self, jacobians):
        """
        Returns sample jacobians.

        :type jacobians: :class:`numpy.ndarray` of shape (num, other_dim, dim)
        :param jacobians: sample jacobians

        """
        self._jacobians = jacobians
        
    def get_jacobians(self):
        """
        Returns sample jacobians.

        :rtype: :class:`numpy.ndarray` of shape (num, other_dim, dim)
        :returns: sample jacobians

        """
        return self._jacobians 

    def append_jacobians(self, new_jacobians):
        """
        Appends the ``new_jacobians`` to ``self._jacobians``. 

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_jacobians: New jacobians to append.
        :type new_jacobians: :class:`numpy.ndarray` of shape (num, other_dim, 
            dim)

        """
        self._jacobians = np.concatenate((self._jacobians, new_jacobians),
                axis=0)

    def set_error_estimates(self, error_estimates):
        """
        Returns sample error estimates.

        :type error_estimates: :class:`numpy.ndarray` of shape (num,)
        :param error_estimates: sample error estimates

        """
        self._error_estimates = error_estimates

    def get_error_estimates(self):
        """
        Returns sample error_estimates.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample error_estimates

        """
        return self._error_estimates

    def append_error_estimates(self, new_error_estimates):
        """
        Appends the ``new_error_estimates`` to ``self._error_estimates``. 

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_error_estimates: New error_estimates to append.
        :type new_error_estimates: :class:`numpy.ndarray` of shape (num,)

        """
        self._error_estimates = np.concatenate((self._error_estimates,
            new_error_estimates), axis=0) 
        
    def set_kdtree(self):
        """
        Creates a :class:`scipy.spatial.KDTree` for this set of samples.
        """
        self._kdtree = spatial.KDTree(self._values)
        
    def get_kdtree(self):
        """
        Returns a :class:`scipy.spatial.KDTree` for this set of samples.
        
        :rtype: :class:`scipy.spatial.KDTree`
        :returns: :class:`scipy.spatial.KDTree` for this set of samples.
        
        """
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
        """
        Makes global arrays from available local ones.
        """
        for array_name in self._array_names:
            current_array_local = getattr(self, array_name + "_local")
            if current_array_local is not None:
                setattr(self, array_name, util.get_global_values(current_array_local))
        pass

    def global_to_local(self):
        """
        Makes local arrays from available global ones.
        """
        num = self.check_num()
        global_index = np.arange(num, dtype=np.int)
        self._local_index = np.array_split(global_index, comm.size)[comm.rank]
        for array_name in self._array_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(self, array_name + "_local",
                        current_array[self._local_index]) 
                

class discretization(object):
    """
    A data structure to store all of the :class:`~bet.sample.sample_set`
    objects and associated pointers to solve an stochastic inverse problem. 
    """
    def __init__(self, input_sample_set, output_sample_set,
            emulated_input_sample_set=None, emulated_output_sample_set=None,
            output_probability_set=None):
        #: Input sample set :class:`~bet.sample.sample_set`
        self._input_sample_set = input_sample_set
        #: Output sample set :class:`~bet.sample.sample_set`
        self._output_sample_set = output_sample_set
        #: Emulated Input sample set :class:`~bet.sample.sample_set`
        self._emulated_input_sample_set = emulated_input_sample_set
        #: Emulated output sample set :class:`~bet.sample.sample_set`
        self._emulated_output_sample_set = emulated_output_sample_set
        #: Output probability set :class:`~bet.sample.sample_set`
        self._output_probability_set = output_probability_set
        #: Pointer from ``self._output_sample_set`` to 
        #: ``self._output_probability_set`` 
        self._io_ptr = None
        #: Pointer from ``self._emulated_input_sample_set`` to
        #: ``self._input_sample_set`` 
        self._emulated_ii_ptr = None
        #: Pointer from ``self._emulated_output_sample_set`` to 
        #: ``self._output_probability_set``
        self._emulated_oo_ptr = None
        #: local io pointer for parallelism
        self._io_ptr_local = None
        #: local emulated ii ptr for parallelsim
        self._emulated_ii_ptr_local = None
        #: local emulated oo ptr for parallelism
        self._emulated_oo_ptr_local = None
        self.check_nums()
        
    def check_nums(self):
        """
        
        Checks that ``self._input_sample_set`` and ``self._output_sample_set``
        both have the same number of samples.

        :rtype: int
        :returns: Number of samples

        """
        if self._input_sample_set._values.shape[0] != \
                self._output_sample_set._values.shape[0]:
            raise length_not_matching("input and output lengths do not match")
        else:
            return self._input_sample_set.check_num()

    def set_io_ptr(self, globalize=True):
        """
        
        Creates the pointer from ``self._output_sample_set`` to
        ``self._output_probability_set``

        .. seealso::
            
            :meth:`scipy.spatial.KDTree.query``
        
        """
        if not self._output_sample_set._values_local:
            self._output_sample_set.get_local_values()
        (_, self._io_ptr_local) = self._output_probability_set.get_kdtree.query\
                (self._output_sample_set.values_local)
        if globalize:
            self._io_ptr = util.get_global_values(self._io_ptr_local)
       
    def get_io_ptr(self):
        """
        
        Returns the pointer from ``self._output_sample_set`` to
        ``self._output_probability_set``

        .. seealso::
            
            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._output_sample_set._values.shape[0],)
        :returns: self._io_ptr

        """
        return self._io_ptr
                
    def set_emulated_ii_ptr(self, globalize=True):
        """
        
        Creates the pointer from ``self._emulated_input_sample_set`` to
        ``self._input_sample_set``

        .. seealso::
            
            :meth:`scipy.spatial.KDTree.query``

        """
        if not self._emulated_input_sample_set.values._local:
            self._output_sample_set.get_local_values()
        (_, self._emulated_ii_ptr_local) = self._input_sample_set.get_kdtree.\
                query(self._emulated_input_sample_set._values_local)
        if globalize:
            self._emulated_ii_ptr = util.get_global_values\
                    (self._emulated_ii_ptr_local)

    def get_emulated_ii_ptr(self):
        """
        
        Returns the pointer from ``self._emulated_input_sample_set`` to
        ``self._input_sample_set``

        .. seealso::
            
            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._output_sample_set._values.shape[0],)
        :returns: self._emulated_ii_ptr

        """
        return self._emulated_ii_ptr

    def set_emulated_oo_ptr(self, globalize=True):
        """
        
        Creates the pointer from ``self._emulated_output_sample_set`` to
        ``self._output_probability_set``

        .. seealso::
            
            :meth:`scipy.spatial.KDTree.query``

        """
        if not self._emulated_output_sample_set.values._local:
            self._emulated_output_sampe_set.get_local_values()
        (_, self._emulated_oo_ptr_local) = self._output_probability_set.\
                get_kdtree.query(self._emulated_output_sample_set._values_local)
        if globalize:
            self._emulated_oo_ptr = util.get_global_values\
                    (self._emulated_oo_ptr_local)

    def get_emulated_oo_ptr(self):
        """
        
        Returns the pointer from ``self._emulated_output_sample_set`` to
        ``self._output_probabilityset``

        .. seealso::
            
            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._output_sample_set._values.shape[0],)
        :returns: self._emulated_ii_ptr

        """
        return self._emulated_oo_ptr
