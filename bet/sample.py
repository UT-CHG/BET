# Copyright (C) 2016 The BET Development Team


"""
This module contains data structure/storage classes for BET. Notably:
    :class:`bet.sample.sample_set`
    :class:`bet.sample.discretization`
    :class:`bet.sample.length_not_matching`
    :class:`bet.sample.dim_not_matching`
"""

import numpy as np
import bet.util as util
import scipy.spatial as spatial

class length_not_matching(exception):
    """
    Exception for when the length of the array is inconsistent.
    """
    pass

class dim_not_matching(exception):
    """
    Exception for when the dimension of the array is inconsistent.
    """
    pass

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
        pass

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
            if current_array:
                if num is None:
                    num = current_array.shape[0]
                    first_array = array_name
                else:
                    if num != current_array.shape[0]:
                        raise length_not_matching("length of " + array_name +"\
                                inconsistent with " + first_array) 
        reutrn num

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
        if self._values.shape[0] != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        pass

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
        pass

    def set_domain(self, domain):
        """
        Sets the domain.

        :param domain: Sample domain
        :type domain: :class:`numpy.ndarray` of shape (dim, 2)
        
        """
        if domiain.shape[0] != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        else:
            self._domain = domain
        pass

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
        pass

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
        pass

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
        pass

    def get_jacobians(self):
         """
        Returns sample jacobians.

        :rtype: :class:`numpy.ndarray` of shape (num, other_dim, dim)
        :returns: sample jacobians

        """
        return self._jacobians = jacobians

    def append_jacobians(self, new_jacobians):
         """
        Appends the ``new_jacobians`` to ``self._jacobians``. 

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_jacobians: New jacobians to append.
        :type new_jacobians: :class:`numpy.ndarray` of shape (num, other_dim, dim)

        """
        self._jacobians = np.concatenate((self._jacobians, new_jacobians), axis=0)
        pass

    def set_error_estimates(self, error_estimates):
        """
        Returns sample error estimates.

        :type error_estimates: :class:`numpy.ndarray` of shape (num,)
        :param error_estimates: sample error estimates

        """
        self._error_estimates = error_estimates
        pass

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
        pass

    def set_kdtree(self):
        """
        Creates a :class:`scipy.spatial.KDTree` for this set of samples.
        """
        self._kdtree = spatial.KDTree(self._values)
        pass

    def get_kdtree(self):
        """
        Returns a :class:`scipy.spatial.KDTree` for this set of samples.
        
        :rtype: :class:`scipy.spatial.KDTree`
        :returns: :class:`scipy.spatial.KDTree` for this set of samples.
        
        """
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
        else:
            return self._input_sample_set.check_num()

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
