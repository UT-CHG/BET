# Copyright (C) 2016 The BET Development Team

"""
This module contains data structure/storage classes for BET. Notably:
    :class:`bet.sample.sample_set`
    :class:`bet.sample.discretization`
    :class:`bet.sample.length_not_matching`
    :class:`bet.sample.dim_not_matching`
"""

import os, logging
import numpy as np
import scipy.spatial as spatial
import scipy.io as sio
import scipy.stats
from bet.Comm import comm, MPI
import bet.util as util
import bet.sampling.LpGeneralizedSamples as lp

class length_not_matching(Exception):
    """
    Exception for when the length of the array is inconsistent.
    """
    

class dim_not_matching(Exception):
    """
    Exception for when the dimension of the array is inconsistent.
    """
    
def save_sample_set(save_set, file_name, sample_set_name=None):
    """
    Saves this :class:`bet.sample.sample_set` as a ``.mat`` file. Each
    attribute is added to a dictionary of names and arrays which are then
    saved to a MATLAB-style file.

    :param save_set: sample set to save
    :type save_set: :class:`bet.sample.sample_set`
    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string sample_set_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.sample_set` objects to a single
        ``.mat`` file

    """
    if os.path.exists(file_name) or os.path.exists(file_name+'.mat'):
        mdat = sio.loadmat(file_name)
    else:
        mdat = dict()
    if sample_set_name is None:
        sample_set_name = 'default'
    for attrname in sample_set.vector_names:
        curr_attr = getattr(save_set, attrname)
        if curr_attr is not None:
            mdat[sample_set_name+attrname] = curr_attr
    for attrname in sample_set.all_ndarray_names:
        curr_attr = getattr(save_set, attrname)
        if curr_attr is not None:
            mdat[sample_set_name+attrname] = curr_attr
    if comm.rank == 0:
        sio.savemat(file_name, mdat)

def load_sample_set(file_name, sample_set_name=None):
    """
    Loads a :class:`~bet.sample.sample_set` from a ``.mat`` file. If a file
    contains multiple :class:`~bet.sample.sample_set` objects then
    ``sample_set_name`` is used to distinguish which between different
    :class:`~bet.sample.sample_set` objects.

    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string sample_set_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.sample_set` objects to a single
        ``.mat`` file

    :rtype: :class:`~bet.sample.sample_set`
    :returns: the ``sample_set`` that matches the ``sample_set_name``
    """
    mdat = sio.loadmat(file_name)
    if sample_set_name is None:
        sample_set_name = 'default'
    
    if sample_set_name+"_dim" in mdat.keys():
        loaded_set = sample_set(np.squeeze(mdat[sample_set_name+"_dim"]))
    else:
        logging.info("No sample_set named {} with _dim in file".\
                format(sample_set_name))
        return None

    for attrname in sample_set.vector_names:
        if attrname is not '_dim':
            if sample_set_name+attrname in mdat.keys():
                setattr(loaded_set, attrname,
                    np.squeeze(mdat[sample_set_name+attrname]))
    for attrname in sample_set.all_ndarray_names:
        if sample_set_name+attrname in mdat.keys():
            setattr(loaded_set, attrname, mdat[sample_set_name+attrname])
    
    # localize arrays if necessary
    if sample_set_name+"_values_local" in mdat.keys():
        loaded_set.global_to_local()

    return loaded_set

class sample_set_base(object):
    """

    A data structure containing arrays specific to a set of samples.

    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray` or int/float
    vector_names = ['_probabilities', '_probabilities_local', '_volumes',
            '_volumes_local', '_local_index', '_dim']
    #: List of global attribute names for attributes that are 
    #: :class:`numpy.ndarray`
    array_names = ['_values', '_volumes', '_probabilities', '_jacobians',
                   '_error_estimates', '_right', '_left', '_width',
                   '_kdtree_values'] 
    #: List of attribute names for attributes that are
    #: :class:`numpy.ndarray` with dim > 1
    all_ndarray_names = ['_error_estimates', '_error_estimates_local',
                         '_values', '_values_local', '_left', '_left_local', 
                         '_right', '_right_local', '_width', '_width_local', 
                         '_domain', '_kdtree_values'] 


    def __init__(self, dim):
        """

        Initialization
        
        :param int dim: Dimension of the space in which these samples reside.

        """
        #: Dimension of the sample space
        self._dim = dim 
        #: :class:`numpy.ndarray` of sample values of shape (num, dim)
        self._values = None
        #: :class:`numpy.ndarray` of sample Voronoi volumes of shape (num,)
        self._volumes = None
        #: :class:`numpy.ndarray` of sample probabilities of shape (num,)
        self._probabilities = None
        #: :class:`numpy.ndarray` of Jacobians at samples of shape (num,
        #: other_dim, dim)
        self._jacobians = None
        #: :class:`numpy.ndarray` of model error estimates at samples of shape
        #: (num, dim) 
        self._error_estimates = None
        #: The sample domain :class:`numpy.ndarray` of shape (dim, 2)
        self._domain = None
        #: Bounding box of values, :class:`numpy.ndarray`of shape (dim, 2)
        self._bounding_box = None
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
        #: (local_num, dim)
        self._local_index = None
        #: :class:`scipy.spatial.KDTree`
        self._kdtree = None
        #: Values defining kd tree, :class:`numpy.ndarray` of shape (num, dim)
        self._kdtree_values = None
        #: Local values defining kd tree, :class:`numpy.ndarray` of 
        #: shape (num, dim)
        self._kdtree_values_local = None
        #: Local pointwise left (local_num, dim)
        self._left_local = None
        #: Local pointwise right (local_num, dim)
        self._right_local = None
        #: Local pointwise width (local_num, dim)
        self._width_local = None

        #: Pointwise left (num, dim)
        self._left = None
        #: Pointwise right (num, dim)
        self._right = None
        #: Pointwise width (num, dim)
        self._width = None

    def update_bounds(self, num=None):
        """
        Creates ``self._right``, ``self._left``, ``self._width``.

        :param int num: Determinzes shape of pointwise bounds (num, dim)

        """
        if num == None:
            num = self._values.shape[0]
        self._left = np.repeat([self._domain[:, 0]], num, 0)
        self._right = np.repeat([self._domain[:, 1]], num, 0)
        self._width = self._right-self._left

    def update_bounds_local(self, local_num=None):
        """
        Creates local versions of ``self._right``, ``self._left``,
        ``self._width`` (``self._right_local``, ``self._left_local``,
        ``self._width_local``).

        :param int local_num: Determinzes shape of local pointwise bounds
            (local_num, dim)

        """
        if local_num == None:
            local_num = self._values_local.shape[0]
        self._left_local = np.repeat([self._domain[:, 0]], local_num, 0)
        self._right_local = np.repeat([self._domain[:, 1]], local_num, 0)
        self._width_local = self._right_local-self._left_local

    def append_values(self, values):
        """
        Appends the values in ``_values`` to ``self._values``.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values: values to append
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._values = np.concatenate((self._values,
                util.fix_dimensions_data(values)), 0)

    def append_values_local(self, values_local):
        """
        Appends the values in ``_values_local`` to ``self._values``.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values_local: values to append
        :type values_local: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._values_local = np.concatenate((self._values_local,
                util.fix_dimensions_data(values_local)), 0)

    def check_num(self):
        """
        
        Checks that the number of entries in ``self._values``,
        ``self._volumes``, ``self._probabilities``, ``self._jacobians``, and
        ``self._error_estimates`` all match (assuming the named array exists).
        
        :rtype: int
        :returns: num

        """
        num = None
        for array_name in sample_set.array_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                if num is None:
                    num = current_array.shape[0]
                    first_array = array_name
                else:
                    if num != current_array.shape[0]:
                        raise length_not_matching("length of {} inconsistent \
                                with {}".format(array_name,
                                                  first_array)) 
        if self._values is not None and self._values.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        return num

    def get_dim(self):
        """

        Return the dimension of the sample space.
        
        :rtype: int
        :returns: Dimension of the sample space.

        """
        return self._dim

    def set_bounding_box(self):
        """
        Set the bounding box of the values.
        """
        mins = np.min(self._values, axis=0)
        maxes = np.max(self._values, axis=0)
        self._bounding_box = np.vstack((mins, maxes)).transpose()
        pass

    def get_bounding_box(self):
        """
        Get the bounding box of the values.
        """
        if self._bounding_box is None:
            self.set_bounding_box()
        return self._bounding_box

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
        
    def set_domain(self, domain):
        """
        Sets the domain.

        :param domain: Sample domain
        :type domain: :class:`numpy.ndarray` of shape (dim, 2)
        
        """
        if (domain.shape[0], 2) != (self._dim, 2):
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
        Sets sample cell volumes.

        :type volumes: :class:`numpy.ndarray` of shape (num,)
        :param volumes: sample cell volumes

        """
        self._volumes = volumes
        
    def get_volumes(self):
        """
        Returns sample cell volumes.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample cell volumes

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
        

    def set_values_local(self, values_local):
        """
        Sets the local sample values. 
        
        :param values_local: sample local values
        :type values_local: :class:`numpy.ndarray` of shape (local_num, dim)

        """
        self._values_local = util.fix_dimensions_data(values_local)
        if self._values_local.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        pass

    def set_kdtree(self):
        """
        Creates a :class:`scipy.spatial.KDTree` for this set of samples.
        """
        self._kdtree = spatial.KDTree(self._values)
        self._kdtree_values = self._kdtree.data

    def get_kdtree(self):
        """
        Returns a :class:`scipy.spatial.KDTree` for this set of samples.
        
        :rtype: :class:`scipy.spatial.KDTree`
        :returns: :class:`scipy.spatial.KDTree` for this set of samples.
        
        """
        return self._kdtree
        
    def get_values_local(self):
        """
        Returns sample local values.

        :rtype: :class:`numpy.ndarray`
        :returns: sample local values

        """
        return self._values_local

    def set_volumes_local(self, volumes_local):
        """
        Sets local sample cell volumes.

        :type volumes_local: :class:`numpy.ndarray` of shape (num,)
        :param volumes_local: local sample cell volumes

        """
        self._volumes_local = volumes_local
        pass

    def get_volumes_local(self):
        """
        Returns sample local volumes.

        :rtype: :class:`numpy.ndarray`
        :returns: sample local volumes

        """
        return self._volumes_local

    def set_probabilities_local(self, probabilities_local):
        """
        Set sample local probabilities.

        :type probabilities_local: :class:`numpy.ndarray` of shape (num,)
        :param probabilities_local: local sample probabilities

        """
        self._probabilities_local = probabilities_local
        pass

    def get_probabilities_local(self):
        """
        Returns sample local probablities.

        :rtype: :class:`numpy.ndarray`
        :returns: sample local probablities

        """

        return self._probabilities_local

    def set_jacobians_local(self, jacobians_local):
        """
        Returns local sample jacobians.

        :type jacobians_local: :class:`numpy.ndarray` of shape (num, other_dim,
            dim) 
        :param jacobians_local: local sample jacobians

        """
        self._jacobians_local = jacobians_local
        pass

    def get_jacobians_local(self):
        """
        Returns local sample jacobians.

        :rtype: :class:`numpy.ndarray` of shape (num, other_dim, dim)
        :returns: local sample jacobians

        """
        return self._jacobians_local

    def set_error_estimates_local(self, error_estimates_local):
        """
        Returns local sample error estimates.

        :type error_estimates_local: :class:`numpy.ndarray` of shape (num,)
        :param error_estimates_local: local sample error estimates

        """
        self._error_estimates_local = error_estimates_local
        pass

    def get_error_estimates_local(self):
        """
        Returns sample error_estimates_local.

        :rtype: :class:`numpy.ndarray` of shape (num,)
        :returns: sample error_estimates_local

        """
        return self._error_estimates_local

    def local_to_global(self):
        """
        Makes global arrays from available local ones.
        """
        for array_name in sample_set.array_names:
            current_array_local = getattr(self, array_name + "_local")
            if current_array_local is not None:
                setattr(self, array_name,
                        util.get_global_values(current_array_local))
    def query(self, x):
        """
        Identify which value points x are associated with for discretization.

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape (*, dim)
        """
        pass

    def estimate_volume(self, n_mc_points=int(1E4)):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration. 

        .. todo::

           This currently presumes a uniform Lesbegue measure on the
           ``domain``. Currently the way this is written
           ``emulated_input_sample_set`` is NOT used to calculate the volume.
           This should at least be an option. 

        :param int n_mc_points: If estimate is True, number of MC points to use
        """
        num = self.check_num()
        n_mc_points_local = (n_mc_points/comm.size) + \
                            (comm.rank < n_mc_points%comm.size)
        width = self._domain[:, 1] - self._domain[:, 0]
        mc_points = width*np.random.random((n_mc_points_local,
            self._domain.shape[0])) + self._domain[:, 0]
        (_, emulate_ptr) = self.query(mc_points)
        vol = np.zeros((num,))
        for i in range(num):
            vol[i] = np.sum(np.equal(emulate_ptr, i))
        cvol = np.copy(vol)
        comm.Allreduce([vol, MPI.DOUBLE], [cvol, MPI.DOUBLE], op=MPI.SUM)
        vol = cvol
        vol = vol/float(n_mc_points)
        self._volumes = vol
        self.global_to_local()
        
    def estimate_volume_mc(self):
        """
        Give all cells the same volume fraction based on the Monte Carlo
        assumption.  
        """
        num = self.check_num()
        self._volumes = 1.0/float(num)*np.ones((num,))
        self.global_to_local()

    def global_to_local(self):
        """
        Makes local arrays from available global ones.
        """
        num = self.check_num()
        global_index = np.arange(num, dtype=np.int)
        self._local_index = np.array_split(global_index, comm.size)[comm.rank]
        for array_name in sample_set.array_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(self, array_name + "_local",
                        np.array_split(current_array, comm.size)[comm.rank])

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.sample.sample_set`
        :returns: Copy of this :class:`~bet.sample.sample_set`

        """
        my_copy = sample_set(self.get_dim())
        for array_name in sample_set.all_ndarray_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(my_copy, array_name,
                        np.copy(current_array))
        for vector_name in sample_set.vector_names:
            if vector_name is not "_dim":
                current_vector = getattr(self, vector_name)
                if current_vector is not None:
                    setattr(my_copy, vector_name, np.copy(current_vector))
        if self._kdtree is not None:
            my_copy.set_kdtree()
        return my_copy

    def shape(self):
        """
        
        Returns the shape of ``self._values``

        :rtype: tuple
        :returns: (num, dim)

        """
        return self._values.shape
    
    def shape_local(self):
        """
        
        Returns the shape of ``self._values_local``

        :rtype: tuple
        :returns: (local_num, dim)

        """
        return self._values_local.shape

    def calculate_volumes(self):
        """

        Calculate the volumes of cells. Depends on sample set type.

        """

def save_discretization(save_disc, file_name, discretization_name=None):
    """
    Saves this :class:`bet.sample.discretization` as a ``.mat`` file. Each
    attribute is added to a dictionary of names and arrays which are then
    saved to a MATLAB-style file.

    :param save_disc: sample set to save
    :type save_disc: :class:`bet.sample.discretization`
    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string discretization_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.discretization` objects to a single
        ``.mat`` file

    """
    new_mdat = dict()

    if discretization_name is None:
        discretization_name = 'default'

    for attrname in discretization.sample_set_names:
        curr_attr = getattr(save_disc, attrname)
        if curr_attr is not None:
            if attrname in discretization.sample_set_names:
                save_sample_set(curr_attr, file_name,
                    discretization_name+attrname)
    
    for attrname in discretization.vector_names:
        curr_attr = getattr(save_disc, attrname)
        if curr_attr is not None:
            new_mdat[discretization_name+attrname] = curr_attr
    
    if comm.rank == 0:
        if os.path.exists(file_name) or os.path.exists(file_name+'.mat'):
            mdat = sio.loadmat(file_name)
            for i, v in new_mdat.iteritems():
                mdat[i] = v
            sio.savemat(file_name, mdat)
        else:
            sio.savemat(file_name, new_mdat)

def load_discretization(file_name, discretization_name=None):
    """
    Loads a :class:`~bet.sample.discretization` from a ``.mat`` file. If a file
    contains multiple :class:`~bet.sample.discretization` objects then
    ``discretization_name`` is used to distinguish which between different
    :class:`~bet.sample.discretization` objects.

    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string discretization_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.discretization` objects to a single
        ``.mat`` file

    :rtype: :class:`~bet.sample.discretization`
    :returns: the ``discretization`` that matches the ``discretization_name``
    """
    mdat = sio.loadmat(file_name)
    if discretization_name is None:
        discretization_name = 'default'

    input_sample_set = load_sample_set(file_name,
            discretization_name+'_input_sample_set')

    output_sample_set = load_sample_set(file_name,
            discretization_name+'_output_sample_set')

    loaded_disc = discretization(input_sample_set, output_sample_set)
        
    for attrname in discretization.sample_set_names:
        if attrname is not '_input_sample_set' and \
                attrname is not '_output_sample_set':
            setattr(loaded_disc, attrname, load_sample_set(file_name,
                    discretization_name+attrname))
    
    for attrname in discretization.vector_names:
        if discretization_name+attrname in mdat.keys():
            setattr(loaded_disc, attrname,
                        np.squeeze(mdat[discretization_name+attrname]))
    return loaded_disc

class voronoi_sample_set(sample_set_base):
    """

    A data structure containing arrays specific to a set of samples defining
    a Voronoi tesselation.

    """
    def __init__(self, dim, p_norm=2):
        sample_set_base.__init__(self, dim)
        #: p-norm to use for nearest neighbor search
        self.p_norm = p_norm

    def query(self, x):
        """
        Identify which value points x are associated with for discretization.

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape (*, dim)

        :rtype: tuple
        :returns: (dist, ptr)
        """
        if self._kdtree is None:
            self.set_kdtree()
        else:
            self.check_num()


        #TODO add exception if dimensions of x are wrong
        (dist, ptr) = self._kdtree.query(x, p=self.p_norm)
        return (dist, ptr)

    def exact_volume_1D(self, distribution='uniform', a=None, b=None):
        r"""
        
        Exactly calculates the volume fraction of the Voronoic cells.
        Specifically we are calculating 
        :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
        
        :param string distribution: Probability distribution (uniform, normal,
        truncnorm, beta)
        :param float a: mean or alpha (normal/truncnorm, beta)
        :param float b: covariance or beta (normal/truncnorm, beta)
        """
        self.check_num()
        if self._dim != 1:
            raise dim_not_matching("Only applicable for 1D domains.")

        # sort the samples
        sort_ind = np.squeeze(np.argsort(self._values, 0))
        sorted_samples = self._values[sort_ind]
        domain_width = self._domain[:, 1] - self._domain[:, 0]

        # determine the mid_points which are the edges of the associated voronoi
        # cells and bound the cells by the domain
        edges = np.concatenate(([self._domain[:, 0]], (sorted_samples[:-1, :] +\
        sorted_samples[1:, :])*.5, [self._domain[:, 1]]))
        if distribution == 'normal':
            edges = scipy.stats.norm.cdf(edges, loc=a, scale=np.sqrt(b))
        elif distribution == 'truncnorm':
            l = (self._domain[:, 0] - a) / np.sqrt(b)
            r = (self._domain[:, 1] - a) / np.sqrt(b)
            edges = scipy.stats.truncnorm.cdf(edges, a=l, b=r, loc=a,
                    scale=np.sqrt(b)) 
        elif distribution == 'beta':
            edges = scipy.stats.beta.cdf(edges, a=a, b=b, 
                    loc=self._domain[:, 0], scale=domain_width)
        # calculate difference between right and left of each cell and
        # renormalize
        sorted_lam_vol = np.squeeze(edges[1:, :] - edges[:-1, :])
        lam_vol = np.zeros(sorted_lam_vol.shape)
        lam_vol[sort_ind] = sorted_lam_vol
        if distribution == 'uniform':
            lam_vol = lam_vol/domain_width
        self._volumes = lam_vol
        self.global_to_local()

    def estimate_local_volume(self, num_l_emulate_local=100,
            max_num_l_emulate=1e3): 
        r"""

        Estimates the volume fraction of the Voronoice cells associated
        with ``samples``. Specifically we are calculating
        :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
        Here all of the samples are drawn from the generalized Lp uniform
        distribution.

        .. note ::

            If this :class:`~bet.sample.voronoi_sample_set` has exact/estimated
            radii of the Voronoi cell associated with each sample for a domain
            normalized to the unit hypercube (``_normalized_radii``).

        .. todo ::

            When we move away from domains defined on hypercubes this will need
            to be updated to use whatever ``_in_domain`` method exists.

        Volume of the L-p ball is obtained from  Wang, X.. (2005). Volumes of
        Generalized Unit Balls. Mathematics Magazine, 78(5), 390-395.
        `DOI 10.2307/30044198 <http://doi.org/10.2307/30044198>`_
        
        :param int num_l_emulate_local: The number of emulated samples.
        :param int max_num_l_emulate: Maximum number of local emulated samples
        
        """
        self.check_num()
        # normalize the samples
        samples = np.copy(self.get_values())
        self.update_bounds()
        samples = samples - self._left
        samples = samples/self._width

        kdtree = spatial.KDTree(samples)

        # for each sample determine the appropriate radius of the Lp ball (this
        # should be the distance to the farthest neighboring Voronoi cell)
        # calculating this exactly is hard so we will estimate it as follows
        # TODO it is unclear whether to use min, mean, or the first n nearest
        # samples
        sample_radii = None
        if hasattr(self, '_normalized_radii'):
            sample_radii = np.copy(getattr(self, '_normalized_radii'))

        if sample_radii is None:
            # Calculate the pairwise distances
            if not np.isinf(self.p_norm):
                pairwise_distance = spatial.distance.pdist(samples,
                        p=self.p_norm)
            else:
                pairwise_distance = spatial.distance.pdist(samples, p='chebyshev')
            pairwise_distance = spatial.distance.squareform(pairwise_distance)
            pairwise_distance_ma = np.ma.masked_less_equal(pairwise_distance, 0.)
            # Calculate mean, std of pairwise distances
            sample_radii = np.std(pairwise_distance_ma, 0)*3
        elif np.sum(sample_radii <=0) > 0:
            # Calculate the pairwise distances
            if not np.isinf(self.p_norm):
                pairwise_distance = spatial.distance.pdist(samples,
                        p=self.p_norm)
            else:
                pairwise_distance = spatial.distance.pdist(samples, p='chebyshev')
            pairwise_distance = spatial.distance.squareform(pairwise_distance)
            pairwise_distance_ma = np.ma.masked_less_equal(pairwise_distance, 0.)
            # Calculate mean, std of pairwise distances
            # TODO this may be too large/small
            # Estimate radius as 2.*STD of the pairwise distance
            sample_radii[sample_radii <= 0] = np.std(pairwise_distance_ma, 0)*2.

        # determine the volume of the Lp ball
        if not np.isinf(self.p_norm):
            sample_Lp_ball_vol = sample_radii**self._dim * \
                    scipy.special.gamma(1+1./self.p_norm) / \
                    scipy.special.gamma(1+float(self._dim)/self.p_norm)
        else:
            sample_Lp_ball_vol = (2.0*sample_radii)**self._dim

        # Set up local arrays for parallelism
        self.global_to_local()
        lam_vol_local = np.zeros(self._local_index.shape)

        # parallize 
        for i, iglobal in enumerate(self._local_index):
            samples_in_cell = 0
            total_samples = 10
            while samples_in_cell < num_l_emulate_local and \
                    total_samples < max_num_l_emulate:
                total_samples = total_samples*10
                # Sample within an Lp ball until num_l_emulate_local samples are
                # present in the Voronoi cell
                local_lambda_emulate = lp.Lp_generalized_uniform(self._dim,
                        total_samples, self.p_norm, scale=sample_radii[iglobal],
                        loc=samples[iglobal])

                # determine the number of samples in the Voronoi cell (intersected
                # with the input_domain)
                if self._domain is not None:
                    inside = np.all(np.logical_and(local_lambda_emulate >= 0.0,
                            local_lambda_emulate <= 1.0), 1)
                    local_lambda_emulate = local_lambda_emulate[inside]

                (_, emulate_ptr) = kdtree.query(local_lambda_emulate,
                        p=self.p_norm,
                        distance_upper_bound=sample_radii[iglobal])

                samples_in_cell = np.sum(np.equal(emulate_ptr, iglobal))

            # the volume for the Voronoi cell corresponding to this sample is the
            # the volume of the Lp ball times the ratio
            # "num_samples_in_cell/num_total_local_emulated_samples" 
            lam_vol_local[i] = sample_Lp_ball_vol[iglobal]*float(samples_in_cell)\
                    /float(total_samples)

        self.set_volumes_local(lam_vol_local)
        self.local_to_global()

        # normalize by the volume of the input_domain
        domain_vol = np.sum(self.get_volumes())
        self.set_volumes(self._volumes / domain_vol)
        self.set_volumes_local(self._volumes_local / domain_vol)


class sample_set(voronoi_sample_set):
    """
    Set Voronoi cells as the default for now.
    """
                
class discretization(object):
    """
    A data structure to store all of the :class:`~bet.sample.sample_set`
    objects and associated pointers to solve an stochastic inverse problem. 
    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray`
    vector_names = ['_io_ptr', '_io_ptr_local', '_emulated_ii_ptr',
        '_emulated_ii_ptr_local', '_emulated_oo_ptr', '_emulated_oo_ptr_local']
    #: List of attribute names for attributes that are
    #: :class:`sample.sample_set``
    sample_set_names = ['_input_sample_set', '_output_sample_set',
        '_emulated_input_sample_set', '_emulated_output_sample_set',
        '_output_probability_set'] 

 
    def __init__(self, input_sample_set, output_sample_set,
                 output_probability_set=None,
                 emulated_input_sample_set=None,
                 emulated_output_sample_set=None): 
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
        if output_sample_set is not None:
            self.check_nums()
        else:
            logging.info("No output_sample_set")
        
    def check_nums(self):
        """
        
        Checks that ``self._input_sample_set`` and ``self._output_sample_set``
        both have the same number of samples.

        :rtype: int
        :returns: Number of samples

        """
        out_num = self._output_sample_set.check_num()
        in_num = self._input_sample_set.check_num()
        if out_num != in_num:
            raise length_not_matching("input and output lengths do not match")
        else:
            return in_num

    def set_io_ptr(self, globalize=True):
        """
        
        Creates the pointer from ``self._output_sample_set`` to
        ``self._output_probability_set``

        :param bool globalize: flag whether or not to globalize
            ``self._output_sample_set``
        
        """
        if self._output_sample_set._values_local is None:
            self._output_sample_set.global_to_local()
        if self._output_probability_set._kdtree is None:
            self._output_probability_set.set_kdtree()
        (_, self._io_ptr_local) = self._output_probability_set.query(\
                self._output_sample_set._values_local)
                                                            
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
            
        :param bool globalize: flag whether or not to globalize
            ``self._output_sample_set``
        :param int p: Which Minkowski p-norm to use. (1 <= p <= infinity)

        """
        if self._emulated_input_sample_set._values_local is None:
            self._emulated_input_sample_set.global_to_local()
        if self._input_sample_set._kdtree is None:
            self._input_sample_set.set_kdtree()
        (_, self._emulated_ii_ptr_local) = self._input_sample_set.query(\
                self._emulated_input_sample_set._values_local)
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
            
        :param bool globalize: flag whether or not to globalize
            ``self._output_sample_set``
        :param int p: Which Minkowski p-norm to use. (1 <= p <= infinity)

        """
        if self._emulated_output_sample_set._values_local is None:
            self._emulated_output_sample_set.global_to_local()
        if self._output_probability_set._kdtree is None:
            self._output_probability_set.set_kdtree()
        (_, self._emulated_oo_ptr_local) = self._output_probability_set.query(\
                self._emulated_output_sample_set._values_local)
                                                                
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

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.sample.discretization`
        :returns: Copy of this :class:`~bet.sample.discretization`

        """
        my_copy = discretization(self._input_sample_set.copy(),
                self._output_sample_set.copy())
        
        for attrname in discretization.sample_set_names:
            if attrname is not '_input_sample_set' and \
                    attrname is not '_output_sample_set':
                curr_sample_set = getattr(self, attrname)
                if curr_sample_set is not None:
                    setattr(my_copy, attrname, curr_sample_set.copy())
        
        for array_name in discretization.vector_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(my_copy, array_name, np.copy(current_array))
        return my_copy

    def get_input_sample_set(self):
        """

        Returns a reference to the input sample set for this discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: input sample set

        """
        return self._input_sample_set

    def set_input_sample_set(self, input_sample_set):
        """

        Sets the input sample set for this discretization.

        :param input_sample_set: input sample set.
        :type input_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(input_sample_set, sample_set_base):
            self._input_sample_set = input_sample_set
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def get_output_sample_set(self):
        """

        Returns a reference to the output sample set for this discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: output sample set

        """
        return self._output_sample_set

    def set_output_sample_set(self, output_sample_set):
        """

        Sets the output sample set for this discretization.

        :param output_sample_set: output sample set.
        :type output_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(output_sample_set, sample_set_base):
            self._output_sample_set = output_sample_set
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def get_output_probability_set(self):
        """

        Returns a reference to the output probability sample set for this discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: output probability sample set

        """
        return self._output_probability_set

    def set_output_probability_set(self, output_probability_set):
        """

        Sets the output probability sample set for this discretization.

        :param output_probability_set: output probability sample set.
        :type output_probability_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(output_probability_set, sample_set_base):
            output_dims = []
            output_dims.append(output_probability_set.get_dim())
            if hasattr(self, "_output_sample_set"):
                output_dims.append(self._output_sample_set.get_dim())
            if hasattr(self, "_emulated_output_sample_set"):
                output_dims.append(self._emulated_output_sample_set.get_dim())
            if output_dims = 1:
                self._output_probability_set = output_probability_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._output_probability_set = output_probability_set
            else:
                raise dim_not_matching("dimension of values incorrect")
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def get_emulated_output_sample_set(self):
        """

        Returns a reference to the emulated_output sample set for this discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: emulated_output sample set

        """
        return self._emulated_output_sample_set

    def set_emulated_output_sample_set(self, emulated_output_sample_set):
        """

        Sets the emulated_output sample set for this discretization.

        :param emulated_output_sample_set: emupated output sample set.
        :type emulated_output_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(emulated_output_sample_set, sample_set_base):
            output_dims = []
            output_dims.append(emulated_output_probability_set.get_dim())
            if hasattr(self, "_output_sample_set"):
                output_dims.append(self._output_sample_set.get_dim())
            if hasattr(self, "_output_probablity_set"):
                output_dims.append(self._output_probability_set.get_dim())
            if output_dims = 1:
                self._emulated_output_probability_set = emulated_output_probability_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._emulated_output_probability_set = emulated_output_probability_set
            else:
                raise dim_not_matching("dimension of values incorrect")
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def get_emulated_input_sample_set(self):
        """

        Returns a reference to the emulated_input sample set for this discretization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: emulated_input sample set

        """
        return self._emulated_input_sample_set

    def set_emulated_input_sample_set(self, emulated_input_sample_set):
        """

        Sets the emulated_input sample set for this discretization.

        :param emulated_input_sample_set: emupated input sample set.
        :type emulated_input_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(emulated_input_sample_set, sample_set_base):
            input_dims = []
            input_dims.append(emulated_input_probability_set.get_dim())
            if hasattr(self, "_input_sample_set"):
                if self._input_sample_set.get_dim() == \
                        emupated_input_sample_set.get_dim():
                    self._emulated_input_probability_set = emulated_input_probability_set
                else:
                    raise dim_not_matching("dimension of values incorrect")
            else:
                self._emulated_input_probability_set = emulated_input_probability_set
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")
