# Copyright (C) 2016 The BET Development Team

"""
This module contains data structure/storage classes for BET. Notably:
    :class:`bet.sample.sample_set`
    :class:`bet.sample.discretization`
    :class:`bet.sample.length_not_matching`
    :class:`bet.sample.dim_not_matching`
"""

import os, logging, glob
import numpy as np
import scipy.spatial as spatial
import scipy.io as sio
import scipy.stats
from bet.Comm import comm, MPI
import bet.util as util
import bet.sampling.LpGeneralizedSamples as lp
import numpy.linalg as linalg

class length_not_matching(Exception):
    """
    Exception for when the length of the array is inconsistent.
    """
    

class dim_not_matching(Exception):
    """
    Exception for when the dimension of the array is inconsistent.
    """
    
def save_sample_set(save_set, file_name, sample_set_name=None, globalize=False):
    """
    Saves this :class:`bet.sample.sample_set` as a ``.mat`` file. Each
    attribute is added to a dictionary of names and arrays which are then
    saved to a MATLAB-style file.

    :param save_set: sample set to save
    :type save_set: :class:`bet.sample.sample_set_base`
    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string sample_set_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.sample_set_base` objects to a single
        ``.mat`` file
    :param bool globalize: flag whether or not to globalize

    :rtype: string
    :returns: local file name

    """
    # create processor specific file name
    if comm.size > 1 and not globalize:
        local_file_name = os.path.join(os.path.dirname(file_name),
                "proc{}_{}".format(comm.rank, os.path.basename(file_name)))
    else:
        local_file_name = file_name

    # globalize
    if globalize and save_set._values_local is not None:
        save_set.local_to_global()
    comm.barrier()

    # create temporary dictionary
    new_mdat = dict()

    # store sample set in dictionary
    if sample_set_name is None:
        sample_set_name = 'default'
    for attrname in save_set.vector_names:
        curr_attr = getattr(save_set, attrname)
        if curr_attr is not None:
            new_mdat[sample_set_name+attrname] = curr_attr
    for attrname in save_set.all_ndarray_names:
        curr_attr = getattr(save_set, attrname)
        if curr_attr is not None:
            new_mdat[sample_set_name+attrname] = curr_attr
    new_mdat[sample_set_name + '_sample_set_type'] = save_set.__class__.__name__
    comm.barrier()

    # save new file or append to existing file
    if (globalize and comm.rank == 0) or not globalize:
        if os.path.exists(local_file_name) or \
                os.path.exists(local_file_name+'.mat'):
            mdat = sio.loadmat(local_file_name)
            new_mdat.update(mdat)
            sio.savemat(local_file_name, new_mdat)
        else:
            sio.savemat(local_file_name, new_mdat)
    comm.barrier()
    return local_file_name

def load_sample_set(file_name, sample_set_name=None, localize=True):
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
    :param bool localize: Flag whether or not to re-localize arrays. If
        ``file_name`` is prepended by ``proc_{}`` localize is set to ``False``.

    :rtype: :class:`~bet.sample.sample_set`
    :returns: the ``sample_set`` that matches the ``sample_set_name``
    
    """
    # check to see if parallel file name
    if file_name.startswith('proc_'):
        localize = False
    elif not os.path.exists(file_name) and os.path.exists(os.path.join(\
            os.path.dirname(file_name), "proc{}_0".format(\
                os.path.basename(file_name)))):
        return load_sample_set_parallel(file_name, sample_set_name)

    mdat = sio.loadmat(file_name)
    if sample_set_name is None:
        sample_set_name = 'default'
    
    if sample_set_name+"_dim" in mdat.keys():
        loaded_set = eval(mdat[sample_set_name + '_sample_set_type'][0])(
            np.squeeze(mdat[sample_set_name+"_dim"]))
    else:
        logging.info("No sample_set named {} with _dim in file".\
                format(sample_set_name))
        return None

    for attrname in loaded_set.vector_names:
        if attrname is not '_dim':
            if sample_set_name+attrname in mdat.keys():
                setattr(loaded_set, attrname,
                    np.squeeze(mdat[sample_set_name+attrname]))
    for attrname in loaded_set.all_ndarray_names:
        if sample_set_name+attrname in mdat.keys():
            setattr(loaded_set, attrname, mdat[sample_set_name+attrname])

    if localize:
        # re-localize if necessary
        loaded_set.global_to_local()
    
    return loaded_set

def load_sample_set_parallel(file_name, sample_set_name=None):
    """
    Loads a :class:`~bet.sample.sample_set` from a ``.mat`` file in parallel
    and correctly re-localizes data if necessary. If a file contains multiple
    :class:`~bet.sample.sample_set` objects then ``sample_set_name`` is used to
    distinguish which between different :class:`~bet.sample.sample_set`
    objects.

    :param string file_name: Name of the ``.mat`` file, no extension is
        needed.
    :param string sample_set_name: String to prepend to attribute names when
        saving multiple :class`bet.sample.sample_set` objects to a single
        ``.mat`` file

    :rtype: :class:`~bet.sample.sample_set`
    :returns: the ``sample_set`` that matches the ``sample_set_name``
    """
   
    if sample_set_name is None:
            sample_set_name = 'default'
   # Find and open save files
    save_dir = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    mdat_files = glob.glob(os.path.join(save_dir,
            "proc*_{}".format(base_name)))
    
    if len(mdat_files) == comm.size:
        logging.info("Loading {} sample set using parallel files (same nproc)"\
                .format(sample_set_name))
        # if the number of processors is the same then set mdat to
        # be the one with the matching processor number (doesn't
        # really matter)
        local_file_name = os.path.join(os.path.dirname(file_name),
                "proc{}_{}".format(comm.rank, os.path.basename(file_name)))
        return load_sample_set(local_file_name, sample_set_name)
    else:
        logging.info("Loading {} sample set using parallel files (diff nproc)"\
            .format(sample_set_name))        
                # Determine how many processors the previous data used
        # otherwise gather the data from mdat and then scatter
        # among the processors and update mdat
        mdat_files_local = comm.scatter(mdat_files)
        mdat_local = [sio.loadmat(m) for m in mdat_files_local]
        mdat_list = comm.allgather(mdat_files_local)
        mdat_global = []
        # instead of a list of lists, create a list of mdat
        for mlist in mdat_list: 
            mdat_global.extend(mlist)
        # get num_proc and num_chains_pproc for previous run
        old_num_proc = max((len(mdat_list), 1))
        
        if sample_set_name+"_dim" in mdat_global[0].keys():
            loaded_set = eval(mdat_global[0][sample_set_name + \
                    '_sample_set_type'][0])(
                    np.squeeze(mdat_global[0][sample_set_name+"_dim"]))
        else:
            logging.info("No sample_set named {} with _dim in file".\
                    format(sample_set_name))
            return None

        # load attributes
        for attrname in loaded_set.vector_names:
            if attrname is not '_dim':
                if sample_set_name+attrname in mdat_global[0].keys():
                    # create lists of local data
                    if attrname.endswith('_local'): 
                        temp_input = []
                        for mdat in mdat_global:
                            temp_input.append(np.squeeze(mdat[sample_set_name+attrname]))
                        # turn into arrays
                        temp_input = np.concatenate(temp_input)
                    else:
                        temp_input = np.squeeze(mdat[sample_set_name+attrname])
                    setattr(loaded_set, attrname, temp_input) 
        for attrname in loaded_set.all_ndarray_names:
            if sample_set_name+attrname in mdat_global[0].keys():
                if attrname.endswith('_local'): 
                    # create lists of local data
                    temp_input = []
                    for mdat in mdat_global:
                        temp_input.append(mdat[sample_set_name+attrname])
                    # turn into arrays
                    temp_input = np.concatenate(temp_input)
                else:
                    temp_input = mdat[sample_set_name+attrname]
                setattr(loaded_set, attrname, temp_input)

        # re-localize if necessary
        loaded_set.local_to_global()


class sample_set_base(object):
    """

    A data structure containing arrays specific to a set of samples.

    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray` or int/float
    vector_names = ['_probabilities', '_probabilities_local', '_volumes',
                    '_volumes_local', '_local_index', '_dim', '_p_norm',
                    '_radii', '_normalized_radii', '_region', '_region_local',
                    '_error_id', '_error_id_local']
    #: List of global attribute names for attributes that are 
    #: :class:`numpy.ndarray`
    array_names = ['_values', '_volumes', '_probabilities', '_jacobians',
                   '_error_estimates', '_right', '_left', '_width',
                   '_kdtree_values', '_radii', '_normalized_radii',
                   '_region', '_error_id'] 
    #: List of attribute names for attributes that are
    #: :class:`numpy.ndarray` with dim > 1
    all_ndarray_names = ['_error_estimates', '_error_estimates_local',
                         '_values', '_values_local', '_left', '_left_local', 
                         '_right', '_right_local', '_width', '_width_local', 
                         '_domain', '_kdtree_values', '_jacobians', 
                         '_jacobians_local'] 


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
        #: p-norm for discretization
        self._p_norm = 2.0
        #: :class:`numpy.ndarray` of sample radii of shape (num,)
        self._radii = None
        #: :class:`numpy.ndarray` of sample radii of shape (local_num,)
        self._radii_local = None
        #: :class:`numpy.ndarray` of normalized sample radii of shape (num,)
        self._normalized_radii = None
        #: :class:`numpy.ndarray` of normalized sample radii of shape (local_num,)
        self._normalized_radii_local = None
        #: :class:`numpy.ndarray` of integers marking regions of the domain
        self._region = None
        #: :class:`numpy.ndarray` of integers marking regions of the domain
        self._region_local = None
        #: :class:`numpy.ndarray` of error identifiers  of shape (num,)
        self._error_id = None
        #: :class:`numpy.ndarray` of error identifiers  of shape (local_num,)
        self._error_id_local = None

    def set_p_norm(self, p_norm):
        """
        Sets p-norm for sample set.

        :param float p_norm: p-norm to use

        """
        self._p_norm = p_norm

    def get_p_norm(self):
        """
        Returns p-norm for sample set
        """
        return self._p_norm

    def set_region(self, region):
        """
        Sets region for sample set.

        :param region: array of regions
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._region = region

    def get_region(self):
        """
        Returns region.
        """
        return self._region

    def set_region_local(self, region):
        """
        Sets local region for sample set.

        :param region: array of regions
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._region_local = region

    def get_region_local(self):
        """
        Returns local region.
        """
        return self._region_local  

    def set_error_id(self, error_id):
        """
        Sets error_id for sample set.

        :param error_id: array of error identifiers
        :type error_id: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._error_id = error_id

    def get_error_id(self):
        """
        Returns error identifiers.
        """
        return self._error_id

    def set_error_id_local(self, error_id):
        """
        Sets local error id for sample set.

        :param error_id: array of error identifiers
        :type error_id: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        self._error_local = error_id

    def get_error_id_local(self):
        """
        Returns local error identifier.
        """
        return self._error_id_local
        
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
        for array_name in self.array_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                if num is None:
                    num = current_array.shape[0]
                    first_array = array_name
                else:
                    if num != current_array.shape[0]:
                        errortxt = "length of {} inconsistent with {}"
                        raise length_not_matching(errortxt.format(array_name,
                                                  first_array)) 
        if self._values is not None and self._values.shape[1] != self._dim:
            raise dim_not_matching("dimension of values incorrect")
            
        if num is None:
            num_local = self.check_num_local()
            if num_local is None:
                num_local = 0
            num = comm.allreduce(num_local, op=MPI.SUM)
           
        return num

    def check_num_local(self):
        """
        
        Checks that the number of entries in ``self._values_local``,
        ``self._volumes_local``, ``self._probabilities_local``, 
        ``self._jacobians_local``, and ``self._error_estimates_local`` 
        all match (assuming the named array exists).
        
        :rtype: int
        :returns: num

        """
        num = None
        for array_name in self.array_names:
            array_name_local = array_name + "_local"
            current_array = getattr(self, array_name_local)
            if current_array is not None:
                if num is None:
                    num = current_array.shape[0]
                    first_array = array_name
                else:
                    if num != current_array.shape[0]:
                        errortxt = "length of {} inconsistent with {}"
                        raise length_not_matching(errortxt.format(array_name,
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
        for array_name in self.array_names:
            current_array_local = getattr(self, array_name + "_local")
            if current_array_local is not None:
                setattr(self, array_name,
                        util.get_global_values(current_array_local))

    def query(self, x, k=1):
        """
        Identify which value points x are associated with for discretization.

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param int k: number of nearest neighbors to return

        """
        pass

    def estimate_volume(self, n_mc_points=int(1E4)):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration. 

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

    def estimate_volume_emulated(self, emulated_sample_set):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration.

        .. note ::

            This could be re-written to just use an ``emulated_ii_ptr`` instead
            of an ``emulated_sample_set``.

        :param emulated_sample_set: The set of samples used to approximate the
            volume measure.
        :type emulated_sample_set: :class:`bet.sample.sample_set_base`

        """
        num = self.check_num()

        if emulated_sample_set._values_local is None:
            emulated_sample_set.global_to_local()

        (_, emulate_ptr) = self.query(emulated_sample_set._values_local)

        vol = np.zeros((num,))
        for i in range(num):
            vol[i] = np.sum(np.equal(emulate_ptr, i))
        cvol = np.copy(vol)
        comm.Allreduce([vol, MPI.DOUBLE], [cvol, MPI.DOUBLE], op=MPI.SUM)
        num_emulate = emulated_sample_set._values_local.shape[0]
        num_emulate = comm.allreduce(num_emulate, op=MPI.SUM)
        vol = cvol
        vol = vol/float(num_emulate)
        self._volumes = vol
        self.global_to_local()

    def estimate_volume_mc(self, globalize=True):
        """
        Give all cells the same volume fraction based on the Monte Carlo
        assumption.  
        """
        num = self.check_num()
        if globalize:
            self._volumes = 1.0/float(num)*np.ones((num,))
            self.global_to_local()
        else:
            num_local = self.check_num_local()
            self._volumes_local = 1.0/float(num)*np.ones((num_local,))

    def global_to_local(self):
        """
        Makes local arrays from available global ones.
        """
        num = self.check_num()
        global_index = np.arange(num, dtype=np.int)
        self._local_index = np.array_split(global_index, comm.size)[comm.rank]
        for array_name in self.array_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(self, array_name + "_local",
                        np.array_split(current_array, comm.size)[comm.rank])
        comm.barrier()

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: Copy of this :class:`~bet.sample.sample_set_base`

        """
        my_copy = eval(self.__class__.__name__)(self.get_dim())
        for array_name in self.all_ndarray_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(my_copy, array_name,
                        np.copy(current_array))
        for vector_name in self.vector_names:
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

def save_discretization(save_disc, file_name, discretization_name=None,
        globalize=False):
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
    :param bool globalize: flag whether or not to globalize
        :class:`bet.sample.sample_set_base` objects stored in this
        discretization

    :rtype: string
    :returns: local file name

    """
    # create temporary dictionary
    new_mdat = dict()

    # create processor specific file name
    if comm.size > 1 and not globalize:
        local_file_name = os.path.join(os.path.dirname(file_name),
                "proc{}_{}".format(comm.rank, os.path.basename(file_name)))
    else:
        local_file_name = file_name

    # set name if doesn't exist
    if discretization_name is None:
        discretization_name = 'default'

    # globalize the pointers
    if globalize:
        save_disc.globalize_ptrs()
    # save sample sets if they exist
    for attrname in discretization.sample_set_names:
        curr_attr = getattr(save_disc, attrname)
        if curr_attr is not None:
            if attrname in discretization.sample_set_names:
                save_sample_set(curr_attr, file_name,
                    discretization_name+attrname, globalize)
    
    # store discretization in dictionary
    for attrname in discretization.vector_names:
        curr_attr = getattr(save_disc, attrname)
        if curr_attr is not None:
            new_mdat[discretization_name+attrname] = curr_attr
    comm.barrier()

    # save new file or append to existing file
    if (globalize and comm.rank == 0) or not globalize:
        if os.path.exists(local_file_name) or \
                os.path.exists(local_file_name+'.mat'):
            mdat = sio.loadmat(local_file_name)
            new_mdat.update(mdat)
            sio.savemat(local_file_name, new_mdat)
        else:
            sio.savemat(local_file_name, new_mdat)
    comm.barrier()
    return local_file_name

def load_discretization_parallel(file_name, discretization_name=None):
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
    # Find and open save files
    save_dir = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    mdat_files = glob.glob(os.path.join(save_dir,
            "proc*_{}".format(base_name)))

    if len(mdat_files) == comm.size:
        logging.info("Loading {} sample set using parallel files (same nproc)"\
                .format(discretization_name))
        # if the number of processors is the same then set mdat to
        # be the one with the matching processor number (doesn't
        # really matter)
        return load_discretization(mdat_files[comm.rank], discretization_name)
    else:
        logging.info("Loading {} sample set using parallel files (diff nproc)"\
            .format(discretization_name)) 
        
        if discretization_name is None:
            discretization_name = 'default'

        input_sample_set = load_sample_set(file_name,
                discretization_name+'_input_sample_set')

        output_sample_set = load_sample_set(file_name,
                discretization_name+'_output_sample_set')

        loaded_disc = discretization(input_sample_set, output_sample_set)
       
        # Determine how many processors the previous data used
        # otherwise gather the data from mdat and then scatter
        # among the processors and update mdat
        mdat_files_local = comm.scatter(mdat_files)
        mdat_local = [sio.loadmat(m) for m in mdat_files_local]
        mdat_list = comm.allgather(mdat_local)
        mdat_global = []
        # instead of a list of lists, create a list of mdat
        for mlist in mdat_list: 
            mdat_global.extend(mlist)
        
        # load attributes
        for attrname in discretization.vector_names:
            if discretization_name+attrname in mdat_global[0].keys():
                if attrname.endswith('_local') and comm.size != len(mdat_list): 
                    # create lists of local data
                    temp_input = None 
                else:
                        temp_input = np.squeeze(mdat[discretization_name+attrname])
                setattr(loaded_disc, attrname, temp_input) 
        
        # load sample sets
        for attrname in discretization.sample_set_names:
            if attrname is not '_input_sample_set' and \
                    attrname is not '_output_sample_set':
                setattr(loaded_disc, attrname, load_sample_set(file_name,
                        discretization_name+attrname))
        
        # re-localize if necessary
        if file_name.startswith('proc_') and comm.size > 1 \
                and comm.size != len(mdat_list):
            warn_string = "Local pointers have been removed and will be"
            warn_string += " re-created as necessary)"
            warnings.warn(warn_string)
            #loaded_disc._io_ptr_local = None
            #loaded_disc._emulated_ii_ptr_local = None
            #loaded_disc._emulated_oo_ptr_local = None
    return loaded_disc

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

    # check to see if parallel file name
    if file_name.startswith('proc_'):
        pass
    elif not os.path.exists(file_name) and os.path.exists(os.path.join(\
            os.path.dirname(file_name), "proc{}_{}".format(comm.rank,
                os.path.basename(file_name)))):
        return load_discretization_parallel(file_name, discretization_name)

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
    
    # re-localize if necessary
    if file_name.rfind('proc_') == 0 and comm.size > 1:
        loaded_disc._io_ptr_local = None
        loaded_disc._emulated_ii_ptr_local = None
        loaded_disc._emulated_oo_ptr_local = None
            
    return loaded_disc


class voronoi_sample_set(sample_set_base):
    """

    A data structure containing arrays specific to a set of samples defining
    a Voronoi tesselation.

    """

    def query(self, x, k=1):
        """
        Identify which value points x are associated with for discretization.

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param int k: number of nearest neighbors to return

        :rtype: tuple
        :returns: (dist, ptr)
        """
        if self._kdtree is None:
            self.set_kdtree()
        else:
            self.check_num()
       
        (dist, ptr) = self._kdtree.query(x, p=self._p_norm, k=k)
        return (dist, ptr)

    def exact_volume_1D(self):
        r"""
        
        Exactly calculates the volume fraction of the Voronoic cells.
        Specifically we are calculating 
        :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
        
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
        # calculate difference between right and left of each cell and
        # renormalize
        sorted_lam_vol = np.squeeze(edges[1:, :] - edges[:-1, :])
        lam_vol = np.zeros(sorted_lam_vol.shape)
        lam_vol[sort_ind] = sorted_lam_vol
        lam_vol = lam_vol/domain_width
        self._volumes = lam_vol
        self.global_to_local()

    def estimate_radii(self, n_mc_points=int(1E4), normalize=True):
        """
        Calculate the radii of cells approximately using Monte
        Carlo integration. 

        .. todo::

           This currently presumes a uniform Lesbegue measure on the
           ``domain``. Currently the way this is written
           ``emulated_input_sample_set`` is NOT used to calculate the volume.
           This should at least be an option. 

        :param int n_mc_points: If estimate is True, number of MC points to use
        :param bool normalize: estimate normalized radius

        """
        num = self.check_num()

        samples = np.copy(self.get_values())
        n_mc_points_local = (n_mc_points/comm.size) + \
                            (comm.rank < n_mc_points%comm.size)

        # normalize the samples
        if normalize:
            self.update_bounds()
            samples = samples - self._left
            samples = samples/self._width

        width = self._domain[:, 1] - self._domain[:, 0]
        mc_points = width*np.random.random((n_mc_points_local,
                self._domain.shape[0])) + self._domain[:, 0]

        (_, emulate_ptr) = self.query(mc_points)

        if normalize:
            self.update_bounds(n_mc_points_local)
            mc_points = mc_points - self._left
            mc_points = mc_points/self._width
            self._left = None
            self._right = None
            self._width = None

        rad = np.zeros((num,))

        for i in range(num):
            rad[i] = np.max(np.linalg.norm(mc_points[np.equal(emulate_ptr, i),\
                :] - samples[i, :], ord=self._p_norm, axis=1))

        crad = np.copy(rad)
        comm.Allreduce([rad, MPI.DOUBLE], [crad, MPI.DOUBLE], op=MPI.MAX)
        rad = crad

        if normalize:
            self._normalized_radii = rad
        else:
            self._radii = rad
        
        self.global_to_local()

    def estimate_radii_and_volume(self, n_mc_points=int(1E4), normalize=True):
        """
        Calculate the radii and volume faction of cells approximately using Monte
        Carlo integration. 

        .. todo::

           This currently presumes a uniform Lesbegue measure on the
           ``domain``. Currently the way this is written
           ``emulated_input_sample_set`` is NOT used to calculate the volume.
           This should at least be an option. 

        :param int n_mc_points: If estimate is True, number of MC points to use
        :param bool normalize: estimate normalized radius

        """
        num = self.check_num()

        samples = np.copy(self.get_values())
        n_mc_points_local = (n_mc_points/comm.size) + \
                            (comm.rank < n_mc_points%comm.size)

        # normalize the samples
        if normalize:
            self.update_bounds()
            samples = samples - self._left
            samples = samples/self._width
        
        width = self._domain[:, 1] - self._domain[:, 0]
        mc_points = width*np.random.random((n_mc_points_local,
                self._domain.shape[0])) + self._domain[:, 0]

        (_, emulate_ptr) = self.query(mc_points)

        if normalize:
            self.update_bounds(n_mc_points_local)
            mc_points = mc_points - self._left
            mc_points = mc_points/self._width
            self._left = None
            self._right = None
            self._width = None

        vol = np.zeros((num,))
        rad = np.zeros((num,))
        for i in range(num):
            vol[i] = np.sum(np.equal(emulate_ptr, i))
            rad[i] = np.max(np.linalg.norm(mc_points[np.equal(emulate_ptr, i),\
                :] - samples[i, :], ord=self._p_norm, axis=1))

        crad = np.copy(rad)
        comm.Allreduce([rad, MPI.DOUBLE], [crad, MPI.DOUBLE], op=MPI.MAX)
        rad = crad

        if normalize:
            self._normalized_radii = rad
        else:
            self._radii = rad

        cvol = np.copy(vol)
        comm.Allreduce([vol, MPI.DOUBLE], [cvol, MPI.DOUBLE], op=MPI.SUM)
        vol = cvol
        vol = vol/float(n_mc_points)
        self._volumes = vol
        self.global_to_local()

    def estimate_local_volume(self, num_emulate_local=500,
            max_num_emulate=int(1e4)): 
        r"""

        Estimates the volume fraction of the Voronoice cells associated
        with ``samples``. Specifically we are calculating
        :math:`\mu_\Lambda(\mathcal(V)_{i,N} \cap A)/\mu_\Lambda(\Lambda)`.
        Here all of the samples are drawn from the generalized Lp uniform
        distribution.

        .. note ::

            If this :class:`~bet.sample.voronoi_sample_set` has exact/estimated
            radii of the Voronoi cell associated with each sample for a domain
            normalized to the unit hypercube (``_normalized_radii``). Note that
            these are not centroidal Voronoi tesselations meaning that the
            centroid is NOT the generator of the Voronoi cell. What we desire
            for the radius is actually 
            :math:`sup_{\lambda \in \mathcal{V}_{i, N}} d_v(\lambda, \lambda^{(i)})`.

        .. todo ::

            When we move away from domains defined on hypercubes this will need
            to be updated to use whatever ``_in_domain`` method exists.

        Volume of the L-p ball is obtained from  Wang, X.. (2005). Volumes of
        Generalized Unit Balls. Mathematics Magazine, 78(5), 390-395.
        `DOI 10.2307/30044198 <http://doi.org/10.2307/30044198>`_
        
        :param int num_emulate_local: The number of emulated samples.
        :param int max_num_emulate: Maximum number of local emulated samples
        
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
        if self._normalized_radii is not None:
                sample_radii = np.copy(self._normalized_radii)
    
        if sample_radii is None:
            num_mc_points = np.max([1e4, samples.shape[0]*20])
            self.estimate_radii(n_mc_points=int(num_mc_points)) 
            sample_radii = 1.5*np.copy(self._normalized_radii)
        if np.sum(sample_radii <= 0) > 0:
            # Calculate the pairwise distances
            if not np.isinf(self._p_norm):
                pairwise_distance = spatial.distance.pdist(samples,
                        p=self._p_norm)
            else:
                pairwise_distance = spatial.distance.pdist(samples, p='chebyshev')
            pairwise_distance = spatial.distance.squareform(pairwise_distance)
            pairwise_distance_ma = np.ma.masked_less_equal(pairwise_distance, 0.)
            prob_est_radii = np.std(pairwise_distance_ma*.5, 0)*2.
            # Calculate mean, std of pairwise distances
            # TODO this may be too large/small
            # Estimate radius as 2.*STD of the pairwise distance
            sample_radii[sample_radii <= 0] = prob_est_radii[sample_radii <= 0] 

        # determine the volume of the Lp ball
        if not np.isinf(self._p_norm):
            sample_Lp_ball_vol = sample_radii**self._dim * \
                    scipy.special.gamma(1+1./self._p_norm) / \
                    scipy.special.gamma(1+float(self._dim)/self._p_norm)
        else:
            sample_Lp_ball_vol = (2.0*sample_radii)**self._dim

        # Set up local arrays for parallelism
        self.global_to_local()
        lam_vol_local = np.zeros(self._local_index.shape)

        # parallize

        for i, iglobal in enumerate(self._local_index):
            samples_in_cell = 0
            total_samples = 10
            while samples_in_cell < num_emulate_local and \
                    total_samples < max_num_emulate:
                total_samples = total_samples*10
                # Sample within an Lp ball until num_emulate_local samples are
                # present in the Voronoi cell
                local_lambda_emulate = lp.Lp_generalized_uniform(self._dim,
                        total_samples, self._p_norm, scale=sample_radii[iglobal],
                        loc=samples[iglobal])

                # determine the number of samples in the Voronoi cell (intersected
                # with the input_domain)
                if self._domain is not None:
                    inside = np.all(np.logical_and(local_lambda_emulate >= 0.0,
                            local_lambda_emulate <= 1.0), 1)
                    local_lambda_emulate = local_lambda_emulate[inside]

                (_, emulate_ptr) = kdtree.query(local_lambda_emulate,
                        p=self._p_norm,
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

class rectangle_sample_set(sample_set_base):
    r"""
    A data structure containing arrays specific to a set of samples defining
    a hyperrectangle discretization.

    A series of n hyperrectangles :math:`A_i \subset \Lambda` with 
    :math:`A_i \cap A_j = \emptyset` 
    for :math:`i \neq j`. The last entry represents the remainder 
    :math:`\Lambda \setminus ( \cup_{i-1}^n A_i)`.
    
    """

    def setup(self, maxes, mins):
        """

        Initialization

        :param maxes: array or list of maxes for hyperrectangles
        :type maxes: iterable with components of length dim
        :param mins: array or list of mins for hyperrectangles
        :type mins: iterable with components of length dim

        """
        # Check dimensions
        if len(maxes) != len(mins):
            raise length_not_matching("Different number of maxes and mins")
        for i in range(len(maxes)):
            if (len(maxes[i]) != self._dim) or (len(mins[i]) != self._dim):
                raise length_not_matching("Rectangle " + `i` + " has the wrong number of entries.")
                
        values = np.zeros((len(maxes)+1, self._dim))
        self._right = np.zeros((len(maxes)+1, self._dim))
        self._left = np.zeros((len(mins)+1, self._dim))
        for i in range(len(maxes)):
            values[i,:] = 0.5*(np.array(maxes[i]) + np.array(mins[i]))
            self._right[i,:] = maxes[i]
            self._left[i,:] = mins[i]
        values[-1,:] = np.inf
        self._right[-1,:] = np.inf
        self._left[-1,:] = -np.inf
        self._width = self._right - self._left
        self.set_values(values)
        if len(maxes) > 1:
            logging.warning("If rectangles intersect on a set nonzero measure, calculated values will be wrong.")
        self._region = np.arange(len(maxes) + 1) 

        
                    
    def update_bounds(self, num=None):
        """
        Does nothing for this type of sample set.
        
        """
        logging.warning("Bounds cannot be updated for this type of sample set.")

        pass

    def update_bounds_local(self, num_local=None):
        """
        Does nothing for this type of sample set.
        
        """
        logging.warning("Bounds cannot be updated for this type of sample set.")

        pass
    def append_values(self, values):
        """
        Does nothing for this type of sample_set.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values: values to append
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        logging.warning("Values cannot be appended for this type of sample set.")
        pass

    def append_values_local(self, values_local):
        """
        Does nothing for this type of sample_set.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values_local: values to append
        :type values_local: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        logging.warning("Values cannot be appended for this type of sample set.")
        pass

    def append_jacobians(self, new_jacobians):
        """
        Does nothing for this type of sample set. 

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_jacobians: New jacobians to append.
        :type new_jacobians: :class:`numpy.ndarray` of shape (num, other_dim, 
            dim)

        """
        logging.warning("Values cannot be appended for this type of sample set.")
        pass

    def append_error_estimates(self, new_error_estimates):
        """
        Does nothing for this type of sample set.

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_error_estimates: New error_estimates to append.
        :type new_error_estimates: :class:`numpy.ndarray` of shape (num,)

        """
        logging.warning("Values cannot be appended for this type of sample set.")
        pass
        
    def query(self, x, k=1):
        """
        Identify which value points x are associated with for discretization.
        Only returns the neighbors for which :math:`x_i \in A_k`. The distance
        is set to 0 if it is in the rectangle and infinity if it is not.
        It is only considered in or out.

        .. seealso::

            :meth:`scipy.spatial.KDTree.query`

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param int k: number of nearest neighbors to return
        :rtype: tuple
        :returns: (dist, ptr)

        """
        num = self.check_num()
        dist = np.inf * np.ones((x.shape[0], k), dtype=np.float)
        pt = (num - 1) * np.ones((x.shape[0], k), dtype=np.int)
        for i in range(num - 1):
            in_r = np.all(np.less_equal(x, self._right[i,:]), axis=1)
            in_l = np.all(np.greater(x, self._left[i,:]), axis=1)
            in_rec = np.logical_and(in_r, in_l)
            for j in range(k):
                if j == 0:
                    in_rec_now = np.logical_and(np.equal(pt[:,j],num-1), in_rec)
                else:
                    in_rec_now = np.logical_and(np.logical_and(np.equal(pt[:,j],num-1), in_rec), np.not_equal(pt[:,j-1],i))
                pt[:,j][in_rec_now]  = i
                dist[:,j][in_rec_now] = 0.0
        if k == 1:
            dist = dist[:, 0]
            pt = pt[:, 0]
            
        
        return (dist, pt)

    def exact_volume_lebesgue(self):
        r"""
        
        Exactly calculates the Lebesgue volume fraction of the cells.

        """
        num = self.check_num()
        self._volumes = np.zeros((num, ))
        domain_width = self._domain[:, 1] - self._domain[:, 0]
        self._volumes[0:-1] = np.prod(self._width[0:-1]/domain_width, axis=1)
        self._volumes[-1] = 1.0 - np.sum(self._volumes[0:-1])

class ball_sample_set(sample_set_base):
    r"""
    A data structure containing arrays specific to a set of samples defining
    discretization containing a number of balls.
    Only returns the neighbors for which :math:`x_i \in A_k`.

    A series of n balls :math:`A_i \subset \Lambda` with 
    :math:`A_i \cap A_j = \emptyset` 
    for :math:`i \neq j`. The last entry represents the remainder 
    :math:`\Lambda \setminus ( \cup_{i-1}^n A_i)`.
    
    """
    def setup(self, centers, radii):
        """
        Initialize.
        
        :param centers: centers of balls
        :type centers: iterable of shape (num-1, dim)
        :param radii: radii of balls
        :type radii: iterable of length num-1
        
        """
        if len(centers) != len(radii):
            raise length_not_matching("Different number of centers and radii.")
        for i in range(len(centers)):
            if (len(centers[i]) != self._dim):
                raise length_not_matching("Center " + `i` + " has the wrong number of entries.")
        values = np.zeros((len(centers)+1, self._dim))
        values[0:-1,:] = centers
        values[-1,:] = np.nan
        self.set_values(values)
        self._radii = np.zeros((len(centers)+1,))
        self._radii[0:-1] = radii
        self._radii[-1] = np.inf
        if len(centers) > 1:
            logging.warning("If balls intersect on a set nonzero measure, calculated values will be wrong.")
        self._region = np.arange(len(centers) + 1)

    def append_values(self, values):
        """
        Does nothing for this type of sample_set.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values: values to append
        :type values: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        logging.warning("Values cannot be appended for this type of sample set.")
        pass

    def append_values_local(self, values_local):
        """
        Does nothing for this type of sample_set.

        .. seealso::

            :meth:`numpy.concatenate`

        :param values_local: values to append
        :type values_local: :class:`numpy.ndarray` of shape (some_num, dim)
        """
        logging.warning("Values cannot be appended for this type of sample set.")
        pass

    def append_jacobians(self, new_jacobians):
        """
        Does nothing for this type of sample set. 

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_jacobians: New jacobians to append.
        :type new_jacobians: :class:`numpy.ndarray` of shape (num, other_dim, 
            dim)

        """
        logging.warning("Values cannot be appended for this type of sample set.")
        pass

    def append_error_estimates(self, new_error_estimates):
        """
        Does nothing for this type of sample set.

        .. note::

            Remember to update the other member attribute arrays so that
            :meth:`~sample.sample.check_num` does not fail.

        :param new_error_estimates: New error_estimates to append.
        :type new_error_estimates: :class:`numpy.ndarray` of shape (num,)

        """
        logging.warning("Values cannot be appended for this type of sample set.")
        pass

    def update_bounds(self, num=None):
        """
        Does nothing for this type of sample set.
        
        """
        logging.warning("Bounds cannot be updated for this type of sample set.")

        pass

    def update_bounds_local(self, num_local=None):
        """
        Does nothing for this type of sample set.
        
        """
        logging.warning("Bounds cannot be updated for this type of sample set.")

        pass
        
    def query(self, x, k=1):
        """
        Identify which value points x are associated with for discretization.
        The distance is set to 0 if it is in the rectangle and infinity 
        if it is not.
        It is only considered in or out.

        .. seealso::

            :meth:`scipy.spatial.KDTree.query`

        :param x: points for query
        :type x: :class:`numpy.ndarray` of shape ``(*, dim)``
        :param int k: number of nearest neighbors to return
        :rtype: tuple
        :returns: (dist, ptr)
        """
        num = self.check_num()
        dist = np.inf * np.ones((x.shape[0], k), dtype=np.float)
        pt = (num - 1) * np.ones((x.shape[0], k), dtype=np.int)
        for i in range(num - 1):
            in_rec = np.less(linalg.norm(x-self._values[i,:], self._p_norm, axis=1), self._radii[i])
            for j in range(k):
                if j == 0:
                    in_rec_now = np.logical_and(np.equal(pt[:,j],num-1), in_rec)
                else:
                    in_rec_now = np.logical_and(np.logical_and(np.equal(pt[:,j],num-1), in_rec), np.not_equal(pt[:,j-1],i))
                pt[:,j][in_rec_now]  = i
                dist[:,j][in_rec_now] = 0.0
        if k == 1:
            dist = dist[:, 0]
            pt = pt[:, 0]
        
        return (dist, pt)

    def exact_volume(self):
        """
        Calculate the exact volume fraction given the given p-norm.
        
         
        """
        num = self.check_num()
        self._volumes = np.zeros((num, ))
        domain_vol = np.product(self._domain[:, 1] - self._domain[:, 0])
        self._volumes[0:-1] = 2.0**self._dim * self._radii[0:-1]**self._dim * \
                    scipy.special.gamma(1+1./self._p_norm)**self._dim / \
                    scipy.special.gamma(1+float(self._dim)/self._p_norm)
        self._volumes[0:-1] *= 1.0/domain_vol
        self._volumes[-1] = 1.0 - np.sum(self._volumes[0:-1])

class cartesian_sample_set(rectangle_sample_set):
    """
    Defines a hyperrectangle discretization based on a Cartesian grid.

        .. seealso::

            :meth:`bet.sample.rectangle_sample_set`

    """
    def setup(self, xi):
        """
        Initialize.

        #x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid
        """
        if len(xi) != self._dim:
            raise dim_not_matching("dimension of values incorrect")
        xmin = []
        xmax = []
        for xv in xi:
            xmin.append(xv[0:-1])
            xmax.append(xv[1::])
        maxes = np.vstack(np.array(np.meshgrid(*xmax)).T)
        mins = np.vstack(np.array(np.meshgrid(*xmin)).T)
        shp = np.array(maxes.shape)
        pd = np.product(shp[0:-1])
        maxes = maxes.reshape((pd,shp[-1]))
        mins = mins.reshape((pd, shp[-1]))
                          
        rectangle_sample_set.setup(self, maxes, mins)
        
class discretization(object):
    """
    A data structure to store all of the :class:`~bet.sample.sample_set_base`
    objects and associated pointers to solve an stochastic inverse problem. 
    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray`
    vector_names = ['_io_ptr', '_io_ptr_local', '_emulated_ii_ptr',
        '_emulated_ii_ptr_local', '_emulated_oo_ptr', '_emulated_oo_ptr_local']
    #: List of attribute names for attributes that are
    #: :class:`sample.sample_set_base`
    sample_set_names = ['_input_sample_set', '_output_sample_set',
        '_emulated_input_sample_set', '_emulated_output_sample_set',
        '_output_probability_set'] 

 
    def __init__(self, input_sample_set, output_sample_set,
                 output_probability_set=None,
                 emulated_input_sample_set=None,
                 emulated_output_sample_set=None): 
        #: Input sample set :class:`~bet.sample.sample_set_base`
        self._input_sample_set = input_sample_set
        #: Output sample set :class:`~bet.sample.sample_set_base`
        self._output_sample_set = output_sample_set
        #: Emulated Input sample set :class:`~bet.sample.sample_set_base`
        self._emulated_input_sample_set = emulated_input_sample_set
        #: Emulated output sample set :class:`~bet.sample.sample_set_base`
        self._emulated_output_sample_set = emulated_output_sample_set
        #: Output probability set :class:`~bet.sample.sample_set_base`
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
        if out_num != in_num and self._output_sample_set._values is not None \
                and self._input_sample_set._values is not None: 
            raise length_not_matching("input {} and output {} lengths do not\
                    match".format(in_num, out_num))
        else:
            return in_num

    def globalize_ptrs(self):
        """
        Globalizes discretization pointers.

        """
        if (self._io_ptr_local is not None) and  (self._io_ptr is  None):
            self._io_ptr = util.get_global_values(self._io_ptr_local)
        if (self._emulated_ii_ptr_local is not None) and  (self._emulated_ii_ptr is  None):
            self._emulated_ii_ptr = util.get_global_values(self._emulated_ii_ptr_local)
        if (self._emulated_oo_ptr_local is not None) and  (self._emulated_oo_ptr is  None):
            self._emulated_oo_ptr = util.get_global_values(self._emulated_oo_ptr_local)

    def set_io_ptr(self, globalize=True):
        """
        
        Creates the pointer from ``self._output_sample_set`` to
        ``self._output_probability_set``

        :param bool globalize: flag whether or not to globalize
            ``self._output_sample_set``
        
        """
        if self._output_sample_set._values_local is None:
            self._output_sample_set.global_to_local()
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
        (_, self._emulated_oo_ptr_local) = self._output_probability_set.query(\
                self._emulated_output_sample_set._values_local)
                                                                
        if globalize:
            self._emulated_oo_ptr = util.get_global_values\
                    (self._emulated_oo_ptr_local)

    def get_emulated_oo_ptr(self):
        """
        
        Returns the pointer from ``self._emulated_output_sample_set`` to
        ``self._output_probability_set``

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
            if self._output_sample_set is not None:
                output_dims.append(self._output_sample_set.get_dim())
            if self._emulated_output_sample_set is not None:
                output_dims.append(self._emulated_output_sample_set.get_dim())
            if len(output_dims) == 1:
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
            output_dims.append(emulated_output_sample_set.get_dim())
            if self._output_sample_set is not None:
                output_dims.append(self._output_sample_set.get_dim())
            if self._output_probability_set is not None:
                output_dims.append(self._output_probability_set.get_dim())
            if len(output_dims) == 1:
                self._emulated_output_sample_set = emulated_output_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._emulated_output_sample_set = emulated_output_sample_set
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
            if self._input_sample_set is not None:
                if self._input_sample_set.get_dim() == \
                        emulated_input_sample_set.get_dim():
                    self._emulated_input_sample_set = emulated_input_sample_set
                else:
                    raise dim_not_matching("dimension of values incorrect")
            else:
                self._emulated_input_sample_set = emulated_input_sample_set
        else:
            raise AttributeError("Wrong Type: Should be sample_set_base type")

    def estimate_input_volume_emulated(self):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration.

        .. note ::

            This could be re-written to just use ``emulated_ii_ptr`` instead
            of ``_emulated_input_sample_set``.

        """
        if self._emulated_input_sample_set is None:
            raise AttributeError("Required: _emulated_input_sample_set")
        else:
            self._input_sample_set.estimate_volume_emulated(self._emulated_input_sample_set)

    def estimate_output_volume_emulated(self):
        """
        Calculate the volume faction of cells approximately using Monte
        Carlo integration.

        .. note ::

            This could be re-written to just use ``emulated_oo_ptr`` instead
            of ``_emulated_output_sample_set``.


        """
        if self._emulated_output_sample_set is None:
            raise AttributeError("Required: _emulated_output_sample_set")
        else:
            self._output_sample_set.estimate_volume_emulated(self._emulated_output_sample_set)
