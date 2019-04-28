import numpy as np
import logging
import bet.util as util
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import scipy.spatial.distance as ds


def density(sample_set, ptr=None):
    r"""
    Compute density for a sample set and write it to the `_emulated_density`
    attribute.
    """
    if sample_set is None:
        raise AttributeError("Missing sample set.")
    elif hasattr(sample_set, '_density'):
        # this is our way of checking if we used sampling-approach
        # if already computed, avoid re-computation.
        if ptr is not None:
            den = sample_set._density[ptr]
        else:
            den = sample_set._density
        sample_set._emulated_density = den
    else:  # not none
        if sample_set._probabilities is None:
            raise AttributeError("Missing probabilities from sample set.")
        if sample_set._volumes is None:
            raise AttributeError("Missing volumes from sample set.")
        if sample_set._probabilities_local is None:
            sample_set.global_to_local()

        if ptr is None:
            den = np.divide(sample_set._probabilities.ravel(),
                            sample_set._volumes.ravel())
        else:
            den = np.divide(sample_set._probabilities[ptr].ravel(),
                            sample_set._volumes[ptr].ravel())
        sample_set._emulated_density = den
    if ptr is None:  # create pointer to density to avoid re-run
        sample_set._density = sample_set._emulated_density
    else:
        sample_set._prob = sample_set._probabilities[ptr].ravel()
    sample_set.local_to_global()
    return sample_set


def metric(left_set, right_set, num_mc_points=1000):
    r"""
    Creates and returns a `~bet.postProcess.metrization` object

    :param int cnum: number of values of sample set to return

    :rtype: :class:`~bet.postProcess.metrization`
    :returns: metrization object

    :class:`sample.sample_set_base`
    """
    # extract sample set
    if isinstance(left_set, samp.discretization):
        left_set = left_set.get_input_sample_set()
    if isinstance(right_set, samp.discretization):
        right_set = right_set.get_input_sample_set()
    if not num_mc_points > 0:
        raise ValueError("Please specify positive num_mc_points")

    # make integration sample set
    assert left_set.get_dim() == right_set.get_dim()
    assert np.array_equal(left_set.get_domain(), right_set.get_domain())
    int_set = samp.sample_set(left_set.get_dim())
    int_set.set_domain(right_set.get_domain())
    int_set = bsam.random_sample_set('r', int_set, num_mc_points)

    # to be generating a new random sample set pass an integer argument
    metrc = metrization(int_set, left_set, right_set)

    return metrc


class metrization(object):
    """

    A data structure containing :class:`~bet.sample.sample_set_base` objects and
    associated methods for computing measures of distance between pairs of them.
    Distances have two slots, hence the language for left/right.
    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray`
    vector_names = ['_ptr_left', '_ptr_left_local',
                    '_ptr_right', '_ptr_right_local', '_domain']

    #: List of attribute names for attributes that are
    #: :class:`sample.sample_set_base`
    sample_set_names = ['_sample_set_left', '_sample_set_right',
                        '_emulated_sample_set']

    def __init__(self, emulated_sample_set,
                 sample_set_left=None, sample_set_right=None,
                 ptr_left=None, ptr_right=None):
        #: Left sample set
        self._sample_set_left = None
        #: Right sample set
        self._sample_set_right = None
        #: Integration/Emulation set :class:`~bet.sample.sample_set_base`
        self._emulated_sample_set = emulated_sample_set
        #: Pointer from ``self._emulated_sample_set`` to
        #: ``self._sample_set_left``
        self._ptr_left = ptr_left
        #: Pointer from ``self._emulated_sample_set`` to
        #: ``self._sample_set_right``
        self._ptr_right = ptr_right
        #: local integration left ptr for parallelsim
        self._ptr_left_local = None
        #: local integration right ptr for parallelism
        self._ptr_right_local = None
        #: Domain
        self._domain = None
        #: Left sample set density evaluated on emulation set.
        self._den_left = None
        #: Right sample set density evaluated on emulation set.
        self._den_right = None

        # extract sample set
        if isinstance(sample_set_left, samp.sample_set_base):
            # left sample set
            self._sample_set_left = sample_set_left
            self._domain = sample_set_left.get_domain()
        if isinstance(sample_set_right, samp.sample_set_base):
            # right sample set
            self._sample_set_right = sample_set_right
            if self._domain is not None:
                if not np.allclose(self._domain, sample_set_right._domain):
                    raise samp.domain_not_matching(
                        "Left and Right domains do not match")
            else:
                self._domain = sample_set_right.get_domain()

        # check dimension consistency
        if isinstance(emulated_sample_set, samp.sample_set_base):
            self._num_samples = emulated_sample_set.check_num()
            output_dims = []
            output_dims.append(emulated_sample_set.get_dim())
            if self._sample_set_right is not None:
                output_dims.append(self._sample_set_right.get_dim())
            if self._sample_set_left is not None:
                output_dims.append(self._sample_set_left.get_dim())
            if len(output_dims) == 1:
                self._emulated_sample_set = emulated_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._emulated_sample_set = emulated_sample_set
            else:
                raise samp.dim_not_matching("dimension of values incorrect")

            if not isinstance(emulated_sample_set.get_domain(), np.ndarray):
                # domain can be missing if left/right sample sets present
                if self._sample_set_left is not None:
                    emulated_sample_set.set_domain(self._domain)
                else:
                    if self._sample_set_right is not None:
                        emulated_sample_set.set_domain(self._domain)
                    else:  # no sample sets provided
                        msg = "Must provide at least one set from\n"
                        msg += "\twhich a domain can be inferred."
                        raise AttributeError(msg)
        else:
            if (self._sample_set_left is not None) or \
               (self._sample_set_right is not None):
                pass
            else:
                raise AttributeError(
                    "Wrong Type: Should be samp.sample_set_base type")

        if (ptr_left is not None):
            if len(ptr_left) != self._num_samples:
                raise AttributeError(
                    "Left pointer length must match integration set.")
            if (ptr_right is not None):
                if not np.allclose(ptr_left.shape, ptr_right.shape):
                    raise AttributeError("Pointers must be of same length.")
        if (ptr_right is not None):
            if len(ptr_right) != self._num_samples:
                raise AttributeError(
                    "Right pointer length must match integration set.")

    # set density functions, maybe print a
    # message if MC assumption is used to estimate volumes

    # evaluate density functions at integration points, store for re-use

    # metric - wrapper around scipy now that
    # passes density values with proper shapes.

    def check_num(self):
        r"""
        Checks that the sizes of all pointers are consistent
        """
        return self._emulated_sample_set.check_num()

    def check_dim(self):
        r"""
        Checks that dimensions of left and right sample sets match.
        """
        left_set = self.get_left()
        right_set = self.get_right()
        if left_set.get_dim() != right_set.get_dim():
            msg = "These sample sets must have the same dimension."
            raise samp.dim_not_matching(msg)
        else:
            dim = left_set.get_dim()

        il, ir = self.get_ptr_left(), self.get_ptr_right()
        if (il is not None) and (ir is not None):
            if len(il) != len(ir):
                msg = "The pointers have inconsistent sizes."
                msg += "\nTry running set_ptr_left() [or _right()]"
                raise samp.dim_not_matching(msg)
        return dim

    def check_domain(self):
        r"""
        Checks that all domains match.
        """
        left_set = self.get_left()
        right_set = self.get_right()
        if left_set._domain is not None and right_set._domain is not None:
            if not np.allclose(left_set._domain, right_set._domain):
                msg = "These sample sets have different domains."
                raise samp.domain_not_matching(msg)
            else:
                domain = left_set.get_domain()
        else:
            if left_set._domain is None or right_set._domain is None:
                msg = "One or more of your sets is missing a domain."
                raise samp.domain_not_matching(msg)
            else:  # since the domains match, we can choose either.
                domain = left_set.get_domain()
        if not np.allclose(self._emulated_sample_set.get_domain(), domain):
            msg = "Integration domain mismatch."
            raise samp.domain_not_matching(msg)
        self._domain = domain
        return domain

    def globalize_ptrs(self):
        r"""
        Globalizes metrization pointers.

        """
        if (self._ptr_left_local is not None) and\
                (self._ptr_left is None):
            self._ptr_left = util.get_global_values(
                self._ptr_left_local)
        if (self._ptr_right_local is not None) and\
                (self._ptr_right is None):
            self._ptr_right = util.get_global_values(
                self._ptr_right_local)

    def set_ptr_left(self, globalize=True):
        """

        Creates the pointer from ``self._emulated_sample_set`` to
        ``self._sample_set_left``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._ptr_left``
        """
        if self._emulated_sample_set._values_local is None:
            self._emulated_sample_set.global_to_local()

        (_, self._ptr_left_local) = self._sample_set_left.query(
            self._emulated_sample_set._values_local)

        if globalize:
            self._ptr_left = util.get_global_values(
                self._ptr_left_local)
        assert self._sample_set_left.check_num() >= max(self._ptr_left_local)

    def get_ptr_left(self):
        """

        Returns the pointer from ``self._emulated_sample_set`` to
        ``self._sample_set_left``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._sample_set_left._values.shape[0],)
        :returns: self._ptr_left

        """
        return self._ptr_left

    def set_ptr_right(self, globalize=True):
        """

        Creates the pointer from ``self._emulated_sample_set`` to
        ``self._sample_set_right``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._ptr_right``

        """
        if self._emulated_sample_set._values_local is None:
            self._emulated_sample_set.global_to_local()

        (_, self._ptr_right_local) = self._sample_set_right.query(
            self._emulated_sample_set._values_local)

        if globalize:
            self._ptr_right = util.get_global_values(
                self._ptr_right_local)
        assert self._sample_set_right.check_num() >= max(self._ptr_right_local)

    def get_ptr_right(self):
        """

        Returns the pointer from ``self._emulated_sample_set`` to
        ``self._sample_set_right``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._sample_set_right._values.shape[0],)
        :returns: self._ptr_right

        """
        return self._ptr_right

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.sample.metrization`
        :returns: Copy of this :class:`~bet.sample.metrization`

        """
        my_copy = metrization(self._emulated_sample_set.copy(),
                              self._sample_set_left.copy(),
                              self._sample_set_right.copy())

        for attrname in metrization.sample_set_names:
            if attrname is not '_sample_set_left' and \
                    attrname is not '_sample_set_right':
                curr_sample_set = getattr(self, attrname)
                if curr_sample_set is not None:
                    setattr(my_copy, attrname, curr_sample_set.copy())

        for array_name in metrization.vector_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(my_copy, array_name, np.copy(current_array))
        return my_copy

    def get_sample_set_left(self):
        """

        Returns a reference to the left sample set for this metrization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: left sample set

        """
        return self._sample_set_left

    def get_left(self):
        r"""

        Wrapper for `get_sample_set_left`.

        """
        return self.get_sample_set_left()

    def set_sample_set_left(self, sample_set_left):
        """

        Sets the left sample set for this metrization.

        :param sample_set_left: left sample set
        :type sample_set_left: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(sample_set_left, samp.sample_set_base):
            self._sample_set_left = sample_set_left
            self._ptr_left = None
            self._ptr_left_local = None
            self._den_left = None
        else:
            raise TypeError(
                "Wrong Type: Should be samp.sample_set_base type")
        if self._emulated_sample_set._domain is None:
            self._emulated_sample_set.set_domain(
                sample_set_left.get_domain())
        else:
            if not np.allclose(self._emulated_sample_set._domain,
                               sample_set_left._domain):
                raise samp.domain_not_matching(
                    "Domain does not match integration set.")

    def set_left(self, sample_set):
        r"""

        Wrapper for `set_sample_set_left`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_sample_set_left(sample_set)

    def get_sample_set_right(self):
        """

        Returns a reference to the right sample set for this metrization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: right sample set

        """
        return self._sample_set_right

    def get_right(self):
        r"""

        Wrapper for `get_sample_set_right`.

        """
        return self.get_sample_set_right()

    def set_right(self, sample_set):
        r"""

        Wrapper for `set_sample_set_right`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_sample_set_right(sample_set)

    def set_sample_set_right(self, sample_set_right):
        """

        Sets the right sample set for this metrization.

        :param sample_set_right: right sample set
        :type sample_set_right: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(sample_set_right, samp.sample_set_base):
            self._sample_set_right = sample_set_right
            self._ptr_right = None
            self._ptr_right_local = None
            self._den_right = None
        else:
            raise TypeError(
                "Wrong Type: Should be samp.sample_set_base type")

        if self._emulated_sample_set._domain is None:
            self._emulated_sample_set.set_domain(
                sample_set_right.get_domain())
        else:
            if not np.allclose(self._emulated_sample_set._domain,
                               sample_set_right._domain):
                raise samp.domain_not_matching(
                    "Domain does not match integration set.")

    def get_emulated_sample_set(self):
        r"""

        Returns a reference to the output probability sample set for this
        metrization.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: output probability sample set

        """
        return self._emulated_sample_set

    def set_emulated_sample_set(self, emulated_sample_set):
        r"""

        Sets the integration_output sample set for this metrization.

        :param emulated_sample_set: integration sample set.
        :type emulated_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(emulated_sample_set, samp.sample_set_base):
            output_dims = []
            output_dims.append(emulated_sample_set.get_dim())
            if self._sample_set_right is not None:
                output_dims.append(self._sample_set_right.get_dim())
            if self._sample_set_left is not None:
                output_dims.append(self._sample_set_left.get_dim())
            if len(output_dims) == 1:
                self._emulated_sample_set = emulated_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._emulated_sample_set = emulated_sample_set
            else:
                raise samp.dim_not_matching("dimension of values incorrect")
        else:
            raise AttributeError(
                "Wrong Type: Should be samp.sample_set_base type")
        # if a new emulation set is provided, forget the emulated evaluation.
        if self._sample_set_left is not None:
            self._sample_set_left._emulated_density = None
        if self._sample_set_right is not None:
            self._sample_set_right._emulated_density = None

    def get_em(self):
        r"""

        Wrapper for `get_emulated_sample_set`.

        """
        return self.get_emulated_sample_set()

    def get_int(self):
        r"""

        Wrapper for `get_emulated_sample_set`.

        """
        return self.get_emulated_sample_set()

    def get_integration_sample_set(self):
        r"""

        Wrapper for `get_emulated_sample_set`.

        """
        return self.get_emulated_sample_set()

    def get_emulated(self):
        r"""

        Wrapper for `get_emulated_sample_set`.

        """
        return self.get_emulated_sample_set()

    def set_emulated(self, sample_set):
        r"""

        Wrapper for `set_emulated_sample_set`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_emulated_sample_set(sample_set)

    def set_em(self, sample_set):
        r"""

        Wrapper for `set_emulated_sample_set`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_emulated_sample_set(sample_set)

    def set_int(self, sample_set):
        r"""

        Wrapper for `set_emulated_sample_set`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_emulated_sample_set(sample_set)

    def set_integration_sample_set(self, sample_set):
        r"""

        Wrapper for `set_emulated_sample_set`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_emulated_sample_set(sample_set)

    def clip(self, lnum, rnum=None):
        r"""
        Creates and returns a metrization with the the first `lnum`
        and `rnum` entries of the left and right sample sets, resp.

        :param int cnum: number of values of sample set to return

        :rtype: :class:`~bet.sample.metrization`
        :returns: clipped metrization

        """
        if rnum is None:  # can clip by same amount
            rnum = lnum
        if lnum > 0:
            cl = self._sample_set_left.clip(lnum)
        else:
            cl = self._sample_set_left.copy()
        if rnum > 0:
            cr = self._sample_set_right.clip(rnum)
        else:
            cr = self._sample_set_right.copy()

        return metrization(sample_set_left=cl,
                           sample_set_right=cr,
                           emulated_sample_set=self.
                           _emulated_sample_set.copy())

    def merge(self, metr):
        r"""
        Merges a given metrization with this one by merging the input and
        output sample sets.

        :param metr: metrization object to merge with.
        :type metr: :class:`bet.sample.metrization`

        :rtype: :class:`bet.sample.metrization`
        :returns: Merged metrization
        """
        ml = self._sample_set_left.merge(metr._sample_set_left)
        mr = self._sample_set_right.merge(metr._sample_set_right)
        il, ir = self._ptr_left, self._ptr_right
        if metr._ptr_left is not None:
            il += metr._ptr_left
        if metr._ptr_right is not None:
            ir += metr._ptr_right
        return metrization(sample_set_left=ml,
                           sample_set_right=mr,
                           emulated_sample_set=self._emulated_sample_set,
                           ptr_left=il,
                           ptr_right=ir)

    def slice(self,
              dims=None):
        r"""
        Slices the left and right of the metrization.

        :param list dims: list of indices (dimensions) of sample set to include
        :param list right: list of indices of right sample set to include

        :rtype: :class:`~bet.sample.metrization`
        :returns: sliced metrization

        """
        slice_list = ['_values', '_values_local',
                      '_error_estimates', '_error_estimates_local',
                      ]
        slice_list2 = ['_jacobians', '_jacobians_local']

        int_ss = samp.sample_set(len(dims))
        left_ss = samp.sample_set(len(dims))
        right_ss = samp.sample_set(len(dims))

        if self._emulated_sample_set._domain is not None:
            int_ss.set_domain(self._emulated_sample_set._domain[dims, :])

        if self._sample_set_left._domain is not None:
            left_ss.set_domain(self._sample_set_left._domain[dims, :])
        if self._sample_set_left._reference_value is not None:
            left_ss.set_reference_value(
                self._sample_set_left._reference_value[dims])

        if self._sample_set_right._domain is not None:
            right_ss.set_domain(self._sample_set_right._domain[dims, :])
        if self._sample_set_right._reference_value is not None:
            right_ss.set_reference_value(
                self._sample_set_right._reference_value[dims])

        for obj in slice_list:
            val = getattr(self._sample_set_left, obj)
            if val is not None:
                setattr(left_ss, obj, val[:, dims])
            val = getattr(self._sample_set_right, obj)
            if val is not None:
                setattr(right_ss, obj, val[:, dims])
            val = getattr(self._emulated_sample_set, obj)
            if val is not None:
                setattr(int_ss, obj, val[:, dims])
        for obj in slice_list2:
            val = getattr(self._sample_set_left, obj)
            if val is not None:
                nval = np.copy(val)
                nval = nval.take(dims, axis=1)
                nval = nval.take(dims, axis=2)
                setattr(left_ss, obj, nval)
            val = getattr(self._sample_set_right, obj)
            if val is not None:
                nval = np.copy(val)
                nval = nval.take(dims, axis=1)
                nval = nval.take(dims, axis=2)
                setattr(right_ss, obj, nval)

        metr = metrization(sample_set_left=left_ss,
                           sample_set_right=right_ss,
                           emulated_sample_set=int_ss)
        # additional attributes to copy over here. TODO: maybe slice through
        return metr

    def global_to_local(self):
        """
        Call global_to_local for ``sample_set_left`` and
        ``sample_set_right``.
        """
        if self._sample_set_left is not None:
            self._sample_set_left.global_to_local()
        if self._sample_set_right is not None:
            self._sample_set_right.global_to_local()
        if self._emulated_sample_set is not None:
            self._emulated_sample_set.global_to_local()

    def local_to_global(self):
        """
        Call local_to_global for ``sample_set_left``,
        ``sample_set_right``, and ``emulation_sample_set``.
        """
        if self._sample_set_left is not None:
            self._sample_set_left.local_to_global()
        if self._sample_set_right is not None:
            self._sample_set_right.local_to_global()
        if self._emulated_sample_set is not None:
            self._emulated_sample_set.local_to_global()

    def estimate_volume_mc(self):
        r"""
        Applies MC assumption to volumes of both sets.
        """
        self._sample_set_left.estimate_volume_mc()
        self._sample_set_right.estimate_volume_mc()

    def set_left_probabilities(self, probabilities):
        r"""
        Allow overwriting of probabilities for the left set.
        """
        if self.get_left().check_num() != len(probabilities):
            raise AttributeError("Length of probabilities incorrect.")
        self._sample_set_left._probabilities = probabilities
        self._sample_set_left.global_to_local()
        self._sample_set_left._emulated_density = None
        self._den_left = None

    def set_right_probabilities(self, probabilities):
        r"""
        Allow overwriting of probabilities for the right set.
        """
        if self.get_right().check_num() != len(probabilities):
            raise AttributeError("Length of probabilities incorrect.")
        self._sample_set_right._probabilities = probabilities
        self._sample_set_right.global_to_local()
        self._sample_set_right._emulated_density = None
        self._den_right = None

    def get_left_probabilities(self):
        r"""
        Return probabilities for the left set.
        """
        return self._sample_set_left._probabilities

    def get_right_probabilities(self):
        r"""
        Return probabilities for the right set.
        """
        return self._sample_set_right._probabilities

    def set_volume_emulated(self, sample_set, emulated_sample_set=None):
        r"""
        Wrapper to use the emulated sample set for the
        calculation of volumes on the sample sets.
        """
        if emulated_sample_set is not None:
            if not isinstance(emulated_sample_set, samp.sample_set_base):
                msg = "Wrong type specified for `emulation_set`.\n"
                msg += "Please specify a `~bet.sample.sample_set_base`."
                raise AttributeError(msg)
            else:
                sample_set.estimate_volume_emulated(emulated_sample_set)
        else:
            # if not defined, use existing emulated set for volumes.
            sample_set.estimate_volume_emulated(self._emulated_sample_set)

    def set_left_volume_emulated(self, emulated_sample_set=None):
        r"""
        Use an emulated sample set to define volumes for the left set.
        """
        self.set_volume_emulated(self.get_left(), emulated_sample_set)
        self._den_left = None  # if volumes change, so will densities.

    def set_right_volume_emulated(self, emulated_sample_set=None):
        r"""
        Use an emulated sample set to define volumes for the right set.
        """
        self.set_volume_emulated(self.get_right(), emulated_sample_set)
        self._den_right = None  # if volumes change, so will densities.

    def estimate_density_left(self):
        r"""
        Evaluates density function for the left probability measure
        at the set of samples defined in `emulated_sample_set`.
        """
        s_set = self.get_left()
        if self._ptr_left_local is None:
            self.set_ptr_left()
        s_set = density(s_set, self._ptr_left_local)
        self._den_left = s_set._emulated_density
        return self._den_left

    def estimate_density_right(self):
        r"""
        Evaluates density function for the right probability measure
        at the set of samples defined in ``emulated_sample_set``.
        """
        s_set = self.get_right()
        if self._ptr_right_local is None:
            self.set_ptr_right()
        s_set = density(s_set, self._ptr_right_local)
        self._den_right = s_set._emulated_density
        return self._den_right

    def estimate_right_density(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.estimate_density_right``.
        """
        return self.estimate_density_right()

    def estimate_left_density(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.estimate_density_left``.
        """
        return self.estimate_density_left()

    def get_density_right(self):
        r"""
        Returns right emulated density.
        """
        return self._den_right

    def get_density_left(self):
        r"""
        Returns left emulated density.
        """
        return self._den_left

    def get_left_density(self):
        r"""
        Wrapper. Returns left emulated density.
        """
        return self.get_density_left()

    def get_right_density(self):
        r"""
        Wrapper. Returns right emulated density.
        """
        return self.get_density_right()

    def estimate_density(self, globalize=True,
                         emulated_sample_set=None):
        r"""
        Evaluate density functions for both left and right sets using
        the set of samples defined in ``self._emulated_sample_set``.
        """
        if globalize:  # in case probabilities were re-set but not local
            self.global_to_local()
        self.check_domain()

        # set pointers if they have not already been set
        if self._ptr_left_local is None:
            self.set_ptr_left(globalize)
        if self._ptr_right_local is None:
            self.set_ptr_right(globalize)
        self.check_dim()

        int_set = self.get_int()
        left_set, right_set = self.get_left(), self.get_right()

        if left_set._volumes is None:
            if emulated_sample_set is None:
                msg = " Volumes missing from left. Using MC assumption."
                logging.log(20, msg)
                left_set.estimate_volume_mc()
            else:
                self.set_left_volume_emulated(emulated_sample_set)
        else:  # volumes present and emulated passed
            if emulated_sample_set is not None:
                msg = " Overwriting left volumes with emulated ones."
                logging.log(20, msg)
                self.set_left_volume_emulated(emulated_sample_set)

        if right_set._volumes is None:
            if emulated_sample_set is None:
                msg = " Volumes missing from right. Using MC assumption."
                logging.log(20, msg)
                right_set.estimate_volume_mc()
            else:
                msg = " Overwriting right volumes with emulated ones."
                logging.log(20, msg)
                self.set_right_volume_emulated(emulated_sample_set)
        else:  # volumes present and emulated passed
            if emulated_sample_set is not None:
                self.set_right_volume_emulated(emulated_sample_set)

        if int_set is None:
            raise AttributeError("Missing integration set.")

        # compute densities
        self.estimate_density_left()
        self.estimate_density_right()

        if len(right_set._emulated_density) != len(left_set._emulated_density):
            msg = "Length of pointers "
            raise samp.dim_not_matching(msg)

        if globalize:
            self.local_to_global()
        return self._den_left, self._den_right

    def distance(self, metric='tv', **kwargs):
        r"""
        Compute distance using evaluated densities on a shared emulated set.
        If either density evaluation is missing, re-compute it.
        """
        left_den, right_den = self.get_left_density(), self.get_right_density()
        if left_den is None:
            # logging.log(20,"Left density missing. Estimating now.")
            left_den = self.estimate_density_left()
        if right_den is None:
            # logging.log(20,"Right density missing. Estimating now.")
            right_den = self.estimate_density_right()

        if metric in ['tv', 'totvar',
                      'total variation', 'total-variation', '1']:
            dist = ds.minkowski(left_den, right_den, 1, w=0.5, **kwargs)
        elif metric in ['mink', 'minkowski']:
            dist = ds.minkowski(left_den, right_den, **kwargs)
        elif metric in ['norm']:
            dist = ds.norm(left_den-right_den, **kwargs)
        elif metric in ['euclidean', '2-norm', '2']:
            dist = ds.minkowski(left_den, right_den, 2, **kwargs)
        elif metric in ['sqhell', 'sqhellinger']:
            dist = ds.sqeuclidean(np.sqrt(left_den), np.sqrt(right_den))/2.0
        elif metric in ['hell', 'hellinger']:
            return np.sqrt(self.distance('sqhell'))
        else:
            dist = metric(left_den, right_den, **kwargs)

        return dist/self.check_num()
