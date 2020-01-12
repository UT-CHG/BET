import numpy as np
import logging
import bet.util as util
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import scipy.spatial.distance as ds


def density_estimate(sample_set, ptr=None):
    r"""
    Evaluate an approximate density on a comparison sample set into
    which the pointer variable ``ptr`` points. This function returns
    the density estimates for a sample set object and write it to the
    ``_comparison_densities`` attribute inside of ``sample_set``

    :param sample_set: sample set with existing probabilities stored
    :type sample_set: :class:`bet.sample.sample_set_base`
    :param ptr: pointer to a reference set against which densities are
        being compared. If ``None``, use samples as they are.
    :type ptr: list, tuple, or ``np.ndarray``

    :rtype: :class:`bet.sample.sample_set_base`
    :returns: sample set object with attribute ``_comparison_densities``

    """
    if sample_set is None:
        raise AttributeError("Required: sample_set object")
    elif sample_set._densities is not None:
        # this is our way of checking if we used sampling-approach
        # if already computed, avoid re-computation.
        if ptr is not None:
            den = sample_set._densities[ptr]
        else:
            den = sample_set._densities
        sample_set._comparison_densities = den
    else:  # missing densities, use probabilities
        if sample_set._probabilities is None:
            if sample_set._probabilities_local is not None:
                sample_set.local_to_global()
            else:
                msg = "Required: _probabilities in sample_set"
                msg += "to construct density estimates."
                raise AttributeError(msg)
        if sample_set._volumes is None:
            msg = "Required: _volumes in sample_set"
            msg += "to construct density estimates."
            raise AttributeError(msg)
        if sample_set._probabilities_local is None:
            sample_set.global_to_local()

        if ptr is None:
            den = np.divide(sample_set._probabilities.ravel(),
                            sample_set._volumes.ravel())
        else:
            den = np.divide(sample_set._probabilities[ptr].ravel(),
                            sample_set._volumes[ptr].ravel())
        sample_set._comparison_densities = den
    if ptr is None:  # create pointer to density estimates to avoid re-run
        sample_set._densities = sample_set._comparison_densities
    else:
        sample_set._prob = sample_set._probabilities[ptr].ravel()
    sample_set.local_to_global()
    return sample_set


class comparison(object):
    """
    This class allows for analytically-sound comparisons between
    probability measures defined on different sigma-algebras. In order
    to compare the similarity of two measures defined on different
    sigma-algebras (induced by the voronoi-cell tesselations implicitly
    defined by the ``_values`` in each sample set), a third sample set
    object contains the set of samples on which the measures will
    be compared. It is referred to as an ``comparison_sample_set``
    and is the only set that is actually required to instantiate a
    ``comparison`` object; the dimension and domain of it will be
    used to enforce proper setting of the left and right sample sets.

    This object can be thought of as a more flexible version of an abstraction
    of a metric, a measure of distance between two probability measures.
    A metric ``d(x,y)`` has two arguments, one to the left (``x``),
    and one to the right (``y``). However, we do not enforce the properties
    that define a formal metric, instead we use the language of "comparisons".

    Technically, any function can be passed for evaluation, including
    ones that fail to satisfy symmetry, so we refrain from referring
    to measures of similarity as metrics, though this is the usual case
    (with the exception of the frequently used KL-Divergence).
    Several common measures of similarity are accessible with keywords.

    The number of samples in this third (reference) sample set is
    given by the argument ``num_mc_points``, and pointers between this
    set and the left/right sets are built on-demand. Methods in this
    class allow for over-writing of any of the three sample set objects
    involved, and pointers are either re-built explictly, or they
    are constructed when a measure of similarity (such as distance)
    is requested to be evaluated.

    .. seealso::

        :meth:`bet.compareP.comparison.value``

    :param comparison_sample_set: Reference set against which comparisons
        will be made.
    :type comparison_sample_set: :class:`bet.sample.sample_set_base`

    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray`
    vector_names = ['_ptr_left', '_ptr_left_local',
                    '_ptr_right', '_ptr_right_local', '_domain']

    #: List of attribute names for attributes that are
    #: :class:`sample.sample_set_base`
    sample_set_names = ['_left_sample_set', '_right_sample_set',
                        '_comparison_sample_set']

    def __init__(self, comparison_sample_set,
                 sample_set_left=None, sample_set_right=None,
                 ptr_left=None, ptr_right=None):
        #: Left sample set
        self._left_sample_set = None
        #: Right sample set
        self._right_sample_set = None
        #: Integration/Emulation set :class:`~bet.sample.sample_set_base`
        self._comparison_sample_set = comparison_sample_set
        #: Pointer from ``self._comparison_sample_set`` to
        #: ``self._left_sample_set``
        self._ptr_left = None
        #: Pointer from ``self._comparison_sample_set`` to
        #: ``self._right_sample_set``
        self._ptr_right = None
        #: local integration left ptr for parallelsim
        self._ptr_left_local = ptr_left
        #: local integration right ptr for parallelism
        self._ptr_right_local = ptr_right
        #: Domain
        self._domain = None
        #: Left sample set density evaluated on emulation set.
        self._den_left = None
        #: Right sample set density evaluated on emulation set.
        self._den_right = None

        # extract sample set
        if isinstance(sample_set_left, samp.sample_set_base):
            # left sample set
            self._left_sample_set = sample_set_left
            self._domain = sample_set_left.get_domain()
        if isinstance(sample_set_right, samp.sample_set_base):
            # right sample set
            self._right_sample_set = sample_set_right
            if self._domain is not None:
                if not np.allclose(self._domain, sample_set_right._domain):
                    raise samp.domain_not_matching(
                        "Left and Right domains do not match")
            else:
                self._domain = sample_set_right.get_domain()

        # check dimension consistency
        if isinstance(comparison_sample_set, samp.sample_set_base):
            self._num_samples = comparison_sample_set.check_num()
            output_dims = []
            output_dims.append(comparison_sample_set.get_dim())
            if self._right_sample_set is not None:
                output_dims.append(self._right_sample_set.get_dim())
            if self._left_sample_set is not None:
                output_dims.append(self._left_sample_set.get_dim())
            if len(output_dims) == 1:
                self._comparison_sample_set = comparison_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._comparison_sample_set = comparison_sample_set
            else:
                raise samp.dim_not_matching("Dimension of values incorrect")

            if not isinstance(comparison_sample_set.get_domain(), np.ndarray):
                # domain can be missing if left/right sample sets present
                if self._left_sample_set is not None:
                    comparison_sample_set.set_domain(self._domain)
                else:
                    if self._right_sample_set is not None:
                        comparison_sample_set.set_domain(self._domain)
                    else:  # no sample sets provided
                        msg = "Must provide at least one set from\n"
                        msg += "\twhich a domain can be inferred."
                        raise AttributeError(msg)
        else:
            if (self._left_sample_set is not None) or \
               (self._right_sample_set is not None):
                pass
            else:
                raise AttributeError(
                    "Wrong Type: Should be samp.sample_set_base type")

        if (ptr_left is not None):
            if len(ptr_left) != self._num_samples:
                raise AttributeError(
                    "Left pointer length must match comparison set.")
            else:
                self._ptr_left_local = ptr_left
            if (ptr_right is not None):
                if not np.allclose(ptr_left.shape, ptr_right.shape):
                    raise AttributeError("Pointers must be of same length.")
        if (ptr_right is not None):
            if len(ptr_right) != self._num_samples:
                raise AttributeError(
                    "Right pointer length must match comparison set.")
            else:
                self._ptr_right_local = ptr_right

    def check_dim(self):
        r"""
        Checks that dimensions of left and right sample sets match
        the dimension of the comparison sample set.

        :rtype: int
        :returns: dimension

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
        Checks that all domains match so that the comparisons
        are being made on measures defined on the same underlying space.

        :rtype: ``np.ndarray`` of shape (ndim, 2)
        :returns: domain bounds

        """
        left_set = self.get_left()
        right_set = self.get_right()
        if left_set._domain is not None and right_set._domain is not None:
            if not np.allclose(left_set._domain, right_set._domain):
                msg = "These sample sets have different domains."
                raise samp.domain_not_matching(msg)
            else:
                domain = left_set.get_domain()
        else:  # since the domains match, we can choose either.
            if left_set._domain is None or right_set._domain is None:
                msg = "One or more of your sets is missing a domain."
                raise samp.domain_not_matching(msg)

        if not np.allclose(self._comparison_sample_set.get_domain(), domain):
            msg = "Integration domain mismatch."
            raise samp.domain_not_matching(msg)
        self._domain = domain
        return domain

    def globalize_ptrs(self):
        r"""
        Globalizes comparison pointers by caling ``get_global_values``
        for both the left and right sample sets.

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
        Creates the pointer from ``self._comparison_sample_set`` to
        ``self._left_sample_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._ptr_left``

        """
        if self._comparison_sample_set._values_local is None:
            self._comparison_sample_set.global_to_local()

        (_, self._ptr_left_local) = self._left_sample_set.query(
            self._comparison_sample_set._values_local)

        if globalize:
            self._ptr_left = util.get_global_values(
                self._ptr_left_local)
        assert self._left_sample_set.check_num() >= max(self._ptr_left_local)

    def get_ptr_left(self):
        """
        Returns the pointer from ``self._comparison_sample_set`` to
        ``self._left_sample_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._left_sample_set._values.shape[0],)
        :returns: self._ptr_left

        """
        return self._ptr_left

    def set_ptr_right(self, globalize=True):
        """
        Creates the pointer from ``self._comparison_sample_set`` to
        ``self._right_sample_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._ptr_right``

        """
        if self._comparison_sample_set._values_local is None:
            self._comparison_sample_set.global_to_local()

        (_, self._ptr_right_local) = self._right_sample_set.query(
            self._comparison_sample_set._values_local)

        if globalize:
            self._ptr_right = util.get_global_values(
                self._ptr_right_local)
        assert self._right_sample_set.check_num() >= max(self._ptr_right_local)

    def get_ptr_right(self):
        """
        Returns the pointer from ``self._comparison_sample_set`` to
        ``self._right_sample_set``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._right_sample_set._values.shape[0],)
        :returns: self._ptr_right

        """
        return self._ptr_right

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.postProcess.compareP.comparison`
        :returns: Copy of a comparison object.

        """
        my_copy = comparison(self._comparison_sample_set.copy(),
                             self._left_sample_set.copy(),
                             self._right_sample_set.copy())

        for attrname in comparison.sample_set_names:
            if attrname is not '_left_sample_set' and \
                    attrname is not '_right_sample_set':
                curr_sample_set = getattr(self, attrname)
                if curr_sample_set is not None:
                    setattr(my_copy, attrname, curr_sample_set.copy())

        for array_name in comparison.vector_names:
            current_array = getattr(self, array_name)
            if current_array is not None:
                setattr(my_copy, array_name, np.copy(current_array))
        return my_copy

    def get_left_sample_set(self):
        """
        Returns a reference to the left sample set for this comparison.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: left sample set

        """
        return self._left_sample_set

    def get_left(self):
        r"""
        Wrapper for `get_left_sample_set`.
        """
        return self.get_left_sample_set()

    def set_left_sample_set(self, sample_set):
        """

        Sets the left sample set for this comparison.

        :param sample_set: left sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(sample_set, samp.sample_set_base):
            self._left_sample_set = sample_set
            self._ptr_left = None
            self._ptr_left_local = None
            self._den_left = None
        elif isinstance(sample_set, samp.discretization):
            msg = "Discretization passed. Assuming input set."
            logging.warning(msg)
            sample_set = sample_set.get_input_sample_set()
            self._left_sample_set = sample_set
            self._ptr_left = None
            self._ptr_left_local = None
            self._den_left = None
        else:
            raise TypeError(
                "Wrong Type: Should be samp.sample_set_base type")
        if self._comparison_sample_set._domain is None:
            self._comparison_sample_set.set_domain(
                sample_set.get_domain())
        else:
            if not np.allclose(self._comparison_sample_set._domain,
                               sample_set._domain):
                raise samp.domain_not_matching(
                    "Domain does not match comparison set.")

    def set_left(self, sample_set):
        r"""

        Wrapper for `set_left_sample_set`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_left_sample_set(sample_set)

    def get_right_sample_set(self):
        """

        Returns a reference to the right sample set for this comparison.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: right sample set

        """
        return self._right_sample_set

    def get_right(self):
        r"""
        Wrapper for `get_right_sample_set`.
        """
        return self.get_right_sample_set()

    def set_right(self, sample_set):
        r"""

        Wrapper for `set_right_sample_set`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_right_sample_set(sample_set)

    def set_right_sample_set(self, sample_set):
        """
        Sets the right sample set for this comparison.

        :param sample_set: right sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(sample_set, samp.sample_set_base):
            self._right_sample_set = sample_set
            self._ptr_right = None
            self._ptr_right_local = None
            self._den_right = None
        elif isinstance(sample_set, samp.discretization):
            msg = "Discretization passed. Assuming input set."
            logging.warning(msg)
            sample_set = sample_set.get_input_sample_set()
            self._right_sample_set = sample_set
            self._ptr_right = None
            self._ptr_right_local = None
            self._den_right = None
        else:
            raise TypeError(
                "Wrong Type: Should be samp.sample_set_base type")
        if self._comparison_sample_set._domain is None:
            self._comparison_sample_set.set_domain(
                sample_set.get_domain())
        else:
            if not np.allclose(self._comparison_sample_set._domain,
                               sample_set._domain):
                raise samp.domain_not_matching(
                    "Domain does not match comparison set.")

    def get_comparison_sample_set(self):
        r"""
        Returns a reference to the comparison sample set for this comparison.

        :rtype: :class:`~bet.sample.sample_set_base`
        :returns: comparison sample set

        """
        return self._comparison_sample_set

    def get_comparison(self):
        r"""
        Wrapper for `get_comparison_sample_set`.
        """
        return self.get_comparison_sample_set()

    def set_comparison_sample_set(self, comparison_sample_set):
        r"""
        Sets the comparison sample set for this comparison.

        :param comparison_sample_set: comparison sample set
        :type comparison_sample_set: :class:`~bet.sample.sample_set_base`

        """
        if isinstance(comparison_sample_set, samp.sample_set_base):
            output_dims = []
            output_dims.append(comparison_sample_set.get_dim())
            if self._right_sample_set is not None:
                output_dims.append(self._right_sample_set.get_dim())
            if self._left_sample_set is not None:
                output_dims.append(self._left_sample_set.get_dim())
            if len(output_dims) == 1:
                self._comparison_sample_set = comparison_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._comparison_sample_set = comparison_sample_set
            else:
                raise samp.dim_not_matching("dimension of values incorrect")
        else:
            raise AttributeError(
                "Wrong Type: Should be samp.sample_set_base type")
        # if a new emulation set is provided, forget the comparison evaluation.
        if self._left_sample_set is not None:
            self._left_sample_set._comparison_densities = None
        if self._right_sample_set is not None:
            self._right_sample_set._comparison_densities = None

    def set_comparison(self, sample_set):
        r"""
        Wrapper for `set_comparison_sample_set`.

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`

        """
        return self.set_comparison_sample_set(sample_set)

    def clip(self, lnum, rnum=None, copy=True):
        r"""
        Creates and returns a comparison with the the first `lnum`
        and `rnum` entries of the left and right sample sets, respectively.

        :param int lnum: number of values in left sample set to return.
        :param int rnum: number of values in right sample set to return.
            If ``rnum==None``, set ``rnum=lnum``.
        :param bool copy: Pass comparison_sample_set by value instead of pass
            by reference (use same pointer to sample set object).

        :rtype: :class:`~bet.sample.comparison`
        :returns: clipped comparison

        """
        if rnum is None:  # can clip by same amount
            rnum = lnum
        if lnum > 0:
            cl = self._left_sample_set.clip(lnum)
        else:
            cl = self._left_sample_set.copy()
        if rnum > 0:
            cr = self._right_sample_set.clip(rnum)
        else:
            cr = self._right_sample_set.copy()

        if copy:
            comp_set = self._comparison_sample_set.copy()
        else:
            comp_set = self._comparison_sample_set

        return comparison(sample_set_left=cl,
                          sample_set_right=cr,
                          comparison_sample_set=comp_set)

    def merge(self, comp):
        r"""
        Merges a given comparison with this one by merging the input and
        output sample sets.

        :param comp: comparison object to merge with.
        :type comp: :class:`bet.sample.comparison`

        :rtype: :class:`bet.sample.comparison`
        :returns: Merged comparison
        """
        ml = self._left_sample_set.merge(comp._left_sample_set)
        mr = self._right_sample_set.merge(comp._right_sample_set)
        il, ir = self._ptr_left, self._ptr_right
        if comp._ptr_left is not None:
            il += comp._ptr_left
        if comp._ptr_right is not None:
            ir += comp._ptr_right
        return comparison(sample_set_left=ml,
                          sample_set_right=mr,
                          comparison_sample_set=self._comparison_sample_set,
                          ptr_left=il,
                          ptr_right=ir)

    def slice(self,
              dims=None):
        r"""
        Slices the left and right of the comparison.

        :param list dims: list of indices (dimensions) of sample set to include

        :rtype: :class:`~bet.sample.comparison`
        :returns: sliced comparison

        """
        slice_list = ['_values', '_values_local',
                      '_error_estimates', '_error_estimates_local',
                      ]
        slice_list2 = ['_jacobians', '_jacobians_local']

        comp_ss = samp.sample_set(len(dims))
        left_ss = samp.sample_set(len(dims))
        right_ss = samp.sample_set(len(dims))

        if self._comparison_sample_set._domain is not None:
            comp_ss.set_domain(self._comparison_sample_set._domain[dims, :])

        if self._left_sample_set._domain is not None:
            left_ss.set_domain(self._left_sample_set._domain[dims, :])
        if self._left_sample_set._reference_value is not None:
            left_ss.set_reference_value(
                self._left_sample_set._reference_value[dims])

        if self._right_sample_set._domain is not None:
            right_ss.set_domain(self._right_sample_set._domain[dims, :])
        if self._right_sample_set._reference_value is not None:
            right_ss.set_reference_value(
                self._right_sample_set._reference_value[dims])

        for obj in slice_list:
            val = getattr(self._left_sample_set, obj)
            if val is not None:
                setattr(left_ss, obj, val[:, dims])
            val = getattr(self._right_sample_set, obj)
            if val is not None:
                setattr(right_ss, obj, val[:, dims])
            val = getattr(self._comparison_sample_set, obj)
            if val is not None:
                setattr(comp_ss, obj, val[:, dims])
        for obj in slice_list2:
            val = getattr(self._left_sample_set, obj)
            if val is not None:
                nval = np.copy(val)
                nval = nval.take(dims, axis=1)
                nval = nval.take(dims, axis=2)
                setattr(left_ss, obj, nval)
            val = getattr(self._right_sample_set, obj)
            if val is not None:
                nval = np.copy(val)
                nval = nval.take(dims, axis=1)
                nval = nval.take(dims, axis=2)
                setattr(right_ss, obj, nval)

        comp = comparison(sample_set_left=left_ss,
                          sample_set_right=right_ss,
                          comparison_sample_set=comp_ss)
        # additional attributes to copy over here.
        # maybe "setup"?
        return comp

    def global_to_local(self):
        """
        Call global_to_local for ``sample_set_left`` and
        ``sample_set_right``.

        """
        if self._left_sample_set is not None:
            self._left_sample_set.global_to_local()
        if self._right_sample_set is not None:
            self._right_sample_set.global_to_local()
        if self._comparison_sample_set is not None:
            self._comparison_sample_set.global_to_local()

    def local_to_global(self):
        """
        Call local_to_global for ``sample_set_left``,
        ``sample_set_right``, and ``comparison_sample_set``.

        """
        if self._left_sample_set is not None:
            self._left_sample_set.local_to_global()
        if self._right_sample_set is not None:
            self._right_sample_set.local_to_global()
        if self._comparison_sample_set is not None:
            self._comparison_sample_set.local_to_global()

    def estimate_volume_mc(self):
        r"""
        Applies MC assumption to volumes of both sets.
        """
        self._left_sample_set.estimate_volume_mc()
        self._right_sample_set.estimate_volume_mc()

    def set_left_probabilities(self, probabilities):
        r"""
        Allow overwriting of probabilities for the left sample set.

        :param probabilities: probabilities to overwrite the ones in the
            left sample set.
        :type probabilities: list, tuple, or `numpy.ndarray`

        """
        if self.get_left().check_num() != len(probabilities):
            raise AttributeError("Length of probabilities incorrect.")
        self._left_sample_set.set_probabilities(probabilities)
        self._left_sample_set.global_to_local()
        self._left_sample_set._comparison_densities = None
        self._den_left = None

    def set_right_probabilities(self, probabilities):
        r"""
        Allow overwriting of probabilities for the right sample set.

        :param probabilities: probabilities to overwrite the ones in the
            right sample set.
        :type probabilities: list, tuple, or `numpy.ndarray`

        """
        if self.get_right().check_num() != len(probabilities):
            raise AttributeError("Length of probabilities incorrect.")
        self._right_sample_set._probabilities = probabilities
        self._right_sample_set.global_to_local()
        self._right_sample_set._comparison_densities = None
        self._den_right = None

    def get_left_probabilities(self):
        r"""
        Wrapper for ``get_probabilities`` for the left sample set.
        """
        return self._left_sample_set.get_probabilities()

    def get_right_probabilities(self):
        r"""
        Wrapper for ``get_probabilities`` for the right sample set.
        """
        return self._right_sample_set.get_probabilities()

    def set_volume_comparison(self, sample_set, comparison_sample_set=None):
        r"""
        Wrapper to use the comparison sample set for the
        calculation of volumes on the sample sets (as opposed to using the
        Monte-Carlo assumption or setting volumes manually.)

        .. seealso::

            :meth:`bet.compareP.comparison.estimate_volume_mc``
            :meth:`bet.compareP.comparison.set_left_volume_comparison``
            :meth:`bet.compareP.comparison.set_right_volume_comparison``

        :param sample_set: sample set
        :type sample_set: :class:`~bet.sample.sample_set_base`
        :param comparison_sample_set: comparison sample set
        :type comparison_sample_set: :class:`~bet.sample.sample_set_base`


        """
        if comparison_sample_set is not None:
            if not isinstance(comparison_sample_set, samp.sample_set_base):
                msg = "Wrong type specified for `emulation_set`.\n"
                msg += "Please specify a `~bet.sample.sample_set_base`."
                raise AttributeError(msg)
            else:
                sample_set.estimate_volume_emulated(comparison_sample_set)
        else:
            # if not defined, use existing comparison set for volumes.
            sample_set.estimate_volume_emulated(self._comparison_sample_set)

    def set_left_volume_comparison(self, comparison_sample_set=None):
        r"""
        Use an comparison sample set to define volumes for the left set.
        """
        self.set_volume_comparison(self.get_left(), comparison_sample_set)
        self._den_left = None  # if volumes change, so will densities.

    def set_right_volume_comparison(self, comparison_sample_set=None):
        r"""
        Use an comparison sample set to define volumes for the right set.

        :param comparison_sample_set: comparison sample set
        :type comparison_sample_set: :class:`~bet.sample.sample_set_base`

        """
        self.set_volume_comparison(self.get_right(), comparison_sample_set)
        self._den_right = None  # if volumes change, so will densities.

    def estimate_densities_left(self):
        r"""
        Evaluates density function for the left probability measure
        at the set of samples defined in `comparison_sample_set`.

        """
        s_set = self.get_left()
        if self._ptr_left_local is None:
            self.set_ptr_left()
        s_set = density_estimate(s_set, self._ptr_left_local)
        self._den_left = s_set._comparison_densities
        return self._den_left

    def estimate_densities_right(self):
        r"""
        Evaluates density function for the right probability measure
        at the set of samples defined in ``comparison_sample_set``.

        """
        s_set = self.get_right()
        if self._ptr_right_local is None:
            self.set_ptr_right()
        s_set = density_estimate(s_set, self._ptr_right_local)
        self._den_right = s_set._comparison_densities
        return self._den_right

    def estimate_right_densities(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.estimate_densities_right``.
        """
        return self.estimate_densities_right()

    def estimate_left_densities(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.estimate_densities_left``.
        """
        return self.estimate_densities_left()

    def get_densities_right(self):
        r"""
        Returns right comparison density.
        """
        return self._den_right

    def get_densities_left(self):
        r"""
        Returns left comparison density.
        """
        return self._den_left

    def get_left_densities(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.get_densities_left``.
        """
        return self.get_densities_left()

    def get_right_densities(self):
        r"""
        Wrapper for ``bet.postProcess.compareP.get_densities_right``.
        """
        return self.get_densities_right()

    def estimate_densities(self, globalize=True,
                           comparison_sample_set=None):
        r"""
        Evaluate density functions for both left and right sets using
        the set of samples defined in ``self._comparison_sample_set``.

        :param bool globalize: globalize left/right sample sets
        :param comparison_sample_set: comparison sample set
        :type comparison_sample_set: :class:`~bet.sample.sample_set_base`

        :rtype: ``numpy.ndarray``, ``numpy.ndarray``
        :returns: left and right density values

        """
        if globalize:  # in case probabilities were re-set but not local
            self.global_to_local()

        comp_set = self.get_comparison_sample_set()
        if comp_set is None:
            raise AttributeError("Missing comparison set.")
        self.check_domain()

        # set pointers if they have not already been set
        if self._ptr_left_local is None:
            self.set_ptr_left(globalize)
        if self._ptr_right_local is None:
            self.set_ptr_right(globalize)
        self.check_dim()

        left_set, right_set = self.get_left(), self.get_right()

        if left_set._volumes is None:
            if comparison_sample_set is None:
                msg = " Volumes missing from left. Using MC assumption."
                logging.warning(msg)
                left_set.estimate_volume_mc()
            else:
                self.set_left_volume_comparison(comparison_sample_set)
        else:  # volumes present and comparison passed
            if comparison_sample_set is not None:
                msg = " Overwriting left volumes with comparison ones."
                logging.warning(msg)
                self.set_left_volume_comparison(comparison_sample_set)

        if right_set._volumes is None:
            if comparison_sample_set is None:
                msg = " Volumes missing from right. Using MC assumption."
                logging.warning(msg)
                right_set.estimate_volume_mc()
            else:
                msg = " Overwriting right volumes with comparison ones."
                logging.warning(msg)
                self.set_right_volume_comparison(comparison_sample_set)
        else:  # volumes present and comparison passed
            if comparison_sample_set is not None:
                self.set_right_volume_comparison(comparison_sample_set)

        # compute densities
        self.estimate_densities_left()
        self.estimate_densities_right()

        if globalize:
            self.local_to_global()
        return self._den_left, self._den_right

    def value(self, functional='tv', **kwargs):
        r"""
        Compute value capturing some meaure of similarity using the
        evaluated densities on a shared comparison set.
        If either density evaluation is missing, re-compute it.

        :param funtional: a function representing a measure of similarity
        :type functional: method that takes in two lists/arrays and returns
            a scalar value (measure of similarity)

        :rtype: float
        :returns: value representing a measurement between the left and right
            sample sets, ideally a measure of similarity, a distance, a metric.

        """
        left_den, right_den = self.get_left_densities(), self.get_right_densities()
        if left_den is None:
            # logging.log(20,"Left density missing. Estimating now.")
            left_den = self.estimate_densities_left()
        if right_den is None:
            # logging.log(20,"Right density missing. Estimating now.")
            right_den = self.estimate_densities_right()

        if functional in ['tv', 'totvar',
                          'total variation', 'total-variation', '1']:
            dist = ds.minkowski(left_den, right_den, 1, w=0.5, **kwargs)
        elif functional in ['mink', 'minkowski']:
            dist = ds.minkowski(left_den, right_den, **kwargs)
        elif functional in ['norm']:
            dist = ds.norm(left_den - right_den, **kwargs)
        elif functional in ['euclidean', '2-norm', '2']:
            dist = ds.minkowski(left_den, right_den, 2, **kwargs)
        elif functional in ['sqhell', 'sqhellinger']:
            dist = ds.sqeuclidean(np.sqrt(left_den), np.sqrt(right_den)) / 2.0
        elif functional in ['hell', 'hellinger']:
            return np.sqrt(self.value('sqhell'))
        else:
            dist = functional(left_den, right_den, **kwargs)

        return dist / self._comparison_sample_set.check_num()


def compare(left_set, right_set, num_mc_points=1000, choice='input'):
    r"""
    This is a convience function to quickly instantiate and return
    a `~bet.postProcess.comparison` object.

    .. seealso::

        :class:`bet.compareP.comparison`
        :meth:`bet.compareP.compare_inputs`
        :meth:`bet.compareP.compare_outputs`

    :param left set: sample set in left position
    :type left set: :class:`bet.sample.sample_set_base`
    :param right set: sample set in right position
    :type right set: :class:`bet.sample.sample_set_base`
    :param int num_mc_points: number of values of sample set to return
    :param choice: If discretization, choose 'input' (default) or 'output'
    :type choice: string

    :rtype: :class:`~bet.postProcess.compareP.comparison`
    :returns: comparison object

    """
    # extract sample set
    if isinstance(left_set, samp.discretization):
        msg = 'Discretization passed. '
        if choice == 'input':
            msg += 'Using input sample set.'
            left_set = left_set.get_input_sample_set()
        else:
            msg += 'Using output sample set.'
            left_set = left_set.get_output_sample_set()
        logging.info(msg)

    if isinstance(right_set, samp.discretization):
        msg = 'Discretization passed. '
        if choice == 'input':
            msg += 'Using input sample set.'
            right_set = right_set.get_input_sample_set()
        else:
            msg += 'Using output sample set.'
            right_set = right_set.get_output_sample_set()
        logging.info(msg)

    if not num_mc_points > 0:
        raise ValueError("Please specify positive num_mc_points")

    # make integration sample set
    assert left_set.get_dim() == right_set.get_dim()
    assert np.array_equal(left_set.get_domain(), right_set.get_domain())
    comp_set = samp.sample_set(left_set.get_dim())
    comp_set.set_domain(right_set.get_domain())
    comp_set = bsam.random_sample_set('r', comp_set, num_mc_points)

    # to be generating a new random sample set pass an integer argument
    comp = comparison(comp_set, left_set, right_set)

    return comp


def compare_inputs(left_set, right_set, num_mc_points=1000):
    r"""
    This is a convience function to quickly instantiate and return
    a `~bet.postProcess.comparison` object. If discretizations are passed,
    the respective input sample sets will be compared.

    .. seealso::

        :class:`bet.compareP.comparison`
        :meth:`bet.compareP.compare`

    :param left set: sample set in left position
    :type left set: :class:`bet.sample.sample_set_base`
    :param right set: sample set in right position
    :type right set: :class:`bet.sample.sample_set_base`
    :param int num_mc_points: number of values of sample set to return

    :rtype: :class:`~bet.postProcess.compareP.comparison`
    :returns: comparison object

    """
    return compare(left_set, right_set, num_mc_points, 'input')


def compare_outputs(left_set, right_set, num_mc_points=1000):
    r"""
    This is a convience function to quickly instantiate and return
    a `~bet.postProcess.comparison` object. If discretizations are passed,
    the respective output sample sets will be compared.

    .. seealso::

        :class:`bet.compareP.comparison`
        :meth:`bet.compareP.compare`

    :param left set: sample set in left position
    :type left set: :class:`bet.sample.sample_set_base`
    :param right set: sample set in right position
    :type right set: :class:`bet.sample.sample_set_base`
    :param int num_mc_points: number of values of sample set to return

    :rtype: :class:`~bet.postProcess.compareP.comparison`
    :returns: comparison object

    """
    return compare(left_set, right_set, num_mc_points, 'output')
