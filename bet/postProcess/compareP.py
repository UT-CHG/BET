import numpy as np
import logging
import bet.util as util
import bet.sample as samp
import bet.sampling.basicSampling as bsam


def distance(left_set, right_set, num_mc_points=100):
    r"""
    Creates and returns a `~bet.postProcess.metrization` object

    :param int cnum: number of values of sample set to return

    :rtype: :class:`~bet.postProcess.metrization`
    :returns: metrization object

    :class:`sample.samp.sample_set_base`
    """
    # extract sample set
    if isinstance(left_set, samp.discretization):
        left_set = left_set.get_input_sample_set()
    if isinstance(right_set, samp.discretization):
        right_set = right_set.get_input_sample_set()
    if not num_mc_points > 0:
        raise ValueError("Please specify positive num_mc_points")

    # to be generating a new random sample set pass an integer argument
    metrc = metrization(num_mc_points, left_set, right_set)

    return metrc


class metrization(object):
    """

    A data structure containing :class:`~bet.sample.samp.sample_set_base` objects and
    associated methods for computing measures of distance between pairs of them.

    """
    #: List of attribute names for attributes which are vectors or 1D
    #: :class:`numpy.ndarray`
    vector_names = ['_io_ptr_left', '_io_ptr_left_local',
                    '_io_ptr_right', '_io_ptr_right_local']

    #: List of attribute names for attributes that are
    #: :class:`sample.samp.sample_set_base`
    sample_set_names = ['_sample_set_left', '_sample_set_right',
                        '_integration_sample_set']

    def __init__(self, integration_sample_set,
                 sample_set_left, sample_set_right,
                 io_ptr_left=None, io_ptr_right=None):
        #: Input sample set :class:`~bet.sample.samp.sample_set_base`
        self._sample_set_left = sample_set_left
        #: Output sample set :class:`~bet.sample.samp.sample_set_base`
        self._sample_set_right = sample_set_right
        #: Integration/Emulation set :class:`~bet.sample.samp.sample_set_base`
        self._integration_sample_set = integration_sample_set
        #: Pointer from ``self._integration_sample_set`` to
        #: ``self._sample_set_left``
        self._io_ptr_left = None
        #: Pointer from ``self._integration_sample_set`` to
        #: ``self._sample_set_right``
        self._io_ptr_right = None
        #: local integration left ptr for parallelsim
        self._io_ptr_left_local = None
        #: local integration right ptr for parallelism
        self._io_ptr_right_local = None

        # extract sample set
        if isinstance(sample_set_left, samp.discretization):
            left_set = sample_set_left.get_input_sample_set()
        if isinstance(sample_set_right, samp.discretization):
            right_set = sample_set_right.get_input_sample_set()

        if isinstance(sample_set_left, samp.sample_set_base):
            left_set = sample_set_left
        else:
            raise TypeError(
                "Please specify a `~bet.sample.samp.sample_set_base` object.")
        if isinstance(sample_set_right, samp.sample_set_base):
            right_set = sample_set_right
        else:
            raise TypeError(
                "Please specify a `~bet.sample.samp.sample_set_base` object.")

        # assert dimensions match
        if left_set._dim != right_set._dim:
            msg = "These sample sets must have the same dimension."
            raise samp.dim_not_matching(msg)
        else:
            dim = left_set.get_dim()

        # assert domains match
        if left_set._domain is not None and right_set._domain is not None:
            if not np.allclose(left_set._domain, right_set._domain):
                msg = "These sample sets have different domains."
                raise samp.domain_not_matching(msg)
        if left_set._domain is None or right_set._domain is None:
            msg = "One or more of your sets is missing a domain."
            raise samp.domain_not_matching(msg)
        else:  # since the domains match, we can choose either.
            domain = left_set._domain

        if integration_sample_set is None:
            logging.info("No integration set defined. Constructing one with MC \
                        assumption with 100 samples. You can add more later using \
                        the returned compareP.metrization object.")
            integration_sample_set = 100
        else:
            if isinstance(integration_sample_set, samp.sample_set_base):
                dim_I = integration_sample_set._values.shape[1]
                if dim_I != dim:
                    raise samp.dim_not_matching(
                        "Dimension of integration set incorrect.")
            # If integration set given as number, we create one using `~bet.sampling.basicSampler`
            if isinstance(integration_sample_set, float):
                integration_sample_set = int(integration_sample_set)
        # if integration_sample_set is given as a number, generate set.
        if isinstance(integration_sample_set, int):
            num_mc_samples = integration_sample_set
            integration_sample_set = samp.sample_set(dim)
            self._integration_sample_set = bsam.random_sample_set('r', integration_sample_set,
                                                                  num_samples=num_mc_samples)
            logging.info(
                "Created integration set with {} MC samples".format(num_mc_samples))

    # method for checking that pointers have been set that will be
    # called by the distance function

    # set density functions, maybe print a message if MC assumption is used to estimate volumes

    # evaluate density functions at integration points, store for re-use

    # metric - wrapper around scipy now that passes density values with proper shapes.

    def globalize_ptrs(self):
        """
        Globalizes metrization pointers.

        """
        if (self._io_ptr_left_local is not None) and\
                (self._io_ptr_left is None):
            self._io_ptr_left = util.get_global_values(
                self._io_ptr_left_local)
        if (self._io_ptr_right_local is not None) and\
                (self._io_ptr_right is None):
            self._io_ptr_right = util.get_global_values(
                self._io_ptr_right_local)

    def set_io_ptr_left(self, globalize=True):
        """

        Creates the pointer from ``self._integration_sample_set`` to
        ``self._sample_set_left``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._io_ptr_left``
        """
        if self._integration_sample_set._values_local is None:
            self._integration_sample_set.global_to_local()
        (_, self._io_ptr_left_local) = self._sample_set_left.query(
            self._integration_sample_set._values_local)
        if globalize:
            self._io_ptr_left = util.get_global_values(self._io_ptr_left_local)

    def get_io_ptr_left(self):
        """

        Returns the pointer from ``self._integration_sample_set`` to
        ``self._sample_set_left``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._sample_set_left._values.shape[0],)
        :returns: self._io_ptr_left

        """
        return self._io_ptr_left

    def set_io_ptr_right(self, globalize=True):
        """

        Creates the pointer from ``self._integration_sample_set`` to
        ``self._sample_set_right``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :param bool globalize: flag whether or not to globalize
            ``self._io_ptr_right``

        """
        if self._integration_sample_set._values_local is None:
            self._integration_sample_set.global_to_local()
        (_, self._io_ptr_right_local) = self._sample_set_right.query(
            self._integration_sample_set._values_local)

        if globalize:
            self._io_ptr_right = util.get_global_values(
                self._io_ptr_right_local)

    def get_io_ptr_right(self):
        """

        Returns the pointer from ``self._integration_sample_set`` to
        ``self._sample_set_right``

        .. seealso::

            :meth:`scipy.spatial.KDTree.query``

        :rtype: :class:`numpy.ndarray` of int of shape
            (self._sample_set_right._values.shape[0],)
        :returns: self._io_ptr_right

        """
        return self._io_ptr_right

    def copy(self):
        """
        Makes a copy using :meth:`numpy.copy`.

        :rtype: :class:`~bet.sample.metrization`
        :returns: Copy of this :class:`~bet.sample.metrization`

        """
        my_copy = metrization(self._integration_sample_set.copy(),
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

        Returns a reference to the left/input sample set for this metrization.

        :rtype: :class:`~bet.sample.samp.sample_set_base`
        :returns: left sample set

        """
        return self._sample_set_left

    def set_sample_set_left(self, sample_set_left):
        """

        Sets the left/input sample set for this metrization.

        :param sample_set_left: left sample set
        :type sample_set_left: :class:`~bet.sample.samp.sample_set_base`

        """
        if isinstance(sample_set_left, sample.samp.sample_set_base):
            self._sample_set_left = sample_set_left
        else:
            raise AttributeError(
                "Wrong Type: Should be samp.sample_set_base type")

    def get_sample_set_right(self):
        """

        Returns a reference to the right/output sample set for this metrization.

        :rtype: :class:`~bet.sample.samp.sample_set_base`
        :returns: right sample set

        """
        return self._sample_set_right

    def set_sample_set_right(self, sample_set_right):
        """

        Sets the right/output sample set for this metrization.

        :param sample_set_right: right sample set
        :type sample_set_right: :class:`~bet.sample.samp.sample_set_base`

        """
        if isinstance(sample_set_right, sample.samp.sample_set_base):
            self._sample_set_right = sample_set_right
        else:
            raise AttributeError(
                "Wrong Type: Should be samp.sample_set_base type")

    def get_integration_sample_set(self):
        """

        Returns a reference to the output probability sample set for this
        metrization.

        :rtype: :class:`~bet.sample.samp.sample_set_base`
        :returns: output probability sample set

        """
        return self._integration_sample_set

    def get_integration_sample_set(self):
        """

        Returns a reference to the integration_output sample set for this
        metrization.

        :rtype: :class:`~bet.sample.samp.sample_set_base`
        :returns: integration_output sample set

        """
        return self._integration_sample_set

# TO DO: FIX THE CHECKS HERE
    def set_integration_sample_set(self, integration_sample_set):
        """

        Sets the integration_output sample set for this metrization.

        :param integration_sample_set: emupated output sample set.
        :type integration_sample_set: :class:`~bet.sample.samp.sample_set_base`

        """
        if isinstance(integration_sample_set, samp.sample_set_base):
            output_dims = []
            output_dims.append(integration_sample_set.get_dim())
            if self._sample_set_right is not None:
                output_dims.append(self._sample_set_right.get_dim())
            if self._integration_sample_set is not None:
                output_dims.append(self._integration_sample_set.get_dim())
            if len(output_dims) == 1:
                self._integration_sample_set = integration_sample_set
            elif np.all(np.array(output_dims) == output_dims[0]):
                self._integration_sample_set = integration_sample_set
            else:
                raise dim_not_matching("dimension of values incorrect")
        else:
            raise AttributeError(
                "Wrong Type: Should be samp.sample_set_base type")

#    def estimate_output_volume_integration(self):
#        """
#        Calculate the volume faction of cells approximately using Monte
#        Carlo integration.
#
#        .. note ::
#
#            This could be re-written to just use ``io_ptr_right`` instead
#            of ``_integration_sample_set``.
#
#
#        """
#        if self._integration_sample_set is None:
#            raise AttributeError("Required: _integration_sample_set")
#        else:
#            self._sample_set_right.estimate_volume_integration(\
#                    self._integration_sample_set)

#    def clip(self, cnum):
#        """
#        Creates and returns a metrization with the the first `cnum`
#        entries of the input and output sample sets.
#
#        :param int cnum: number of values of sample set to return
#
#        :rtype: :class:`~bet.sample.metrization`
#        :returns: clipped metrization
#
#        """
#        ci = self._sample_set_left.clip(cnum)
#        co = self._sample_set_right.clip(cnum)
#
#        return metrization(sample_set_left=ci,
#                              sample_set_right=co,
#                              integration_sample_set=\
#                                      self._integration_sample_set,
#                              integration_sample_set=\
#                                      self._integration_sample_set,
#                              integration_sample_set=\
#                                      self._integration_sample_set)
#
#    def merge(self, disc):
#        """
#        Merges a given metrization with this one by merging the input and
#        output sample sets.
#
#        :param disc: Discretization object to merge with.
#        :type disc: :class:`bet.sample.metrization`
#
#        :rtype: :class:`bet.sample.metrization`
#        :returns: Merged metrization
#        """
#        mi = self._sample_set_left.merge(disc._sample_set_left)
#        mo = self._sample_set_right.merge(disc._sample_set_right)
#        mei = self._integration_sample_set.merge(disc.\
#                _integration_sample_set)
#        meo = self._integration_sample_set.merge(disc.\
#                _integration_sample_set)
#
#        return metrization(sample_set_left=mi,
#                              sample_set_right=mo,
#                              integration_sample_set=\
#                                      self._integration_sample_set,
#                              integration_sample_set=mei,
#                              integration_sample_set=meo)
#
#    def choose_inputs_outputs(self,
#                              inputs=None,
#                              outputs=None):
#        """
#        Slices the inputs and outputs of the metrization.
#
#        :param list inputs: list of indices of input sample set to include
#        :param list outputs: list of indices of output sample set to include
#
#        :rtype: :class:`~bet.sample.metrization`
#        :returns: sliced metrization
#
#        """
#        slice_list = ['_values', '_values_local',
#                      '_error_estimates', '_error_estimates_local']
#        slice_list2 = ['_jacobians', '_jacobians_local']
#
#        input_ss = sample_set(len(inputs))
#        output_ss = sample_set(len(outputs))
#        input_ss.set_p_norm(self._sample_set_left._p_norm)
#        if self._sample_set_left._domain is not None:
#            input_ss.set_domain(self._sample_set_left._domain[inputs, :])
#        if self._sample_set_left._reference_value is not None:
#            input_ss.set_reference_value(self._sample_set_left._reference_value[inputs])
#
#        output_ss.set_p_norm(self._sample_set_right._p_norm)
#        if self._sample_set_right._domain is not None:
#            output_ss.set_domain(self._sample_set_right._domain[outputs, :])
#        if self._sample_set_right._reference_value is not None:
#            output_ss.set_reference_value(self._sample_set_right._reference_value[outputs])
#
#        for obj in slice_list:
#            val = getattr(self._sample_set_left, obj)
#            if val is not None:
#                setattr(input_ss, obj, val[:, inputs])
#            val = getattr(self._sample_set_right, obj)
#            if val is not None:
#                setattr(output_ss, obj, val[:, outputs])
#        for obj in slice_list2:
#            val = getattr(self._sample_set_left, obj)
#            if val is not None:
#                nval = np.copy(val)
#                nval = nval.take(outputs, axis=1)
#                nval = nval.take(inputs, axis=2)
#                setattr(input_ss, obj, nval)
#        disc = metrization(sample_set_left=input_ss,
#                              sample_set_right=output_ss)
#        return disc
#
    def local_to_global(self):
        """
        Call local_to_global for ``sample_set_left`` and
        ``sample_set_right``.
        """
        if self._sample_set_left is not None:
            self._sample_set_left.local_to_global()
        if self._sample_set_right is not None:
            self._sample_set_right.local_to_global()
