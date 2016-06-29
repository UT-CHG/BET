# Copyright (C) 2014-2016 The BET Development Team

"""
This module provides methods for creating simple function approximations to be
used by :mod:`~bet.calculateP.calculateP`. These simple function approximations
are returned as `bet.sample.sample_set` objects.
"""
from bet.Comm import comm, MPI 
import numpy as np
import bet.calculateP.voronoiHistogram as vHist
import collections, logging
import bet.util as util
import bet.sample as samp

class wrong_argument_type(Exception):
    """
    Exception for when the argument for data_set is not one of the acceptible
    types.
    """


def uniform_partition_uniform_distribution_rectangle_size(data_set, Q_ref,
                                                          rect_size, M=50,
                                                          num_d_emulate=1E6):
    r"""
    Creates a simple function approximation of :math:`\rho_{\mathcal{D}}`
    where :math:`\rho_{\mathcal{D}}` is a uniform probability density on
    a generalized rectangle centered at ``Q_ref``.
    The support of this density is defined by ``rect_size``, which determines
    the size of the generalized rectangle.
    The simple function approximation is then defined by determining ``M``
    Voronoi cells (i.e., "bins") partitioning :math:`\mathcal{D}`. These
    bins are only implicitly defined by ``M`` samples in :math:`\mathcal{D}`.
    Finally, the probabilities of each of these bins is computed by
    sampling from :math:`\rho{\mathcal{D}}` and using nearest neighbor
    searches to bin these samples in the ``M`` implicitly defined bins.
    The result is the simple function approximation denoted by
    :math:`\rho_{\mathcal{D},M}`.

    Note that all computations in the measure-theoretic framework that
    follow from this are for the fixed simple function approximation
    :math:`\rho_{\mathcal{D},M}`.

    :param int M: Defines number M samples in D used to define
        :math:`\rho_{\mathcal{D},M}` The choice of M is something of an "art" -
        play around with it and you can get reasonable results with a
        relatively small number here like 50.
    :param rect_size: Determines the size of the support of the
        uniform distribution on a generalized rectangle
    :type rect_size: double or list()
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption
    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` 
        or :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param Q_ref: :math:`Q(`\lambda_{reference})`
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)

    :rtype: :class:`~bet.sample.voronoi_sample_set`
    :returns: sample_set object defininng simple function approximation
    """
    if isinstance(data_set, samp.sample_set_base):
        data_set.check_num()
        dim = data_set._dim
        values = data_set._values
    elif isinstance(data_set, samp.discretization):
        data_set.check_nums()
        dim = data_set._output_sample_set._dim
        values = data_set._output_sample_set._values
    elif isinstance(data_set, np.ndarray):
        data_set.shape[0]
        dim = data_set.shape[1]
        values = data_set
    else:
        msg = "The first argument must be of type bet.sample.sample_set, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    if not isinstance(rect_size, collections.Iterable):
        rect_size = rect_size * np.ones((dim,))
    if np.any(np.less_equal(rect_size, 0)):
        msg = 'rect_size must be greater than 0'
        raise wrong_argument_type(msg)

    r'''
    Create M samples defining M Voronoi cells (i.e., "bins") in D used to
    define the simple function approximation :math:`\rho_{\mathcal{D},M}`.

    This does not have to be random, but here we assume this to be the case.
    We can choose these samples deterministically but that fails to scale with
    dimension efficiently.

    Note that these M samples are chosen for the sole purpose of determining
    the bins used to create the approximation to :math:`rho_{\mathcal{D}}`.

    We call these M samples "d_distr_samples" because they are samples on the
    data space and the distr implies these samples are chosen to create the
    approximation to the probability measure (distribution) on D.

    Note that we create these samples in a set containing the hyperrectangle in
    order to get output cells with zero probability. If all of the
    d_dstr_samples were taken from within the support of
    :math:`\rho_{\mathcal{D}}` then each of the M bins would have positive
    probability. This would in turn imply that the support of
    :math:`\rho_{\Lambda}` is all of :math:`\Lambda`.
    '''

    if comm.rank == 0:
        d_distr_samples = 1.5 * rect_size * (np.random.random((M,
                                                               dim)) - 0.5) + Q_ref
    else:
        d_distr_samples = np.empty((M, dim))
    comm.Bcast([d_distr_samples, MPI.DOUBLE], root=0)

    # Initialize sample set object
    s_set = samp.voronoi_sample_set(dim)
    s_set.set_values(d_distr_samples)
    s_set.set_kdtree()

    r'''
    Compute probabilities in the M bins used to define
    :math:`\rho_{\mathcal{D},M}` by Monte Carlo approximations
    that in this context amount to binning with nearest neighbor
    approximations the num_d_emulate samples taken from
    :math:`\rho_{\mathcal{D}}`.
    '''
    # Generate the samples from :math:`\rho_{\mathcal{D}}`
    num_d_emulate_local = int((num_d_emulate/comm.size) + \
                        (comm.rank < num_d_emulate%comm.size))

    int(num_d_emulate / comm.size) + 1
    d_distr_emulate = rect_size * (np.random.random((num_d_emulate_local,
                                                     dim)) - 0.5) + Q_ref

    # Bin these samples using nearest neighbor searches
    (_, k) = s_set.query(d_distr_emulate)

    count_neighbors = np.zeros((M,), dtype=np.int)
    for i in range(M):
        count_neighbors[i] = np.sum(np.equal(k, i))

    # Use the binning to define :math:`\rho_{\mathcal{D},M}`
    ccount_neighbors = np.copy(count_neighbors)
    comm.Allreduce([count_neighbors, MPI.INT], [ccount_neighbors, MPI.INT],
                   op=MPI.SUM)
    count_neighbors = ccount_neighbors
    rho_D_M = count_neighbors.astype(np.float64) / float(num_d_emulate)
    s_set.set_probabilities(rho_D_M)

    '''
    NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    above, while possibly informed by the sampling of the map Q, do not require
    solving the model EVER! This can be done "offline" so to speak. The results
    can then be stored and accessed later by the algorithm using a completely
    different set of parameter samples and model solves.
    '''
    if isinstance(data_set, samp.discretization):
        data_set._output_probability_set = s_set
    return s_set

def uniform_partition_uniform_distribution_rectangle_scaled(data_set, Q_ref,
                                                            rect_scale=0.2, M=50,
                                                            num_d_emulate=1E6):
    r"""
    Creates a simple function approximation of :math:`\rho_{\mathcal{D}}`
    where :math:`\rho_{\mathcal{D}}` is a uniform probability density on
    a generalized rectangle centered at ``Q_ref``.
    The support of this density is defined by ``rect_scale``, which determines
    the size of the generalized rectangle by scaling the circumscribing 
    generalized rectangle of :math:`\mathcal{D}`.
    The simple function approximation is then defined by determining ``M ``
    Voronoi cells (i.e., "bins") partitioning :math:`\mathcal{D}`. These
    bins are only implicitly defined by ``M`` samples in :math:`\mathcal{D}`.
    Finally, the probabilities of each of these bins is computed by 
    sampling from :math:`\rho{\mathcal{D}}` and using nearest neighbor 
    searches to bin these samples in the ``M`` implicitly defined bins.
    The result is the simple function approximation denoted by
    :math:`\rho_{\mathcal{D},M}`.

    Note that all computations in the measure-theoretic framework that
    follow from this are for the fixed simple function approximation
    :math:`\rho_{\mathcal{D},M}`.

    :param int M: Defines number M samples in D used to define
        :math:`\rho_{\mathcal{D},M}` The choice of M is something of an "art" -
        play around with it and you can get reasonable results with a
        relatively small number here like 50.
    :param rect_scale: The scale used to determine the support of the
        uniform distribution as ``rect_size = (data_max-data_min)*rect_scale``
    :type rect_scale: double or list()
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption 
    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param Q_ref: :math:`Q(`\lambda_{reference})`
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    
    :rtype: :class:`~bet.sample.voronoi_sample_set`
    :returns: sample_set object defininng simple function approximation
    """

    if isinstance(data_set, samp.sample_set_base):
        num = data_set.check_num()
        dim = data_set._dim
        values = data_set._values
    elif isinstance(data_set, samp.discretization):
        num = data_set.check_nums()
        dim = data_set._output_sample_set._dim
        values = data_set._output_sample_set._values
    elif isinstance(data_set, np.ndarray):
        num = data_set.shape[0]
        dim = data_set.shape[1]
        values = data_set
    else:
        msg = "The first argument must be of type bet.sample.sample_set, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    rect_size = (np.max(values, 0) - np.min(values, 0))*rect_scale

    return uniform_partition_uniform_distribution_rectangle_size(data_set, Q_ref,
                                                                 rect_size, M,
                                                                 num_d_emulate)

def uniform_partition_uniform_distribution_rectangle_domain(data_set, rect_domain,
                                                            M=50, num_d_emulate=1E6):
    r"""
    Creates a simple function approximation of :math:`\rho_{\mathcal{D}}`
    where :math:`\rho_{\mathcal{D}}` is a uniform probability density on
    a generalized rectangle defined by ``rect_domain``.
    The simple function approximation is then defined by determining ``M``
    Voronoi cells (i.e., "bins") partitioning :math:`\mathcal{D}`. These
    bins are only implicitly defined by ``M ``samples in :math:`\mathcal{D}`.
    Finally, the probabilities of each of these bins is computed by
    sampling from :math:`\rho{\mathcal{D}}` and using nearest neighbor
    searches to bin these samples in the ``M`` implicitly defined bins.
    The result is the simple function approximation denoted by
    :math:`\rho_{\mathcal{D},M}`.

    Note that all computations in the measure-theoretic framework that
    follow from this are for the fixed simple function approximation
    :math:`\rho_{\mathcal{D},M}`.

    :param int M: Defines number M samples in D used to define
        :math:`\rho_{\mathcal{D},M}` The choice of M is something of an "art" -
        play around with it and you can get reasonable results with a
        relatively small number here like 50.
    :param rect_domain: The support of the density
    :type rect_domain: double or list()
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption
    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param Q_ref: :math:`Q(`\lambda_{reference})`
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)


    :rtype: :class:`~bet.sample.voronoi_sample_set`
    :returns: sample_set object defininng simple function approximation
    """

    # make sure the shape of the data and the domain are correct
    if isinstance(data_set, samp.sample_set_base):
        data_set.check_num()
        values = data_set._values
    elif isinstance(data_set, samp.discretization):
        num = data_set.check_nums()
        dim = data_set._output_sample_set._dim
        values = data_set._output_sample_set._values
    elif isinstance(data_set, np.ndarray):
        data_set.shape[0]
        values = data_set
    else:
        msg = "The first argument must be of type bet.sample.sample_set, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    data = values
    rect_domain = util.fix_dimensions_data(rect_domain, data.shape[1])
    domain_center = np.mean(rect_domain, 0)
    domain_lengths = np.max(rect_domain, 0) - np.min(rect_domain, 0)


    return uniform_partition_uniform_distribution_rectangle_size(data_set,
                        domain_center, domain_lengths, M, num_d_emulate)


def regular_partition_uniform_distribution_rectangle_size(data_set, Q_ref, rect_size, center_pts_per_edge=1):
    r"""
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho_{\mathcal{D},M}` is a uniform probability density
    centered at ``Q_ref`` with ``rect_size`` of the width of a hyperrectangle.

    Since rho_D is a uniform distribution on a hyperrectanlge we can represent
    it exactly.

    :param rect_size: The size used to determine the width of the uniform
        distribution
    :type rect_size: double or list()
    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param Q_ref: :math:`Q(\lambda_{reference})`
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :param center_pts_per_edge: Does nothing. Is here for backward compatability.

    :rtype: :class:`~bet.sample.rectangle_sample_set`
    :returns: sample_set object defining simple function approximation

    """

    if isinstance(data_set, samp.sample_set_base):
        data_set.check_num()
        dim = data_set._dim
        values = data_set._values
    elif isinstance(data_set, samp.discretization):
        data_set.check_nums()
        dim = data_set._output_sample_set._dim
        values = data_set._output_sample_set._values
    elif isinstance(data_set, np.ndarray):
        data_set.shape[0]
        dim = data_set.shape[1]
        values = data_set
    else:
        msg = "The first argument must be of type bet.sample.sample_set_base, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    data = values

    if not isinstance(rect_size, collections.Iterable):
        rect_size = rect_size * np.ones((dim,))
    if np.any(np.less_equal(rect_size, 0)):
        msg = 'rect_size must be greater than 0'
        raise wrong_argument_type(msg)

    maxes = [Q_ref + 0.5*np.array(rect_size)]
    mins = [Q_ref - 0.5*np.array(rect_size)]
    s_set = samp.rectangle_sample_set(dim)
    s_set.setup(maxes, mins)
    s_set.set_probabilities(np.array([1.0, 0.0]))

    if isinstance(data_set, samp.discretization):
        data_set._output_probability_set = s_set
    return s_set


def regular_partition_uniform_distribution_rectangle_domain(data_set,
                                                            rect_domain,
                                                            center_pts_per_edge=1):
    r"""
    Creates a simple function appoximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho{\mathcal{D}, M}` is a uniform probablity density over the
    hyperrectangular domain specified by ``rect_domain``.

    Since :math:`\rho_\mathcal{D}` is a uniform distribution on a
    hyperrectangle we are able to represent it exactly.

    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param rect_domain: The domain overwhich :math:`\rho_\mathcal{D}` is
        uniform.
    :type rect_domain: :class:`numpy.ndarray` of shape (2, mdim)
    :param center_pts_per_edge: Does nothing. Is here for backward compatability.
    
    :rtype: :class:`~bet.sample.rectangle_sample_set`
    :returns: sample_set object defining simple function approximation
    
    """
    # make sure the shape of the data and the domain are correct
    if isinstance(data_set, samp.sample_set_base):
        num = data_set.check_num()
        dim = data_set._dim
        values = data_set._values
    elif isinstance(data_set, samp.discretization):
        num = data_set.check_nums()
        dim = data_set._output_sample_set._dim
        values =  data_set._output_sample_set._values
    elif isinstance(data_set, np.ndarray):
        num = data_set.shape[0]
        dim = data_set.shape[1]
        values = data_set
    else:
        msg = "The first argument must be of type bet.sample.sample_set_base, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    data = values 
    rect_domain = util.fix_dimensions_data(rect_domain, data.shape[1])
    domain_center = np.mean(rect_domain, 0)
    domain_lengths = np.max(rect_domain, 0) - np.min(rect_domain, 0)
 
    return regular_partition_uniform_distribution_rectangle_size(data_set,
                            domain_center, domain_lengths)

def regular_partition_uniform_distribution_rectangle_scaled(data_set, Q_ref,
                                                            rect_scale, center_pts_per_edge=1):
    r"""
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho_{\mathcal{D},M}` is a uniform probability density
    centered at ``Q_ref`` with ``rect_scale`` of the width
    of D.

    Since rho_D is a uniform distribution on a hyperrectanlge we are able
    to represent it exactly.

    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param rect_scale: The scale used to determine the width of the
        uniform distributiion as ``rect_size = (data_max-data_min)*rect_scale``
    :type rect_scale: double or list()
    :param Q_ref: :math:`Q(\lambda_{reference})`
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :param center_pts_per_edge: Does nothing. Is here for backward compatability.
    
    :rtype: :class:`~bet.sample.rectangle_sample_set`
    :returns: sample_set object defining simple function approximation

    """
    if isinstance(data_set, samp.sample_set_base):
        num = data_set.check_num()
        dim = data_set._dim
        values = data_set._values
    elif isinstance(data_set, samp.discretization):
        num = data_set.check_nums()
        dim = data_set._output_sample_set._dim
        values = data_set._output_sample_set._values
    elif isinstance(data_set, np.ndarray):
        num = data_set.shape[0]
        dim = data_set.shape[1]
        values = data_set
    else:
        msg = "The first argument must be of type bet.sample.sample_set_base, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    data = values

    if not isinstance(rect_scale, collections.Iterable):
        rect_scale = rect_scale*np.ones((dim, ))

    rect_size = (np.max(data, 0) - np.min(data, 0))*rect_scale
    return regular_partition_uniform_distribution_rectangle_size(data_set, Q_ref,
            rect_size)

def uniform_partition_uniform_distribution_data_samples(data_set):
    r"""
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho_{\mathcal{D},M}` is a uniform probability density over
    the entire ``data_domain``. Here the ``data_domain`` is the union of
    voronoi cells defined by ``data``. In other words we assign each sample the
    same probability, so ``M = len(data)`` or rather ``len(d_distr_samples) ==
    len(data)``. The purpose of this method is to approximate uniform
    distributions over irregularly shaped domains.
 
    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` 
        or :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`

    :rtype: :class:`~bet.sample.voronoi_sample_set`
    :returns: sample_set object defininng simple function approximation
    """
    if isinstance(data_set, samp.sample_set_base):
        num = data_set.check_num()
        dim = data_set._dim
        values = data_set._values
        s_set = data_set.copy()
    elif isinstance(data_set, samp.discretization):
        num = data_set.check_nums()
        dim = data_set._output_sample_set._dim
        values = data_set._output_sample_set._values
        s_set = data_set._output_sample_set.copy()
    elif isinstance(data_set, np.ndarray):
        num = data_set.shape[0]
        dim = data_set.shape[1]
        values = data_set
        s_set = samp.sample_set(dim=dim)
        s_set.set_values(values)
    else:
        msg = "The first argument must be of type bet.sample.sample_set, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)
    
    s_set.set_probabilities(np.ones((num,), dtype=np.float)/num)

    if isinstance(data_set, samp.discretization):
        data_set._output_probability_set = s_set
    return s_set


def normal_partition_normal_distribution(data_set, Q_ref, std, M, num_d_emulate=1E6):
    r"""
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho_{\mathcal{D},M}` is a multivariate normal probability
    density centered at ``Q_ref`` with standard deviation ``std`` using
    ``M`` bins sampled from the given normal distribution.

    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param int M: Defines number M samples in D used to define
        :math:`\rho_{\mathcal{D},M}` The choice of M is something of an "art" -
        play around with it and you can get reasonable results with a
        relatively small number here like 50.
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption
    :param Q_ref: :math:`Q(\lambda_{reference})`
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :param std: The standard deviation of each QoI
    :type std: :class:`~numpy.ndarray` of size (mdim,)

    :rtype: :class:`~bet.sample.voronoi_sample_set`
    :returns: sample_set object defining simple function approximation

    """
    import scipy.stats as stats
    r'''Create M smaples defining M bins in D used to define
    :math:`\rho_{\mathcal{D},M}` rho_D is assumed to be a multi-variate normal
    distribution with mean Q_ref and standard deviation std.'''
    if not isinstance(Q_ref, collections.Iterable):
        Q_ref = np.array([Q_ref])
    if not isinstance(std, collections.Iterable):
        std = np.array([std])

    covariance = std ** 2

    d_distr_samples = np.zeros((M, len(Q_ref)))
    print "d_distr_samples.shape", d_distr_samples.shape
    print "Q_ref.shape", Q_ref.shape
    print "std.shape", std.shape

    if comm.rank == 0:
        for i in range(len(Q_ref)):
            d_distr_samples[:, i] = np.random.normal(Q_ref[i], std[i], M)
    comm.Bcast([d_distr_samples, MPI.DOUBLE], root=0)

    # Initialize sample set object
    s_set = samp.voronoi_sample_set(len(Q_ref))
    s_set.set_values(d_distr_samples)
    s_set.set_kdtree()

    r'''Now compute probabilities for :math:`\rho_{\mathcal{D},M}` by sampling
    from rho_D First generate samples of rho_D - I sometimes call this
    emulation'''
    num_d_emulate_local = int((num_d_emulate/comm.size) + \
                              (comm.rank < num_d_emulate%comm.size))
    d_distr_emulate = np.zeros((num_d_emulate_local, len(Q_ref)))
    for i in range(len(Q_ref)):
        d_distr_emulate[:, i] = np.random.normal(Q_ref[i], std[i],
                                                 num_d_emulate_local)

        # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)

    (_, k) = s_set.query(d_distr_emulate)
    count_neighbors = np.zeros((M,), dtype=np.int)
    volumes = np.zeros((M,))
    for i in range(M):
        Itemp = np.equal(k, i)
        count_neighbors[i] = np.sum(Itemp)
        volumes[i] = np.sum(1.0 / stats.multivariate_normal.pdf \
            (d_distr_emulate[Itemp, :], Q_ref, covariance))
    # Now define probability of the d_distr_samples
    # This together with d_distr_samples defines :math:`\rho_{\mathcal{D},M}`
    ccount_neighbors = np.copy(count_neighbors)
    comm.Allreduce([count_neighbors, MPI.INT], [ccount_neighbors, MPI.INT],
                   op=MPI.SUM)
    count_neighbors = ccount_neighbors
    cvolumes = np.copy(volumes)
    comm.Allreduce([volumes, MPI.DOUBLE], [cvolumes, MPI.DOUBLE], op=MPI.SUM)
    volumes = cvolumes
    rho_D_M = count_neighbors.astype(np.float64) * volumes
    rho_D_M = rho_D_M / np.sum(rho_D_M)
    s_set.set_probabilities(rho_D_M)
    s_set.set_volumes(volumes)

    # NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    # above, while informed by the sampling of the map Q, do not require
    # solving the model EVER! This can be done "offline" so to speak.
    if isinstance(data_set, samp.discretization):
        data_set._output_probability_set = s_set
    return s_set


def uniform_partition_normal_distribution(data_set, Q_ref, std, M, num_d_emulate=1E6):
    r"""
    Creates a simple function approximation of :math:`\rho_{\mathcal{D},M}`
    where :math:`\rho_{\mathcal{D},M}` is a multivariate normal probability
    density centered at ``Q_ref`` with standard deviation ``std`` using
    ``M`` bins sampled from a uniform distribution with a size 4 standard
    deviations in each direction.

    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param int M: Defines number M samples in D used to define
        :math:`\rho_{\mathcal{D},M}` The choice of M is something of an "art" -
        play around with it and you can get reasonable results with a
        relatively small number here like 50.
    :param int num_d_emulate: Number of samples used to emulate using an MC
        assumption
    :param Q_ref: :math:`Q(\lambda_{reference})`
    :type Q_ref: :class:`~numpy.ndarray` of size (mdim,)
    :param std: The standard deviation of each QoI
    :type std: :class:`~numpy.ndarray` of size (mdim,)

    :rtype: :class:`~bet.sample.voronoi_sample_set`
    :returns: sample_set object defininng simple function approximation

    """
    r'''Create M samples defining M bins in D used to define
    :math:`\rho_{\mathcal{D},M}` rho_D is assumed to be a multi-variate normal
    distribution with mean Q_ref and standard deviation std.'''
    if not isinstance(Q_ref, collections.Iterable):
        Q_ref = np.array([Q_ref])
    if not isinstance(std, collections.Iterable):
        std = np.array([std])

    bin_size = 4.0 * std
    d_distr_samples = np.zeros((M, len(Q_ref)))
    if comm.rank == 0:
        d_distr_samples = bin_size * (np.random.random((M,
                                                        len(Q_ref))) - 0.5) + Q_ref
    comm.Bcast([d_distr_samples, MPI.DOUBLE], root=0)

    # Initialize sample set object
    s_set = samp.voronoi_sample_set(len(Q_ref))
    s_set.set_values(d_distr_samples)
    s_set.set_kdtree()

    r'''Now compute probabilities for :math:`\rho_{\mathcal{D},M}` by sampling
    from rho_D First generate samples of rho_D - I sometimes call this
    emulation'''
    num_d_emulate_local = int((num_d_emulate/comm.size) + \
            (comm.rank < num_d_emulate%comm.size))
    d_distr_emulate = np.zeros((num_d_emulate_local, len(Q_ref)))
    for i in range(len(Q_ref)):
        d_distr_emulate[:, i] = np.random.normal(Q_ref[i], std[i],
                                                 num_d_emulate_local)

        # Now bin samples of rho_D in the M bins of D to compute rho_{D, M}
    if len(d_distr_samples.shape) == 1:
        d_distr_samples = np.expand_dims(d_distr_samples, axis=1)

    (_, k) = s_set.query(d_distr_emulate)
    count_neighbors = np.zeros((M,), dtype=np.int)
    # volumes = np.zeros((M,))
    for i in range(M):
        Itemp = np.equal(k, i)
        count_neighbors[i] = np.sum(Itemp)

    r'''Now define probability of the d_distr_samples This together with
    d_distr_samples defines :math:`\rho_{\mathcal{D},M}`'''
    ccount_neighbors = np.copy(count_neighbors)
    comm.Allreduce([count_neighbors, MPI.INT], [ccount_neighbors, MPI.INT],
                   op=MPI.SUM)
    count_neighbors = ccount_neighbors
    rho_D_M = count_neighbors.astype(np.float64) / float(num_d_emulate)
    s_set.set_probabilities(rho_D_M)
    # NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
    # above, while informed by the sampling of the map Q, do not require
    # solving the model EVER! This can be done "offline" so to speak.
    if isinstance(data_set, samp.discretization):
        data_set._output_probability_set = s_set
    return s_set

def user_partition_user_distribution(data_set, data_partition_set,
                                          data_distribution_set):
    r"""
    Creates a user defined simple function approximation of a user
    defined distribution. The simple function discretization is
    specified in the ``data_partition_set``, and the set of i.i.d.
    samples from the distribution is specified in the
    ``data_distribution_set``.

    :param data_set: Sample set that the probability measure is defined for.
    :type data_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param data_partition_set: Sample set defining the discretization
        of the data space into Voronoi cells for which a simple function
        is defined upon.
    :type data_partition_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`
    :param data_distribution_set: Sample set containing the i.i.d. samples
        from the distribution on the data space that are binned within the
        Voronoi cells implicitly defined by the data_discretization_set.
    :type data_distribution_set: :class:`~bet.sample.discretization` or
        :class:`~bet.sample.sample_set` or :class:`~numpy.ndarray`

    :rtype: :class:`~bet.sample.voronoi_sample_set`
    :returns: sample_set object defininng simple function approximation
    """
    if isinstance(data_set, samp.sample_set_base):
        s_set = data_set.copy()
        dim = s_set._dim
    elif isinstance(data_set, samp.discretization):
        s_set = data_set._output_sample_set.copy()
        dim = s_set._dim
    elif isinstance(data_set, np.ndarray):
        dim = data_set.shape[1]
        values = data_set
        s_set = samp.sample_set(dim=dim)
        s_set.set_values(values)
    else:
        msg = "The first argument must be of type bet.sample.sample_set, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    if isinstance(data_partition_set, samp.sample_set_base):
        M = data_partition_set.check_num()
        d_distr_samples = data_partition_set._values
        dim_simpleFun = d_distr_samples.shape[1]
    elif isinstance(data_partition_set, samp.discretization):
        M = data_partition_set.check_nums()
        d_distr_samples = data_partition_set._output_sample_set._values
        dim_simpleFun = d_distr_samples.shape[1]
    elif isinstance(data_partition_set, np.ndarray):
        M = data_partition_set.shape[0]
        dim_simpleFun = data_partition_set.shape[1]
        d_distr_samples = data_partition_set
    else:
        msg = "The second argument must be of type bet.sample.sample_set, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    if isinstance(data_distribution_set, samp.sample_set_base):
        d_distr_emulate = data_distribution_set._values
        dim_MonteCarlo = d_distr_emulate.shape[1]
        num_d_emulate = data_distribution_set.check_num()
    elif isinstance(data_distribution_set, samp.discretization):
        d_distr_emulate = data_distribution_set._output_sample_set._values
        dim_MonteCarlo = d_distr_emulate.shape[1]
        num_d_emulate = data_distribution_set.check_nums()
    elif isinstance(data_distribution_set, np.ndarray):
        num_d_emulate = data_distribution_set.shape[0]
        dim_MonteCarlo = data_distribution_set.shape[1]
        d_distr_emulate = data_distribution_set
    else:
        msg = "The second argument must be of type bet.sample.sample_set, "
        msg += "bet.sample.discretization or np.ndarray"
        raise wrong_argument_type(msg)

    if np.not_equal(dim_MonteCarlo, dim) or np.not_equal(dim_simpleFun, dim):
        msg = "The argument types have conflicting dimensions"
        raise wrong_argument_type(msg)

    # Initialize sample set object
    s_set = samp.sample_set(dim)
    s_set.set_values(d_distr_samples)
    s_set.set_kdtree()

    (_, k) = s_set.query(d_distr_emulate)

    count_neighbors = np.zeros((M,), dtype=np.int)
    for i in range(M):
        count_neighbors[i] = np.sum(np.equal(k, i))

    # Use the binning to define :math:`\rho_{\mathcal{D},M}`
    ccount_neighbors = np.copy(count_neighbors)
    comm.Allreduce([count_neighbors, MPI.INT], [ccount_neighbors, MPI.INT],
                   op=MPI.SUM)
    count_neighbors = ccount_neighbors
    rho_D_M = count_neighbors.astype(np.float64) / \
              float(num_d_emulate * comm.size)
    s_set.set_probabilities(rho_D_M)

    if isinstance(data_set, samp.discretization):
        data_set._output_probability_set = s_set
    return s_set
