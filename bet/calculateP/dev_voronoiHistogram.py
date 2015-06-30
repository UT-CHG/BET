# -*- coding: utf-8 -*-
import numpy as np
from scipy import spatial
import bet.util as util

def center_and_layer1_points_binsize(center_pts_per_edge, center, r_size, sur_domain):
    """
    Generates a regular grid of center points that define the voronoi
    tesselation of exactly the interior of a hyperrectangle centered at
    ``center`` with sides of length ``r_size`` and the layers
    of voronoi cells that bound these interior cells. The resulting voronoi
    tesselation exactly represents the hyperrectangle.

    This method can also be used to tile ``sur_domain`` with points to define
    voronoi regions if the user sets ``r_ratio = 1``. (use binratio)

    :param list() center_pts_per_edge: number of center points per edge and
        additional two points will be added to create the bounding layer
    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param r_size: size of the length of the sides of the
        hyperrectangle to the surrounding domain
    :type r_size: double or list()
    :param sur_domain: minima and maxima of each dimension defining the
        surrounding domain
    :type sur_domain: :class:`numpy.ndarray` of shape (mdim, 2)

    :rtype: tuple
    :returns: (points, interior_and_layer1, rect_domain) where where points is an
        :class:`numpy.ndarray` of shape (num_points, dim), interior_and_layer1
        is a list() of dim :class:`numpy.ndarray`s of shape
        (center_pts_per_edge+2,), rect_domain is a :class:`numpy.ndarray` of
        shape (mdim, 2)

    """
    # determine the hyperrectangle defined by center and r_size
    rect_width = r_size*np.ones(sur_domain[:,0].shape)
    rect_domain = np.column_stack([center - .5*rect_width,
        center + .5*rect_width])
    if np.all(np.greater(r_size, rect_width)):
        msg = "The hyperrectangle defined by this size is larger than the"
        msg += "original domain."
        print msg
    
    # determine the locations of the points for the 1st bounding layer
    layer1_left = rect_domain[:, 0]-rect_width/(2*center_pts_per_edge)
    layer1_right = rect_domain[:, 1]+rect_width/(2*center_pts_per_edge)

    interior_and_layer1 = list()
    for dim in xrange(sur_domain.shape[0]):
        # create interior points and 1st layer
        int_l1 = np.linspace(layer1_left[dim],
            layer1_right[dim], center_pts_per_edge[dim]+2)
        interior_and_layer1.append(int_l1)

    # use meshgrid to make the hyperrectangle shells
    points = util.meshgrid_ndim(interior_and_layer1)
    return (points, interior_and_layer1, rect_domain)

def center_and_layer1_points(center_pts_per_edge, center, r_ratio, sur_domain):
    r"""
    Generates a regular grid of center points that define the voronoi
    tesselation of exactly the interior of a hyperrectangle centered at
    ``center`` with sides of length ``r_ratio*sur_width`` and the layers
    of voronoi cells that bound these interior cells. The resulting voronoi
    tesselation exactly represents the hyperrectangle.

    This method can also be used to tile ``sur_domain`` with points to define
    voronoi regions if the user sets ``r_ratio = 1``.

    :param list() center_pts_per_edge: number of center points per edge and
        additional two points will be added to create the bounding layer
    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param r_ratio: ratio of the length of the sides of the
        hyperrectangle to the surrounding domain
    :type r_ratio: double or list()
    :param sur_domain: minima and maxima of each dimension defining the
        surrounding domain
    :type sur_domain: :class:`numpy.ndarray` of shape (mdim, 2)

    :rtype: tuple
    :returns: (points, interior_and_layer1) where where points is an
        :class:`numpy.ndarray` of shape (num_points, dim), interior_and_layer1
        is a list() of dim :class:`numpy.ndarray`s of shape
        (center_pts_per_edge+2,), rect_domain is a :class:`numpy.ndarray` of
        shape (mdim, 2).

    """
    if np.all(np.greater(r_ratio, 1)):
        msg = "The hyperrectangle defined by this ratio is larger than the"
        msg += "original domain."
        print msg

    # determine r_size from the width of the surrounding domain
    r_size = r_ratio*(sur_domain[:, 1]-sur_domain[:, 0])

    return center_and_layer1_points_binsize(center_pts_per_edge, center,
            r_size, sur_domain)

def edges_regular_binsize(center_pts_per_edge, center, r_size, sur_domain):
    """
    Generates a sequence of arrays describing the edges of the finite voronoi
    cells in each direction. The voronoi tesselation is defined by regular grid
    of center points that define the voronoi tesselation of exactly the
    interior of a hyperrectangle centered at ``center`` with sides of length
    ``r_size`` and the layers of voronoi cells that bound these
    interior cells. The resulting voronoi tesselation exactly represents the
    hyperrectangle. The bounding voronoi cells are made finite by bounding them
    with an  additional layer to represent ``sur_domain``.
    
    This method can also be used to tile ``sur_domain`` with points to define
    voronoi regions if the user sets ``r_ratio = 1``. use binratio below

    :param list() center_pts_per_edge: number of center points per edge and
        additional two points will be added to create the bounding layer
    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param r_size: size of the length of the sides of the
        hyperrectangle to the surrounding domain
    :type r_size: double or list()
    :param sur_domain: minima and maxima of each dimension defining the
        surrounding domain
    :type sur_domain: :class:`numpy.ndarray` of shape (mdim, 2)

    :rtype: tuple
    :returns: interior_and_layer1 is a list of dim
        :class:`numpy.ndarray`s of shape (center_pts_per_edge+2,)
    """
    # TODO This is redoing stuff that was done when generating the points
    # determine the hyperrectangle defined by center and r_size
    rect_width = r_size*np.ones(sur_domain[:,0].shape)
    rect_domain = np.column_stack([center - .5*rect_width,
        center + .5*rect_width])

    if np.any(np.greater_equal(sur_domain[:, 0], rect_domain[:, 0])):
        msg = "The hyperrectangle defined by this size is larger than the"
        msg += "original domain."
        print msg
    elif np.any(np.less_equal(sur_domain[:, 1], rect_domain[:, 1])):
        msg = "The hyperrectangle defined by this size is larger than the"
        msg += "original domain."
        print msg
    
    rect_edges = list()
    rect_and_sur_edges = list()
    for dim in xrange(sur_domain.shape[0]):
        # create interior points and 1st layer
        int_l1 = np.linspace(rect_domain[dim, 0],
            rect_domain[dim, 1], center_pts_per_edge[dim]+1)
        rect_edges.append(int_l1)
        # add layers together using indexing fu
        int_l2 = np.zeros((int_l1.shape[0]+2,))
        int_l2[1:-1] = int_l1
        int_l2[0] = sur_domain[dim, 0]
        int_l2[-1] = sur_domain[dim, 1]
        rect_and_sur_edges.append(int_l2) 

    return rect_and_sur_edges

def edges_regular(center_pts_per_edge, center, r_ratio, sur_domain):
    """
    Generates a sequence of arrays describing the edges of the finite voronoi
    cells in each direction. The voronoi tesselation is defined by regular grid
    of center points that define the voronoi tesselation of exactly the
    interior of a hyperrectangle centered at ``center`` with sides of length
    ``r_ratio*sur_width`` and the layers of voronoi cells that bound these
    interior cells. The resulting voronoi tesselation exactly represents the
    hyperrectangle. The bounding voronoi cells are made finite by bounding them
    with an  additional layer to represent ``sur_domain``.
    
    This method can also be used to tile ``sur_domain`` with points to define
    voronoi regions if the user sets ``r_ratio = 1``.

    :param list() center_pts_per_edge: number of center points per edge and
        additional two points will be added to create the bounding layer
    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param r_ratio: ratio of the length of the sides of the
        hyperrectangle to the surrounding domain
    :type r_ratio: double or list()
    :param sur_domain: minima and maxima of each dimension defining the
        surrounding domain
    :type sur_domain: :class:`numpy.ndarray` of shape (mdim, 2)
    :rtype: tuple
    :returns: interior_and_layer1 is a list of dim
        :class:`numpy.ndarray`s of shape (center_pts_per_edge+2,)

    """
    if np.any(np.greater(r_ratio, 1)):
        msg = "The hyperrectangle defined by this ratio is larger than the"
        msg += "original domain. RATIO: {}".format(r_ratio)
        print msg
    
    # determine r_size from the width of the surrounding domain
    r_size = r_ratio*(sur_domain[:, 1]-sur_domain[:, 0])

    return edges_regular_binsize(center_pts_per_edge, center, r_size, sur_domain)

def edges_from_points(points):
    """
    Given a sequence of arrays describing the voronoi points in each dimension
    that define a set of bounded hyperrectangular bins returns the edges of bins
    formed by voronoi cells along each dimensions.
    
    :param points: the coordindates of voronoi points that would generate
        these bins in each dimensions
    :type points: list of dim :class:`numpy.ndarray`s of shape (nbins+2,)

    :returns: edges, A sequence of arrays describing the edges of bins along
        each dimension.
    :rtype edges: A list() containing mdim :class:`numpy.ndarray`s of shape
        (nbins_per_dim+1,)

    """
    edges = list()
    for points_dim in points:
        edges.append((points_dim[1:]+points_dim[:-1])/2)
    return edges

def points_from_edges(edges):
    """
    Given a sequence of arrays describing the edges of bins formed by voronoi
    cells along each dimensions returns the voronoi points that would generate
    these cells.

    ..todo:: This method only creates points in the center of the bins. Needs
        Needs to be adjusted so that it creates points in the voronoi cell
        locations. 

    :param edges: A sequence of arrays describing the edges of bins along each
        dimension.
    :type edges: A list() containing mdim :class:`numpy.ndarray`s of shape
        (nbins_per_dim+1,)

    :returns: points, the coordindates of voronoi points that would generate
        these bins
    :rtype: :class:`numpy.ndarray` of shape (num_points, dim) where num_points
        = product(nbins_per_dim)

    """
    #TODO: Implement me.
    # create a point inside each of the bins defined by the edges
    centers = list()
    for e in edges:
        centers.append((e[1:]+e[:-1])/2)

def histogramdd_volumes(edges, points):
    """
    Given a sequence of arrays describing the edges of voronoi cells (bins)
    along each dimension and an 'ij' ordered sequence of points (1 per voronoi
    cell) returns a list of the volumes associated with these voronoic cells.

    :param edges: A sequence of arrays describing the edges of bins along
        each dimension.
    :type edges: A list() containing mdim :class:`numpy.ndarray`s of shape
        (nbins_per_dim+1,)
    :param points: points used to define the voronoi tesselation (only the
        points that define regions of finite volumes)
    :type points: :class:`numpy.ndarrray` of shape (num_points, mdim)

    :rtype: tuple of (H, volume, edges)
    :returns: H is the result of :meth:`np.histogramdd(points, edges,
        normed=True)`, volumes is a :class:`numpy.ndarray` of shape (len(points),)
        continaing the  finite volumes associated with ``points``

    """
    # adjust edges
    points_max = np.max(points, 0)
    points_min = np.min(points, 0)
    # TODO replace with logical arrays to get rid of for loop.
    for dim, e in enumerate(edges):
        if len(edges) == 1:
            if e[0] >= points_min:
                e[0] = points_min-np.finfo(float).eps
            if e[-1] <= points_max:
                e[-1] = points_max+np.finfo(float).eps
        else:
            if e[0] >= points_min[dim]:
                e[0] = points_min[dim]-np.finfo(float).eps
            if e[-1] <= points_max[dim]:
                e[-1] = points_max[dim]+np.finfo(float).eps

    H, _ = np.histogramdd(points, edges, normed=True)
    volume = 1.0/(H*points.shape[0]) # account for number of bins
    # works as long as points are created with 'ij' indexing in meshgrid
    volume = volume.ravel()
    return H, volume, edges

def simple_fun_uniform(points, volumes, rect_domain):
    """
    Given a set of points, the volumes associated with these points, and
    ``rect_domain`` creates a simple function approximation of a uniform
    distribution over the hyperrectangle defined by ``rect_domain``.

    :param points: points used to define the voronoi tesselation (only the
        points that define regions of finite volumes)
    :type points: :class:`numpy.ndarrray` of shape (num_points, mdim)
    :param list() volumes: finite volumes associated with ``points``
    :type points: :class:`numpy.ndarray` of shape (num_points,)
    :param rect_domain: minima and maxima of each dimension defining the
        hyperrectangle of uniform probability
    :type rect_domain: :class:`numpy.ndarray` of shape (mdim, 2)

    :rtype: tuple
    :returns: (rho_D_M, points, d_Tree) where ``rho_D_M`` and
        ``points`` are (mdim, M) :class:`~numpy.ndarray` and
        `d_Tree` is the :class:`~scipy.spatial.KDTree` for points

    """
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=1)
    # TODOthe lines below could be done in one or two lines.
    rect_left = np.repeat([rect_domain[:, 0]], points.shape[0], 0)
    rect_right = np.repeat([rect_domain[:,1]], points.shape[0], 0)
    rect_left = np.all(np.greater_equal(points, rect_left), axis=1)
    rect_right = np.all(np.less_equal(points, rect_right), axis=1)
    inside = np.logical_and(rect_left, rect_right)
    rho_D_M = np.zeros(volumes.shape)
    rho_D_M[inside] = volumes[inside]/np.sum(volumes[inside]) # normalize on Lambda not D

    d_Tree = spatial.KDTree(points)
    return (rho_D_M, points, d_Tree)

