"""
This module provides methods for determing the exact volume of a region of
interest in the parameter space implicitly defined by a ``rect_domain`` in the
data space. Both spaces must be two dimensional. The mapping between the data
space and the parameter space is defined by the linear interpolation of
(samples, data).
"""

import matplotlib.tri as tri
import numpy as np
import collections
import bet.util as util
from scipy.interpolate import griddata

def triangulation_area(triangulation):
    """
    Determine the area enclosed in a triangulation.

    :param triangulation: triangulation for which to calculate the volume
    :type triangulation: :class:`matplotlib.tri.Triangulation`

    :rtype: tuple of (float, array_like of shape (ntriangles,))
    :returns: total area of the triangulation and the areas of each of the
        triangles
    """

    volumes = np.empty((triangulation.triangles.shape[0],))
    points = np.column_stack((triangulation.x, triangulation.y))

    for tri_num, ttri in enumerate(triangulation.triangles):
        volumes[tri_num] = .5*np.linalg.norm(np.cross(points[ttri[0]]-points[ttri[1]],
            points[ttri[0]]-points[ttri[2]]))
        if np.isnan(volumes[tri_num]):
            print points[ttri[0]]-points[ttri[1]], points[ttri[0]]-points[ttri[2]]
            print np.cross(points[ttri[0]]-points[ttri[1]],
                points[ttri[0]]-points[ttri[2]])
            print volumes[tri_num]
    return np.sum(volumes), volumes

def find_interections(triangulation, rect_domain):
    """
    Determine the intersections of the ``rect_domain`` with the
    ``triangulation``. These objects must be both in the same space (i.e. data
    space). 

    :param triangulation: triangulation for which to calculate the
        intersections with ``rect_domain``
    :type triangulation: :class:`matplotlib.tri.Triangulation`
    :param rect_domain: rectangular domain to intersect with the triangulation
    :type rect_domain: :class:`np.ndarray` of shape (mdim, 2)
    :rtype: tuple of (:class:`np.ndarray` of shape (N, mdim) where N is the
        number of points, :class:`np.ndarray` of shaple (N+4, mdim))
    :returns: (int_points, rect_and_int_points) The coordinates of the
        intersections of the triangulation with the rectangular domain and the
        coordinates of the points of the triangulaton inside the rectangular
        domain. Also the intersection points and the points that form the
        rectangular domain.
    """
    from shapely.geometry import Polygon
    from shapely.geometry import LinearRing
    from shapely.geometry import Point 
    rect_domain_meshgrid = util.meshgrid_ndim(rect_domain)
    rect_poly = Polygon(rect_domain_meshgrid).convex_hull

    int_points = set()

    for triangle in triangulation.triangles:
        tri_poly = Polygon(zip(triangulation.x[triangle],
            triangulation.y[triangle]))
        for point in list(tri_poly.exterior.coords):
            if rect_poly.contains(Point(point)):
                int_points.add(point)
        else:
            tring = LinearRing(tri_poly.exterior)
            intersection = tring.intersection(LinearRing(rect_poly.exterior))
            if intersection.type == 'Point':
                int_points.add((intersection.x, intersection.y))
            elif isinstance(intersection, collections.Iterable):
                for int_obj in intersection:
                    if int_obj.type == 'Point':
                        int_points.add((int_obj.x, int_obj.y))

    int_points = np.array(list(int_points))
    return int_points, np.vstack((int_points, rect_domain_meshgrid))

def determine_RoI_volumes(samples, data, rect_domain, param_domain=None,
        show=False):
    r"""
    Determine the exact volume of a region of interest in the parameter space
    implicitly defined by a ``rect_domain`` in the data space. Both spaces must
    be two dimensional. The mapping between the data space and the parameter
    space is defined by the linear interpolation of (samples, data).

    ..note:: 
        In the event that ``rect_domain`` is outside the original convex hull
        of data the ``rect_domain`` is trimmed to fit inside the convex hull.

    :param samples: Samples from the parameter space
    :type samples: :class:`np.ndarray` of shape (N, 2)
    :param data: Corresponding data from the data space
    :type data: :class:`np.ndarray` of shape (N, 2)
    :param rect_domain: rectangular domain in the data space that implicitly
        defines a region of interest in the parameter space, :math:`R \subset
        \mathcal{D}`
    :param param_domain: rectangular bounds that describe the parameter domain
    :type param_domain: :class:`np.ndarray` of shape (ndim, 2)
    :type rect_domain: :class:`np.ndarray` of shape (mdim, 2)
    :rtype: tuple of floats 
    :returns: (volume of :math:`R`, volume of :math:`Q^{-1}(R)`, :math
    """
    # Determine triangulations of the samples and the data
    samples_tri = tri.Triangulation(samples[:, 0], samples[:, 1])
    data_tri = tri.Triangulation(data[:, 0], data[:, 1],
            samples_tri.triangles)
    # Add the points of the data_tri that are interior/intersect the rect_domain
    _, rect_and_int_pts = find_interections(data_tri, rect_domain)
    rect_tri = tri.Triangulation(rect_and_int_pts[:, 0], rect_and_int_pts[:, 1])

    if show:
        bin_size = rect_domain[:,1]-rect_domain[:,0]
        print np.prod(bin_size)
        a, _ = triangulation_area(rect_tri)
        print a
        import matplotlib.pyplot as plt
        plt.figure()
        plt.triplot(data_tri, 'ro-')
        plt.triplot(rect_tri, 'bo-')

    # Interpolate the fine the corresponding sample values
    rect_samples = np.empty(rect_and_int_pts.shape)
    rect_samples[:, 0] = griddata(data, samples[:, 0],
            rect_and_int_pts)
    rect_samples[:, 1] = griddata(data, samples[:, 1],
            rect_and_int_pts)
    # check for NaN (this happens if rect_domain contains points oustide the
    # original convexhull formed by data)
    not_nan_ind = np.logical_not(np.any(np.isnan(rect_samples), axis=1))
    print np.sum(np.logical_not(not_nan_ind))

    rect_tri = tri.Triangulation(rect_and_int_pts[not_nan_ind, 0],
            rect_and_int_pts[not_nan_ind, 1])
    rect_samples_tri = tri.Triangulation(rect_samples[not_nan_ind, 0],
            rect_samples[not_nan_ind, 1],
            rect_tri.triangles)

    if show:
        plt.triplot(rect_tri, 'go-')
        plt.show()

    # Calculate volumes
    data_rvol, _ = triangulation_area(rect_tri)
    if show:
        print data_rvol
    samples_rvol, _ = triangulation_area(rect_samples_tri)
    # Calculate relative volume
    if type(param_domain) != type(None):
        param_vol = np.prod(param_domain[:, 1]-param_domain[:, 0])
        r_samples_rvol = samples_rvol/param_vol
    else:
        r_samples_rvol = None
    return data_rvol, samples_rvol, r_samples_rvol







