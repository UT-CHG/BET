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
    rect_domain_meshgrid = util.meshgrid_ndim(rect_domain)
    rect_poly = Polygon(rect_domain_meshgrid).convex_hull

    int_points = set()

    for triangle in triangulation.triangles:
        tri_poly = Polygon(zip(triangulation.x[triangle],
            triangulation.y[triangle]))
        if rect_poly.contains(tri_poly):
            for point in list(tri_poly.exterior.coords):
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

def determine_RoI_volumes(samples, data, rect_domain, param_domain=None):
    r"""
    Determine the exact volume of a region of interest in the parameter space
    implicitly defined by a ``rect_domain`` in the data space. Both spaces must
    be two dimensional. The mapping between the data space and the parameter
    space is defined by the linear interpolation of (samples, data).

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
    samples_tri = tri.triangulation(samples[:, 0], samples[:, 1])
    data_tri = tri.triangulation(data[:, 0], data[:, 1],
            samples_tri.triangles)
    # Add the points of the data_tri that are interior/intersect the rect_domain
    _, rect_and_int_pts = find_interections(data_tri, rect_domain)
    rect_tri = tri.triangulation(rect_and_int_pts[:, 0], rect_and_int_pts[:, 1])
    # Interpolate the fine the corresponding sample values
    rect_samples = np.empty(rect_and_int_pts.shape)
    rect_samples[:, 0] = griddata(data, samples[:, 0],
            rect_and_int_pts)
    rect_samples[:, 1] = griddata(data, samples[:, 1],
            rect_and_int_pts)
    rect_samples_tri = tri.triangulation(rect_samples[:, 0], rect_samples[:, 1],
            rect_tri)
    # Calculate volumes
    data_rvol = triangulation_area(rect_tri)
    samples_rvol = triangulation_area(rect_samples_tri)
    # Calculate relative volume
    if param_domain:
        param_vol = np.prod(param_domain[:, 1]-param_domain[:, 0])
        r_samples_rvol = samples_rvol/param_vol
    else:
        r_samples_rvol = None
    return data_rvol, samples_rvol, r_samples_rvol







