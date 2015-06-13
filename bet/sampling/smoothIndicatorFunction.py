# Copyright (C) 2014-2015 The BET Development Team
# Lindley Graham 5/25/15

"""

Create an indicator function where the value is 0.0 inside the domain defined by
a set of points and 1.0 at some outer boundary also defined by a set of
points. This means that locating the RoI is a MINIMIZATION or root finding
problem.

"""

import numpy as np
import bet.util as util
from scipy.interpolate import griddata

def smoothed_indicator_inner_outer(inner_boundary, outer_boundary):
    """
    Create an indicator function where the value is 0.0 inside the boundary
    defined by a set of points and 1.0 at some outer boundary also defined by
    a set of points.

    :param inner_boundary: points defining the inner boundary
    :type inner_boundary: :class:`np.ndarray` of shape (m, mdim)
    :param outer_boundary: points defining the outer boundary
    :type outer_boundary: :class:`np.ndarray` of shape (m, mdim)
    :rtype: function
    :returns: function where the value at ``inner_boundary`` is 0.0 and the
        value at ``outer_boundary`` is 1.0 with values interpolated between the
        boundaries
    """
    points = np.concatenate((inner_boundary, outer_boundary))
    values = np.concatenate((np.zeros(inner_boundary.shape[0]),
        np.ones(outer_boundary.shape[0])))
    def indicator_function(inputs):
        "Function wrapper for griddata"
        return griddata(points, values, inputs, fill_value=2.0)
    return indicator_function

def smoothed_indicator_cw_outer(center, width, outer_boundary):
    """
    Create an indicator function where the value is 0.0 inside the boundary
    defined by a set of points and 1.0 at some outer boundary also defined by
    a set of points.

    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param width: location of the width of the hyperrectangle
    :type width: :class:`numpy.ndarray` of shape (mdim,)
    :param outer_boundary: points defining the outer boundary
    :type outer_boundary: :class:`np.ndarray` of shape (m, mdim)
    :rtype: function
    :returns: function where the value at ``inner_boundary`` is 0.0 and the
        value at ``outer_boundary`` is 1.0 with values interpolated between the
        boundaries

    """
    half_width = 0.5*width
    left = center-half_width
    right = center-half_width
    inner_boundary = [[l, r] for l, r in zip(left, right)]
    inner_boundary = util.meshgrid_ndim(inner_boundary)
    return smoothed_indicator_inner_outer(inner_boundary, outer_boundary)

def smoothed_indicator_cws(center, width, sur_domain):
    """
    Create an indicator function where the value is 0.0 inside the boundary
    defined by a set of points and 1.0 at some outer boundary also defined by
    a set of points.

    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param width: location of the width of the hyperrectangle
    :type width: :class:`numpy.ndarray` of shape (mdim,)
    :param sur_domain: minima and maxima of each dimension defining the
        surrounding domain. The surrounding domain is the bounded
        domain in the data space (i.e. the data domain).
    :type sur_domain: :class:`numpy.ndarray` of shape (mdim, 2)
    :rtype: function
    :returns: function where the value at ``inner_boundary`` is 0.0 and the
        value at ``outer_boundary`` is 1.0 with values interpolated between the
        boundaries
    """
    outer_boundary = [[l, r] for l, r in zip(sur_domain[:, 0], 
        sur_domain[:, 1])]
    outer_boundary = util.meshgrid_ndim(outer_boundary)
    return smoothed_indicator_cw_outer(center, width, outer_boundary)

def smoothed_indicator_boundary_inner_outer(inner_boundary, middle_boundary,
        outer_boundary):
    """
    Create an indicator function where the value is 0.3 inside the boundary
    defined by a set of points, 0.0 at the padded middle boundaries, and 1.0
    at some outer boundary also defined by a set of points.

    :param inner_boundary: points defining the inner boundary
    :type inner_boundary: :class:`np.ndarray` of shape (m, mdim)
    :param outer_boundary: points defining the outer boundary
    :type outer_boundary: :class:`np.ndarray` of shape (m, mdim)
    :rtype: function
    :returns: function where the value at ``inner_boundary`` is 0.3 , 0.0
        between the midinner and midouter boudnaries, and the value at
        ``outer_boundary`` is 1.0 with values interpolated between the
        boundaries
    """
    points = np.concatenate((inner_boundary, middle_boundary,
         outer_boundary))
    values = np.concatenate((0.3*np.ones(inner_boundary.shape[0]),
        np.zeros(middle_boundary.shape[0]),
        np.ones(outer_boundary.shape[0])))
    def indicator_function(inputs):
        "Function wrapper for griddata"
        return griddata(points, values, inputs)
    return indicator_function

def smoothed_indicator_boundary_cw_outer(center, width, mid_width,
        outer_boundary): 
    """
    Create an indicator function where the value is 0.3 inside the boundary
    defined by a set of points, 0.0 at the padded middle boundaries, and 1.0
    at some outer boundary also defined by a set of points.

    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param width: location of the width of the hyperrectangle
    :type width: :class:`numpy.ndarray` of shape (mdim,)
    :param outer_boundary: points defining the outer boundary
    :type outer_boundary: :class:`np.ndarray` of shape (m, mdim)
    :rtype: function
    :returns: function where the value at ``inner_boundary`` is 0.3 , 0.0
        between the midinner and midouter boudnaries, and the value at
        ``outer_boundary`` is 1.0 with values interpolated between the
        boundaries

    """
    half_width = 0.5*width
    left = center-half_width
    right = center-half_width
    inner_boundary = [[l+1.5*mid_width, r-1.5*mid_width] for l, r in zip(left, right)]
    midinner_boundary = [[l+.5*mid_width, r-.5*mid_width] for l, r in zip(left, right)]
    midouter_boundary = [[l-.5*mid_width, r+.5*mid_width] for l, r in zip(left, right)]
    inner_boundary = util.meshgrid_ndim(inner_boundary)
    midinner_boundary = util.meshgrid_ndim(midinner_boundary)
    midouter_boundary = util.meshgrid_ndim(midouter_boundary)
    middle_boundary = np.concatenate((midinner_boundary, midouter_boundary))
    return smoothed_indicator_boundary_inner_outer(inner_boundary,
            middle_boundary, outer_boundary)

def smoothed_indicator_boundary_cws(center, width, mid_width, sur_domain):
    """
    Create an indicator function where the value is 0.3 inside the boundary
    defined by a set of points, 0.0 at the padded middle boundaries, and 1.0
    at some outer boundary also defined by a set of points.

    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param width: location of the width of the hyperrectangle
    :type width: :class:`numpy.ndarray` of shape (mdim,)
    :param sur_domain: minima and maxima of each dimension defining the
        surrounding domain. The surrounding domain is the bounded
        domain in the data space (i.e. the data domain).
    :type sur_domain: :class:`numpy.ndarray` of shape (mdim, 2)
    :rtype: function
    :returns: function where the value at ``inner_boundary`` is 0.3 , 0.0
        between the midinner and midouter boudnaries, and the value at
        ``outer_boundary`` is 1.0 with values interpolated between the
        boundaries
    """
    outer_boundary = [[l, r] for l, r in zip(sur_domain[:, 0], 
        sur_domain[:, 1])]
    outer_boundary = util.meshgrid_ndim(outer_boundary)
    return smoothed_indicator_boundary_cw_outer(center, width, mid_width, 
            outer_boundary)
