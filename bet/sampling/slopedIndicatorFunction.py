# Copyright (C) 2014-2015 The BET Development Team
# Lindley Graham 5/25/15

"""

Create an indicator function where the value is 1.0 inside the domain defined by
a set of points and 0.0 at some outer boundary also defined by a set of
points. This means locating the RoI is a MAXIMIZATION problem.

"""

import numpy as np
import bet.util as util
from scipy.interpolate import griddata

def sloped_indicator_inner_outer(inner_boundary, outer_boundary):
    """
    Create an indicator function where the value is 1.0 inside the boundary
    defined by a set of points and zero at some outer boundary also defined by
    a set of points.

    :param inner_boundary: points defining the inner boundary
    :type inner_boundary: :class:`np.ndarray` of shape (m, mdim)
    :param outer_boundary: points defining the outer boundary
    :type outer_boundary: :class:`np.ndarray` of shape (m, mdim)
    :rtype: function
    :returns: function where the value at ``inner_boundary`` is 1.0 and the
        value at ``outer_boundary`` is 0.0 with values interpolated between the
        boundaries
    """
    points = np.concatenate((inner_boundary, outer_boundary))
    values = np.concatenate((np.ones(inner_boundary.shape[0]),
        np.zeros(outer_boundary.shape[0])))
    def indicator_function(inputs):
        "Function wrapper for griddata"
        return griddata(points, values, inputs, fill_value=0.0)
    return indicator_function

def sloped_indicator_cw_outer(center, width, outer_boundary):
    """
    Create an indicator function where the value is 1.0 inside the boundary
    defined by a set of points and zero at some outer boundary also defined by
    a set of points.

    :param center: location of the center of the hyperrectangle
    :type center: :class:`numpy.ndarray` of shape (mdim,)
    :param width: location of the width of the hyperrectangle
    :type width: :class:`numpy.ndarray` of shape (mdim,)
    :param outer_boundary: points defining the outer boundary
    :type outer_boundary: :class:`np.ndarray` of shape (m, mdim)
    :rtype: function
    :returns: function where the value at ``inner_boundary`` is 1.0 and the
        value at ``outer_boundary`` is 0.0 with values interpolated between the
        boundaries

    """
    half_width = 0.5*width
    left = center-half_width
    right = center-half_width
    inner_boundary = [[l, r] for l, r in zip(left, right)]
    inner_boundary = util.meshgrid_ndim(inner_boundary)
    return sloped_indicator_inner_outer(inner_boundary, outer_boundary)

def sloped_indicator_cws(center, width, sur_domain):
    """
    Create an indicator function where the value is 1.0 inside the boundary
    defined by a set of points and zero at some outer boundary also defined by
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
    :returns: function where the value at ``inner_boundary`` is 1.0 and the
        value at ``outer_boundary`` is 0.0 with values interpolated between the
        boundaries
    """
    outer_boundary = [[l, r] for l, r in zip(sur_domain[:, 0], 
        sur_domain[:, 1])]
    outer_boundary = util.meshgrid_ndim(outer_boundary)
    return sloped_indicator_cw_outer(center, width, outer_boundary)

