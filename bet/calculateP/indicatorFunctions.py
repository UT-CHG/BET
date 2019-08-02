# Copyright (C) 2015-2019 The BET Development Team

r"""
This module provides various indicator functions, :math:`\mathbf{1}_A` for
various sets :math:`A \subset \mathbb{R}^n` where given a set of points in
:math:`\{x_i\}_{i=0}^{N} \in \mathbf{R}^n` returns :math:`\{ \mathbf{1}_A(x_i)
\}_{i=0}^{N}`.
"""

# import necessary modules
import numpy as np


def hyperrectangle(left, right):
    r"""
    Pointwise indicator function for a hyperrectangle defined by a leftmost and
    rightmost corner.

    :param left: Leftmost(minimum) corner of the hyperrectangle.
    :type left: :class:`np.ndarray` of shape (ndim,)
    :param right: Rightmost(maximum) corner of the hyperrectangle.
    :type right: :class:`np.ndarray` of shape (ndim,)

    :rtype: function
    :returns: :math:`\mathbf{1}_A`
    """
    def ifun(points):
        r"""
        :param points: set of points in :math:`\{x_i\}_{i=0}^{N} \in
            \mathbf{R}^n`
        :type points: :class:`np.ndarray` of shape (N, ndim)

        :rtype: boolean :class:`np.ndarray` of shape (ndim,)
        :returns: :math:`\{ \mathbf{1}_A(x_i) \}_{i=0}^{N}`
        """
        return np.logical_and(np.all(np.logical_or(np.greater_equal(points,
                                                                    left), np.isclose(points, left)), axis=1),
                              np.all(np.logical_or(np.less_equal(points,
                                                                 right), np.isclose(points, right)), axis=1))
    return ifun


def hyperrectangle_size(center, width):
    r"""
    Pointwise indicator function for a hyperrectangle defined by a center point
    and the width of the hyperrectangle.

    :param left: center of the hyperrectangle.
    :type left: :class:`np.ndarray` of shape (ndim,)
    :param width: length of the size of the sides of the hyperrectangle.
    :type width: :class:`np.ndarray` of shape (ndim,)

    :rtype: function
    :returns: :math:`\mathbf{1}_A`
    """
    left = center - .5 * width
    right = center + .5 * width
    return hyperrectangle(left, right)


def boundary_hyperrectangle(left, right, boundary_width):
    r"""
    Pointwise indicator function for the set of points within
    ``boundary_width`` of the boundary of a hyperrectangle defined by a
    leftmost and rightmost corner.

    :param left: Leftmost(minimum) corner of the hyperrectangle.
    :type left: :class:`np.ndarray` of shape (ndim,)
    :param right: Rightmost(maximum) corner of the hyperrectangle.
    :type right: :class:`np.ndarray` of shape (ndim,)
    :param boundary_width: Width of the boundary
    :type boundary_width: :class:`np.ndarray` of shape (ndim,)

    :rtype: function
    :returns: :math:`\mathbf{1}_{\partial A \plusminus \epsilon}`

    """
    inner = hyperrectangle(
        left + .5 * boundary_width,
        right - .5 * boundary_width)
    outer = hyperrectangle(
        left - .5 * boundary_width,
        right + .5 * boundary_width)

    def ifun(points):
        r"""
        :param points: set of points in :math:`\{x_i\}_{i=0}^{N} \in
            \mathbf{R}^n`
        :type points: :class:`np.ndarray` of shape (N, ndim)

        :rtype: boolean :class:`np.ndarray` of shape (ndim,)
        :returns: :math:`\{ \mathbf{1}_{\partial A \plusminus \epsilon}(x_i)
            \}_{i=0}^{N}`

        """
        return np.logical_and(outer(points), np.logical_not(inner(points)))

    return ifun


def boundary_hyperrectangle_ratio(left, right, boundary_ratio):
    r"""
    Pointwise indicator function for the set of points within
    ``boundary_ratio*hyperrectanlge_width`` of the boundary of a hyperrectangle
    defined by a leftmost and rightmost corner.

    :param left: Leftmost(minimum) corner of the hyperrectangle.
    :type left: :class:`np.ndarray` of shape (ndim,)
    :param right: Rightmost(maximum) corner of the hyperrectangle.
    :type right: :class:`np.ndarray` of shape (ndim,)
    :param boundary_ratio: Ratio of the width of the boundary
    :type boundary_ratio: :class:`np.ndarray` of shape (ndim,)

    :rtype: function
    :returns: :math:`\mathbf{1}_{\partial A \plusminus \epsilon}`

    """
    width = right - left
    boundary_width = width * boundary_ratio
    return boundary_hyperrectangle(left, right, boundary_width)


def boundary_hyperrectangle_size(center, width, boundary_width):
    r"""
    Pointwise indicator function for the set of points within
    ``boundary_width`` of the boundary of  a hyperrectangle defined by a center
    point and the width of the hyperrectangle.

    :param left: center of the hyperrectangle.
    :type left: :class:`np.ndarray` of shape (ndim,)
    :param width: length of the size of the sides of the hyperrectangle.
    :type width: :class:`np.ndarray` of shape (ndim,)
    :param boundary_width: Width of the boundary
    :type boundary_width: :class:`np.ndarray` of shape (ndim,)

    :rtype: function
    :returns: :math:`\mathbf{1}_{\partial A \plusminus \epsilon}`

    """
    left = center - .5 * width
    right = center + .5 * width
    return boundary_hyperrectangle(left, right, boundary_width)


def boundary_hyperrectangle_size_ratio(center, width, boundary_ratio):
    r"""
    Pointwise indicator function for the set of points within
    ``boundary_ratio*hyperrectanlge_width`` of the boundary of  a
    hyperrectangle defined by a center point and the width of the
    hyperrectangle.

    :param left: center of the hyperrectangle.
    :type left: :class:`np.ndarray` of shape (ndim,)
    :param width: length of the size of the sides of the hyperrectangle.
    :type width: :class:`np.ndarray` of shape (ndim,)
    :param boundary_ratio: Ratio of the width of the boundary
    :type boundary_ratio: :class:`np.ndarray` of shape (ndim,)

    :rtype: function
    :returns: :math:`\mathbf{1}_{\partial A \plusminus \epsilon}`

    """
    return boundary_hyperrectangle_size(center, width, width * boundary_ratio)


def hypersphere(center, radius):
    r"""
    Pointwise indicator function for a hypersphere defined by a center and a
    radius.

    If radius is a vector and not a scalar this will work for an hyperellipse.

    :param center: center of the hypersphere/ellipse
    :type center: :class:`numpy.ndarray` of shape (ndim,)
    :param radius: radius or radii of the hypereliipse
    :type radius: ``float`` or :class:`numpy.ndarray` of shape(ndim,)

    :rtype: callable
    :returns: :math:`\mathbf{1}_A` where A is a hypersphere/ellipse

    """
    def ifun(points):
        # calculate distance from the center
        dist = np.linalg.norm(center - points, ord=2, axis=1)
        return dist <= radius
    return ifun


def boundary_hypersphere(center, radius, boundary_width):
    r"""
    Pointwise indicator function for a hypersphere defined by a center and a
    radius.

    If radius is a vector and not a scalar this will work for an hyperellipse.

    :param center: center of the hypersphere/ellipse
    :type center: :class:`numpy.ndarray` of shape (ndim,)
    :param radius: radius or radii of the hypereliipse
    :type radius: ``float`` or :class:`numpy.ndarray` of shape(ndim,)
    :param boundary_width: Width of the boundary

    :rtype: callable
    :returns: :math:`\mathbf{1}_A` where A is a hypersphere/ellipse

    """
    def ifun(points):
        # calculate distance from the center
        dist = np.linalg.norm(center - points, ord=2, axis=1)
        return np.logical_and(dist <= radius + boundary_width * .5,
                              dist >= radius - boundary_width * .5)
    return ifun


def boundary_hypersphere_ratio(center, radius, boundary_ratio):
    r"""
    Pointwise indicator function for a hypersphere defined by a center and a
    radius.

    If radius is a vector and not a scalar this will work for an hyperellipse.

    :param center: center of the hypersphere/ellipse
    :type center: :class:`numpy.ndarray` of shape (ndim,)
    :param radius: radius or radii of the hypereliipse
    :type radius: ``float`` or :class:`numpy.ndarray` of shape(ndim,)
    :param boundary_ratio: Ratio of the width of the boundary

    :rtype: callable
    :returns: :math:`\mathbf{1}_A` where A is a hypersphere/ellipse

    """
    return boundary_hypersphere(center, radius, radius * boundary_ratio)
