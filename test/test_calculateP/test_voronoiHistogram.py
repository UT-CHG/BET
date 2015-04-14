# -*- coding: utf-8 -*-
# Lindley Graham 04/12/2015
"""
Test methods in :mod:`bet.calculateP.voronoiHistogram`. Since this module is
not meant to be directly accessed by the user we only test for dimensions 1, 2,
3. We also assume that ``center_points_per_edge`` is a list as specified in the
docString for the methods in :mod:`bet.calculateP.voronoiHistogram`. In other
words, all the dimensions of any arrays must be correct before calling these
methods.
"""

import os, unittest, collections
import bet.calculateP.voronoiHistogram as vHist
import bet.util as util
import numpy as np
import numpy.testing as nptest

# Do below for dimensions 01, 1, 2, and 3
class domain_1D(object):
    """
    Sets up 1D domain domain problem.
    """
    def createDomain(self):
        """
        Set up data.
        """
        self.center = np.array([5.0])
        self.sur_domain = np.expand_dims(np.array([0.0, 10.0]), axis=0)
        self.mdim = 1
        self.center_pts_per_edge = [1]


class domain_2D(object):
    """
    Sets up 2D domain domain problem.
    """
    def createDomain(self):
        """
        Set up data.
        """
        self.center = np.array([5.0, 5.0])
        self.sur_domain = np.array([[0.0, 10.0], [0.0, 10.0]])
        self.mdim = 2
        self.center_pts_per_edge = [1,2]


class domain_3D(object):
    """
    Sets up 3D domain domain problem.
    """
    def createDomain(self):
        """
        Set up data.
        """
        self.center = np.array([5.0, 5.0, 5.0])
        self.sur_domain = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
        self.mdim = 3
        self.center_pts_per_edge = [1,2,1]

# center_and_layer1_points_binsize
# center_and_layer1_points

# make sure dimensions are correct
# make sure the rect_domain is correct
# make sure the bounding layer is correct 
#class center_and_layer1_points(unittest.TestCase):
class center_and_layer1_points(object):
    """
    Test :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points`
    """
    def setUp(self):
        """
        Set up the problem for
        :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points`
        """
        self.create_output()
        output = vHist.center_and_layer1_points(self.center_pts_per_edge,
                self.center, self.r_ratio, self.sur_domain)
        self.points, self.interior_and_layer1_VH, self.rect_domain_VH = output

    def create_output(self):
        """
        Create output to test the output of
        :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points`
        against.

        :param r_ratio: The ratio of the length of the sides of the
            hyperrectangle to the surrounding domain
        :type r_ratio: int or list()
        """
        sur_width = self.sur_domain[:, 1] - self.sur_domain[:, 0]
        rect_width = self.r_ratio*sur_width
        self.rect_domain = np.empty(self.sur_domain.shape)
        self.rect_domain[:, 0] = self.center - .5*rect_width
        self.rect_domain[:, 1] = self.center + .5*rect_width
        if not isinstance(self.center_pts_per_edge, np.ndarray):
            self.center_pts_per_edge = np.array(self.center_pts_per_edge)
        layer1_left = self.rect_domain[:, 0]-rect_width/(2*self.center_pts_per_edge)
        layer1_right = self.rect_domain[:, 1]+rect_width/(2*self.center_pts_per_edge)
        self.interior_and_layer1 = list()
        for dim in xrange(self.mdim):
            self.interior_and_layer1.append(np.linspace(layer1_left[dim],
                layer1_right[dim], self.center_pts_per_edge[dim]+2))

    def test_dimensions(self):
        """
        Test the dimensions for :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points`
        """
        assert self.points.shape == (np.prod(self.center_pts_per_edge+2),
                self.mdim)
        assert len(self.interior_and_layer1_VH) == self.mdim
        nptest.assert_array_almost_equal(self.center_pts_per_edge+2, 
                [len(dim_layer) for dim_layer in self.interior_and_layer1_VH])
        assert self.rect_domain.shape == (self.mdim, 2)

    def test_rect_domain(self):
        """
        Test that the ``rect_domain`` is correct.
        """
        nptest.assert_array_almost_equal(self.rect_domain,
                self.rect_domain_VH)

    def test_bounding_layer(self):
        """
        Test that the interior_and_layer1 is correct.
        """
        compare_list = list()
        for mine, meth in zip(self.interior_and_layer1, self.interior_and_layer1_VH):
            compare_list.append(np.allclose(mine, meth))
        assert np.all(compare_list)        

    def test_points(self):
        """
        Test that the points are correct.
        """
        nptest.assert_array_almost_equal(self.points,
            util.meshgrid_ndim(self.interior_and_layer1))

class center_and_layer1_points_double(center_and_layer1_points):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` when
    r_ratio is a double.
    """
    def setUp(self):
        self.r_ratio = 0.2
        super(center_and_layer1_points_double, self).setUp()

class center_and_layer1_points_list(center_and_layer1_points):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` when
    r_ratio is a list.
    """
    def setUp(self):
        self.r_ratio = 0.2*np.ones(self.mdim)
        super(center_and_layer1_points_list, self).setUp()


class center_and_layer1_points_binsize(center_and_layer1_points):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points_binsize`
    """
    def setUp(self):
        """
        Set up the problem for
        :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points`
        """
        self.r_ratio = self.r_size/self.sur_domain[:,1]
        super(center_and_layer1_points_binsize, self).create_output()
        output = vHist.center_and_layer1_points_binsize(self.center_pts_per_edge,
                self.center, self.r_size, self.sur_domain)
        self.points, self.interior_and_layer1_VH, self.rect_domain_VH = output


class center_and_layer1_points_binsize_list(center_and_layer1_points_binsize):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points_binsize` when
    r_size is a list.
    """
    def setUp(self):
        self.r_size = self.sur_domain[:,1]*.02
        super(center_and_layer1_points_binsize_list, self).setUp()

class center_and_layer1_points_binsize_double(center_and_layer1_points_binsize):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points_binsize` when
    r_size is a double.
    """
    def setUp(self):
        self.r_size = self.sur_domain[0,1]*.02
        super(center_and_layer1_points_binsize_double, self).setUp()

class test_calp_list_1D(domain_1D, center_and_layer1_points_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 1D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_calp_list_1D, self).createDomain()
        super(test_calp_list_1D, self).setUp()
class test_calp_list_2D(domain_2D, center_and_layer1_points_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 2D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_calp_list_2D, self).createDomain()
        super(test_calp_list_2D, self).setUp()
class test_calp_list_3D(domain_3D, center_and_layer1_points_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 3D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_calp_list_3D, self).createDomain()
        super(test_calp_list_3D, self).setUp()

class test_calp_double_1D(domain_1D, center_and_layer1_points_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 1D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_calp_double_1D, self).createDomain()
        super(test_calp_double_1D, self).setUp()
class test_calp_double_2D(domain_2D, center_and_layer1_points_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 2D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_calp_double_2D, self).createDomain()
        super(test_calp_double_2D, self).setUp()
class test_calp_double_3D(domain_3D, center_and_layer1_points_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 3D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_calp_double_3D, self).createDomain()
        super(test_calp_double_3D, self).setUp()


class test_calps_list_1D(domain_1D, center_and_layer1_points_binsize_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 1D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_calps_list_1D, self).createDomain()
        super(test_calps_list_1D, self).setUp()
class test_calps_list_2D(domain_2D, center_and_layer1_points_binsize_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 2D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_calps_list_2D, self).createDomain()
        super(test_calps_list_2D, self).setUp()
class test_calps_list_3D(domain_3D, center_and_layer1_points_binsize_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 3D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_calps_list_3D, self).createDomain()
        super(test_calps_list_3D, self).setUp()

class test_calps_double_1D(domain_1D, center_and_layer1_points_binsize_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 1D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_calps_double_1D, self).createDomain()
        super(test_calps_double_1D, self).setUp()
class test_calps_double_2D(domain_2D, center_and_layer1_points_binsize_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 2D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_calps_double_2D, self).createDomain()
        super(test_calps_double_2D, self).setUp()
class test_calps_double_3D(domain_3D, center_and_layer1_points_binsize_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points` for a 3D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_calps_double_3D, self).createDomain()
        super(test_calps_double_3D, self).setUp()


# edges_regular_binsize
# edges_regular
# edges_from_points
# make sure dimensions are correct
# make sure int_l1 is correct
# make sure int_l2 is correct
class edges(object):
    def test_dimensions(self):
        pass

class edges_regular(edges, unittest.TestCase):
    def test_l1(self):
        pass
    def test_l2(self):
        pass



# points_from_edges
# make sure dimensions are correct
# make sure shape is correct
# make sure points are correct



# histogrammdd_volumes
# make sure dimensions are correct
# make sure H sums to 1
# make sure volume sums to 1?
# make sure volumes are nonnegative

# simple_fun_uniform
# make sure dimensions are correct
# make sure rho_D_M sums to 1
# make sure rho_D_M is nonnegative




