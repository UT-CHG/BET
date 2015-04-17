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
from test.test_calculateP.test_simpleFunP import prob_uniform

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
        self.r_size = self.sur_domain[:,1]*.2
        super(center_and_layer1_points_binsize_list, self).setUp()

class center_and_layer1_points_binsize_double(center_and_layer1_points_binsize):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.center_and_layer1_points_binsize` when
    r_size is a double.
    """
    def setUp(self):
        self.r_size = self.sur_domain[0,1]*.2
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


class edges(object):
    """
    Provides a method to test that the dimensions of the output (and the output
    itself) from methods with the pattern
    ``bet.calculateP.voronoiHistogram.edges_*`` are correct.
    """
    def test_dimensions(self):
        """
        Compares the dimensions of the output from
        ``bet.calculateP.voronoiHistogram.edges_*``.
        """
        compare_dim = list()
        for edge, center_ppe in zip(self.rect_and_sur_edges,
                self.center_pts_per_edge):
            compare_dim.append(len(edge) == center_ppe+3)
        assert np.all(compare_dim)
        
    def test_output(self):
        """
        Compares the output from ``bet.calcuateP.voronoiHistogram.edges_*``
        with a known solution
        """
        compare_dim = list()
        for edge, edgeVH in zip(self.my_edges, self.rect_and_sur_edges):
            compare_dim.append(np.allclose(edge, edgeVH))
        assert np.all(compare_dim)


class edges_regular(edges):
    """
    Test :meth:`bet.calculateP.voronoiHistogram.edges_regular`
    """
    def create_output(self):
        sur_width = self.sur_domain[:,1]-self.sur_domain[:,0]
        rect_width = self.r_ratio*sur_width
        rect_domain = np.empty(self.sur_domain.shape)
        rect_domain[:, 0] = self.center - .5*rect_width
        rect_domain[:, 1] = self.center + .5*rect_width
        self.my_edges = list()
        for dim in xrange(self.sur_domain.shape[0]):
            int_l1 = np.linspace(rect_domain[dim, 0], rect_domain[dim, 1],
                    self.center_pts_per_edge[dim]+1)
            int_l2 = np.empty((int_l1.shape[0]+2,))
            int_l2[1:-1] = int_l1
            int_l2[0] = self.sur_domain[dim, 0]
            int_l2[-1] = self.sur_domain[dim, 1]
            self.my_edges.append(int_l2)
    
    def setUp(self):
        self.create_output()
        self.rect_and_sur_edges = vHist.edges_regular(self.center_pts_per_edge, self.center,
                self.r_ratio, self.sur_domain)


class edges_regular_binsize(edges_regular):
    """
    Test :meth:`bet.calculateP.voronoiHistogram.edges_regular_binsize`
    """
    def setUp(self):
        self.r_ratio = self.r_size/self.sur_domain[:,1]
        super(edges_regular_binsize, self).create_output()
        self.rect_and_sur_edges = vHist.edges_regular_binsize(self.center_pts_per_edge,
                self.center, self.r_size, self.sur_domain)


class edges_regular_double(edges_regular):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` when
    r_ratio is a double.
    """
    def setUp(self):
        self.r_ratio = 0.2
        super(edges_regular_double, self).setUp()

class edges_regular_list(edges_regular):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` when
    r_ratio is a list.
    """
    def setUp(self):
        self.r_ratio = 0.2*np.ones(self.mdim)
        super(edges_regular_list, self).setUp()

class edges_regular_binsize_list(edges_regular_binsize):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.edges_regular_binsize` when
    r_size is a list.
    """
    def setUp(self):
        self.r_size = self.sur_domain[:,1]*.2
        super(edges_regular_binsize_list, self).setUp()

class edges_regular_binsize_double(edges_regular_binsize):
    """
    Provides set up for
    :meth:`bet.calculateP.voronoiHistogram.edges_regular_binsize` when
    r_size is a double.
    """
    def setUp(self):
        self.r_size = self.sur_domain[0,1]*.2
        super(edges_regular_binsize_double, self).setUp()

class test_er_list_1D(domain_1D, edges_regular_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 1D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_er_list_1D, self).createDomain()
        super(test_er_list_1D, self).setUp()
class test_er_list_2D(domain_2D, edges_regular_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 2D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_er_list_2D, self).createDomain()
        super(test_er_list_2D, self).setUp()
class test_er_list_3D(domain_3D, edges_regular_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 3D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_er_list_3D, self).createDomain()
        super(test_er_list_3D, self).setUp()

class test_er_double_1D(domain_1D, edges_regular_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 1D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_er_double_1D, self).createDomain()
        super(test_er_double_1D, self).setUp()
class test_er_double_2D(domain_2D, edges_regular_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 2D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_er_double_2D, self).createDomain()
        super(test_er_double_2D, self).setUp()
class test_er_double_3D(domain_3D, edges_regular_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 3D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_er_double_3D, self).createDomain()
        super(test_er_double_3D, self).setUp()


class test_ers_list_1D(domain_1D, edges_regular_binsize_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 1D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_ers_list_1D, self).createDomain()
        super(test_ers_list_1D, self).setUp()
class test_ers_list_2D(domain_2D, edges_regular_binsize_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 2D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_ers_list_2D, self).createDomain()
        super(test_ers_list_2D, self).setUp()
class test_ers_list_3D(domain_3D, edges_regular_binsize_list):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 3D
    domain with r_ratio as a list.
    """
    def setUp(self):
        super(test_ers_list_3D, self).createDomain()
        super(test_ers_list_3D, self).setUp()

class test_ers_double_1D(domain_1D, edges_regular_binsize_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 1D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_ers_double_1D, self).createDomain()
        super(test_ers_double_1D, self).setUp()
class test_ers_double_2D(domain_2D, edges_regular_binsize_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 2D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_ers_double_2D, self).createDomain()
        super(test_ers_double_2D, self).setUp()
class test_ers_double_3D(domain_3D, edges_regular_binsize_double):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_regular` for a 3D
    domain with r_ratio as a double.
    """
    def setUp(self):
        super(test_ers_double_3D, self).createDomain()
        super(test_ers_double_3D, self).setUp()

class edges_from_points(edges):
    """
    Test :meth:`bet.calculateP.edges_from_points`
    """
    def setUp(self):
        """
        Set up problem.
        """
        points = list()
        self.my_edges = list()
        for dim in xrange(self.mdim):
            points_dim = np.linspace(self.sur_domain[dim, 0],
                self.sur_domain[dim, 1], 4)
            points.append(points_dim)
            self.my_edges.append((points_dim[1:]+points_dim[:-1])/2)
        self.rect_and_sur_edges = vHist.edges_from_points(points)

    def test_dimensions(self):
        """
        Test dimensions of :meth:`bet.calculateP.edges_from_points`
        """
        compare_dim = list()
        for edge, my_edge in zip(self.rect_and_sur_edges,
                self.my_edges):
            compare_dim.append(len(edge) == len(my_edge))
        assert np.all(compare_dim)
 
class test_efp_1D(domain_1D, edges_from_points):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_from_points` for a 1D
    domain.
    """
    def setUp(self):
        super(test_efp_1D, self).createDomain()
        super(test_efp_1D, self).setUp()
class test_efp_2D(domain_2D, edges_from_points):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_from_points` for a 2D
    domain.
    """
    def setUp(self):
        super(test_efp_2D, self).createDomain()
        super(test_efp_2D, self).setUp()
class test_efp_3D(domain_3D, edges_from_points):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.edges_from_points` for a 3D
    domain.
    """
    def setUp(self):
        super(test_efp_3D, self).createDomain()
        super(test_efp_3D, self).setUp()

class histogramdd_volumes(object):
    """
    Test :meth:`bet.calculateP.voronoiHistogram.histogramdd_volumes`
    """
    def setUp(self):
        points = list()
        self.edges = list()
        for dim in xrange(self.mdim):
            points_dim = np.linspace(self.sur_domain[dim, 0],
                self.sur_domain[dim, 1], 4)
            points.append(points_dim[1:-1])
            self.edges.append((points_dim[1:]+points_dim[:-1])/2.0)
        self.points = util.meshgrid_ndim(points)
        self.H, _ = np.histogramdd(self.points, self.edges, normed=True)
        volume = 1.0/(self.H*(2.0**self.mdim))
        self.volume = volume.ravel()
        output = vHist.histogramdd_volumes(self.edges, self.points)
        self.o_H, self.o_volume, self.o_edges = output

    def test_dimensions_H(self):
        """
        Test the dimensions of H from
        :meth:`bet.calculateP.histogramdd_volumes``
        """
        assert self.H.shape == self.o_H.shape

    def test_dimensions_volume(self):
        """
        Test the dimensions of volume from
        :meth:`bet.calculateP.histogramdd_volumes``
        """
        assert self.volume.shape == self.o_volume.shape

    def test_dimensions_edges(self):
        """
        Test the dimensions of edges from
        :meth:`bet.calculateP.histogramdd_volumes``
        """
        compare_dim = list()
        for edge, my_edge in zip(self.o_edges, self.edges):
            compare_dim.append(len(edge) == len(my_edge))
        assert np.all(compare_dim)

    def test_H_nonnegative(self):
        """
        Test that H from :meth:`bet.calculateP.histogramdd_volumes``
        is nonnegative.
        """
        assert np.all(np.less(0.0, self.o_H))

    def test_volumes_nonnegative(self):
        """
        Test that volume from :meth:`bet.calculateP.histogramdd_volumes``
        is nonnegative.
        """
        assert np.all(np.less(0.0, self.o_volume))

    def test_H(self):
        """
        Test that H from :meth:`bet.calculateP.histogramdd_volumes``
        is correct.
        """
        assert np.allclose(self.H, self.o_H)

    def test_volume(self):
        """
        Test that volume from :meth:`bet.calculateP.histogramdd_volumes``
        is correct.
        """
        assert np.allclose(self.volume, self.o_volume)

    def test_edges(self):
        """
        Test that the edges from :meth:`bet.calculateP.histogramdd_volumes``
        are correct.
        """
        compare_dim = list()
        for edge, my_edge in zip(self.o_edges, self.edges):
            compare_dim.append(np.allclose(edge, my_edge))
        assert np.all(compare_dim)

class test_hddv_1D(domain_1D, histogramdd_volumes):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.histogramdd_volumes` for a 1D
    domain.
    """
    def setUp(self):
        super(test_hddv_1D, self).createDomain()
        super(test_hddv_1D, self).setUp()
class test_hddv_2D(domain_2D, histogramdd_volumes):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.histogramdd_volumes` for a 2D
    domain.
    """
    def setUp(self):
        super(test_hddv_2D, self).createDomain()
        super(test_hddv_2D, self).setUp()
class test_hddv_3D(domain_3D, histogramdd_volumes):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.histogramdd_volumes` for a 3D
    domain.
    """
    def setUp(self):
        super(test_hddv_3D, self).createDomain()
        super(test_hddv_3D, self).setUp()

class simple_fun_uniform(prob_uniform):
    """
    Test :meth:'bet.calculateP.voronoiHistogram.simple_fun_uniform`
    """
    def setUp(self):
        """
        Set up the problem
        """
        points = list()
        edges = list()
        self.rect_domain = np.empty((self.mdim, 2))
        for dim in xrange(self.mdim):
            points_dim = np.linspace(self.sur_domain[dim, 0],
                self.sur_domain[dim, 1], 4)
            points.append(points_dim[1:-1])
            edge = (points_dim[1:]+points_dim[:-1])/2.0
            edges.append(edge)
            self.rect_domain[dim, :] = edge[[0, -1]]
        points = util.meshgrid_ndim(points)
        H, _ = np.histogramdd(points, edges, normed=True)
        volume = 1.0/(H*(2.0**self.mdim))
        volumes = volume.ravel()
        output = vHist.simple_fun_uniform(points, volumes, self.rect_domain)
        self.rho_D_M, self.d_distr_samples, self.d_Tree = output

class test_sfu_1D(domain_1D, simple_fun_uniform):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.simple_fun_uniform` for a 1D
    domain.
    """
    def setUp(self):
        super(test_sfu_1D, self).createDomain()
        super(test_sfu_1D, self).setUp()
class test_sfu_2D(domain_2D, simple_fun_uniform):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.simple_fun_uniform` for a 2D
    domain.
    """
    def setUp(self):
        super(test_sfu_2D, self).createDomain()
        super(test_sfu_2D, self).setUp()
class test_sfu_3D(domain_3D, simple_fun_uniform):
    """
    Test
    :meth:`bet.calculateP.voronoiHistogram.simple_fun_uniform` for a 3D
    domain.
    """
    def setUp(self):
        super(test_sfu_3D, self).createDomain()
        super(test_sfu_3D, self).setUp()
