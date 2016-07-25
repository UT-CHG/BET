# Copyright (C) 2014-2015 The BET Development Team

# -*- coding: utf-8 -*-
# Lindley Graham 10/16/2015
"""
Test methods in :mod:`bet.calculateP.indicatorFunctions`. We only test for
dimensions 1, 2, 3.
"""

import unittest
import bet.calculateP.indicatorFunctions as ifun
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
        self.radius = 5.0
        self.width = np.array([9.0])

class domain_2D(object):
    """
    Sets up 2D domain domain problem.
    """
    def createDomain(self):
        """
        Set up data.
        """
        self.center = np.array([5.0, 5.0])
        self.radius = 3.0
        self.width = np.array([11.0, 7.0])

class domain_3D(object):
    """
    Sets up 3D domain domain problem.
    """
    def createDomain(self):
        """
        Set up data.
        """
        self.center = np.array([5.0, 5.0, 5.0])
        self.domain = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
        self.radius = 2.0
        self.width = np.array([11.0, 7.0, 10.0])

class check_inside(object):
    """
    Test :mod:`bet.calculateP.indicatorFunctions`
    """
    def setUp(self):
        """
        Set up the problem by calculating required ratios and widths.
        """
        self.boundary_ratio_radius = 0.1
        self.boundary_width_radius = self.radius*self.boundary_ratio_radius
        self.boundary_width = np.ones(self.center.shape) + \
                0.1*np.arange(len(self.center))
        self.right = self.center + .5*self.width
        self.left = self.center - .5*self.width
        self.boundary_ratio = self.boundary_width/self.width
        # create a list of coordinates that are outside the domain
        outcoords_rect = []
        outcoords_sphere = []
        # create a list of coordinates that are in on the boundary of the
        # domain
        oncoords_rect = []
        dim = len(self.width)
        for l, r, bw in zip(self.left, self.right, self.boundary_width): 
            outcoords_rect.append(np.array([l-bw, r+bw]))
            outcoords_sphere.append(np.array([self.center-self.radius\
                    -self.boundary_width_radius,
                self.center+self.radius+self.boundary_width_radius]))
            oncoords_rect.append(np.array([l, r]))
        self.outcoords_rect = util.meshgrid_ndim(outcoords_rect)
        self.oncoords_rect = util.meshgrid_ndim(oncoords_rect)
        self.outcoords_sphere = util.meshgrid_ndim(outcoords_sphere)
        self.oncoords_sphere = np.row_stack((-np.eye(dim),
            np.eye(dim).transpose()))*self.radius+self.center 
        print "SPHERE", self.center, self.radius, self.oncoords_sphere

    def test_hyperrectangle(self):
        """
        Test :meth:`bet.calculateP.indicatorFunctions.hyperrectangle`
        """
        indicator = ifun.hyperrectangle(self.left, self.right)
        assert np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_rect))
    
    def test_hyperrectangle_size(self):
        """
        Test :meth:`bet.calculateP.indicatorFunctions.hyperrectangle_size`
        """
        indicator = ifun.hyperrectangle_size(self.center, self.width)
        assert np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_rect))

    def test_boundary_hyperrectangle(self):
        """
        Test :meth:`bet.calculateP.indicatorFunctions.boundary_hyperrectangle`
        """
        indicator = ifun.boundary_hyperrectangle(self.left, self.right,
                self.boundary_width)
        assert False == np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_rect))
        assert np.all(indicator(self.oncoords_rect))
    
    def test_boundary_hyperrectangle_size(self):
        """
        Test
        :meth:`bet.calculateP.indicatorFunctions.boundary_hyperrectangle_size`
        """
        indicator = ifun.boundary_hyperrectangle_size(self.center, self.width,
                self.boundary_width)
        assert False == np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_rect))
        assert np.all(indicator(self.oncoords_rect))

    def test_boundary_hyperrectangle_ratio(self):
        """
        Test
        :meth:`bet.calculateP.indicatorFunctions.boundary_hyperrectangle_ratio`
        """
        indicator = ifun.boundary_hyperrectangle_ratio(self.left, self.right,
                self.boundary_ratio)
        assert False == np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_rect))
        assert np.all(indicator(self.oncoords_rect))
    
    def test_boundary_hyperrectangle_size_ratio(self):
        """
        Test
        :meth:`bet.calculateP.indicatorFunctions.boundary_hyperrectangle_size_ratio`
        """
        indicator = ifun.boundary_hyperrectangle_size_ratio(self.center,
                self.width, self.boundary_ratio)
        assert False == np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_rect))
        assert np.all(indicator(self.oncoords_rect))

    def test_hypersphere(self):
        """
        Test :meth:`bet.calculateP.indicatorFunctions.hypersphere`
        """
        indicator = ifun.hypersphere(self.center, self.radius)
        assert np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_sphere))
    
    def test_boundary_hypersphere(self):
        """
        Test :meth:`bet.calculateP.indicatorFunctions.boundary_hypersphere`
        """
        indicator = ifun.boundary_hypersphere(self.center, self.radius,
                self.boundary_width_radius)
        assert False == np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_sphere))
        assert np.all(indicator(self.oncoords_sphere))
    
    def test_boundary_hypersphere_ratio(self):
        """
        Test
        :meth:`bet.calculateP.indicatorFunctions.boundary_hypersphere_ratio`
        """
        indicator = ifun.boundary_hypersphere_ratio(self.center, self.radius,
                self.boundary_ratio_radius)
        assert False == np.all(indicator(util.fix_dimensions_vector_2darray(self.center)))
        assert False == np.all(indicator(self.outcoords_sphere))
        assert np.all(indicator(self.oncoords_sphere))

class test_1D(domain_1D, check_inside):
    """
    Test :mod:`bet.calculateP.indicatorFunctions` for a 1D domain.
    """
    def setUp(self):
        super(test_1D, self).createDomain()
        super(test_1D, self).setUp()


class test_2D(domain_2D, check_inside):
    """
    Test :mod:`bet.calculateP.indicatorFunctions` for a 2D domain.
    """
    def setUp(self):
        super(test_2D, self).createDomain()
        super(test_2D, self).setUp()


class test_3D(domain_3D, check_inside):
    """
    Test :mod:`bet.calculateP.indicatorFunctions` for a 3D domain.
    """
    def setUp(self):
        super(test_3D, self).createDomain()
        super(test_3D, self).setUp()

