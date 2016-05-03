# Copyright (C) 2014-2015 The BET Development Team

# Lindley Graham 04/09/2015

"""
This module contains tests for :module:`bet.calculateP.simpleFunP`

Some of these tests make sure certain values are within a tolerance rather than
exact due to the stochastic nature of the algorithms being tested. 

The ouput of all the methods being tested is of the form (rho_D_M,
d_distr_samples, d_Tree) where ``rho_D_M`` is (M,) and ``d_distr_samples`` are
(M, mdim) :class:`~numpy.ndarray` and `d_Tree` is the
:class:`~scipy.spatial.KDTree` for d_distr_samples.

"""

import os, bet, unittest, collections   
import bet.calculateP.simpleFunP as sFun
import numpy as np
import numpy.testing as nptest

local_path = os.path.join(os.path.dirname(bet.__file__),
'../test/test_calulateP')

class prob(object):
    """
    Test that the probabilties sum to 1, are non-negative, and have the correct
    dimensions.
    """
    def test_rho_D_M_sum_to_1(self):
        """
        Test that probabilities sum to 1.
        """
        nptest.assert_almost_equal(np.sum(self.rho_D_M), 1.0)
    def test_rho_D_M_pos(self):
        """
        Test that probabilities are non-negative.
        """
        assert True == np.all(self.rho_D_M >= 0.0)
    def test_dimensions(self):
        """
        Test that the dimensions of the outputs are correct.
        """
        assert self.rho_D_M.shape[0] == self.d_distr_samples.shape[0]
        assert self.mdim == self.d_distr_samples.shape[1]
        assert (self.d_Tree.n, self.d_Tree.m) == self.d_distr_samples.shape


class prob_uniform(prob):
    """
    Test that the probabilities within the prescribed domain are non-zero and
    that the probabilities outside of the prescribed domain are zero.
    """

    def test_domain(self):
        """
        Test that the probabilities within the prescribed domain are non-zero
        and that the probabilities outside of the prescribed domain are zero.
        """
        # d_distr_samples are (mdim, M)
        # rect_domain is (mdim, 2)
        inside = np.logical_and(np.all(np.greater_equal(self.d_distr_samples,
            self.rect_domain[:, 0]), axis=1),
            np.all(np.less_equal(self.d_distr_samples,
            self.rect_domain[:, 1]), axis=1)) 
        assert np.all(self.rho_D_M[inside] >= 0.0)
        #print self.rect_domain
        #print "ind, inside", inside
        #print "ind, outside", np.logical_not(inside)
        #print "inside", self.d_distr_samples[inside]
        #print "outside", self.d_distr_samples[np.logical_not(inside)]
        #print "inside", self.rho_D_M[inside]
        #print "outside", self.rho_D_M[np.logical_not(inside)]
        assert np.all(self.rho_D_M[np.logical_not(inside)] == 0.0)


class data_01D(object):
    """
    Sets up 01D data domain problem.
    """
    def createData(self):
        """
        Set up data.
        """
        self.data = np.random.random((100,))*10.0
        self.Q_ref = 5.0
        self.data_domain = np.array([0.0, 10.0])
        self.mdim = 1


class data_1D(object):
    """
    Sets up 1D data domain problem.
    """
    def createData(self):
        """
        Set up data.
        """
        self.data = np.random.random((100, 1))*10.0
        self.Q_ref = np.array([5.0])
        self.data_domain = np.expand_dims(np.array([0.0, 10.0]), axis=0)
        self.mdim = 1


class data_2D(object):
    """
    Sets up 2D data domain problem.
    """
    def createData(self):
        """
        Set up data.
        """
        self.data = np.random.random((100, 2))*10.0
        self.Q_ref = np.array([5.0, 5.0])
        self.data_domain = np.array([[0.0, 10.0], [0.0, 10.0]])
        self.mdim = 2


class data_3D(object):
    """
    Sets up 3D data domain problem.
    """
    def createData(self):
        """
        Set up data.
        """
        self.data = np.random.random((100, 3))*10.0
        self.Q_ref = np.array([5.0, 5.0, 5.0])
        self.data_domain = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
        self.mdim = 3

class unif_unif(prob_uniform):
    """
    Set up :meth:`bet.calculateP.simpleFunP.unif_unif` on data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.unif_unif(self.data, 
                self.Q_ref, M=67, bin_ratio=0.1, num_d_emulate=1E3)

        if type(self.Q_ref) != np.array:
            self.Q_ref = np.array([self.Q_ref])
        if len(self.data_domain.shape) == 1:
            self.data_domain = np.expand_dims(self.data_domain, axis=0)

        self.rect_domain = np.zeros((self.data_domain.shape[0], 2))
        r_width = 0.1*self.data_domain[:, 1]

        self.rect_domain[:, 0] = self.Q_ref - .5*r_width
        self.rect_domain[:, 1] = self.Q_ref + .5*r_width
         
    def test_M(self):
        """
        Test that the right number of d_distr_samples are used to create
        rho_D_M.
        """
        assert len(self.rho_D_M) == 67

    def test_domain(self):
        """
        Test that the probabilities within the prescribed domain are non-zero
        and that the probabilities outside of the prescribed domain are zero.
        """
        # d_distr_samples are (mdim, M)
        # rect_domain is (mdim, 2)
        inside = np.logical_and(np.all(np.greater_equal(self.d_distr_samples,
            self.rect_domain[:, 0]), axis=1),
            np.all(np.less_equal(self.d_distr_samples,
            self.rect_domain[:, 1]), axis=1)) 
        msg = "Due to the inherent randomness of this method, this may fail."
        print msg
        print np.sum(self.rho_D_M[inside] >= 0.0)
        assert np.sum(self.rho_D_M[inside] >= 0.0)<100
        print np.sum(self.rho_D_M[np.logical_not(inside)] == 0.0)
        assert np.sum(self.rho_D_M[np.logical_not(inside)] == 0.0)<100

class test_unif_unif_01D(data_01D, unif_unif):
    """
    Tests :meth:`bet.calculateP.simpleFunP.unif_unif` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_unif_unif_01D, self).createData()
        super(test_unif_unif_01D, self).setUp()

class test_unif_unif_1D(data_1D, unif_unif):
    """
    Tests :meth:`bet.calculateP.simpleFunP.unif_unif` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_unif_unif_1D, self).createData()
        super(test_unif_unif_1D, self).setUp()


class test_unif_unif_2D(data_2D, unif_unif):
    """
    Tests :meth:`bet.calculateP.simpleFunP.unif_unif` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_unif_unif_2D, self).createData()
        super(test_unif_unif_2D, self).setUp()


class test_unif_unif_3D(data_3D, unif_unif):
    """
    Tests :meth:`bet.calculateP.simpleFunP.unif_unif` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_unif_unif_3D, self).createData()
        super(test_unif_unif_3D, self).setUp()

class normal_normal(prob):
    """
    Set up :meth:`bet.calculateP.simpleFunP.normal_normal` on data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        if type(self.Q_ref) != np.array and type(self.Q_ref) != np.ndarray:
            std = 1.0
        else:
            std = np.ones(self.Q_ref.shape)
        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.normal_normal(self.Q_ref, 
		M=67, std=std, num_d_emulate=1E3)
         
    def test_M(self):
        """
        Test that the right number of d_distr_samples are used to create
        rho_D_M.
        """
        assert len(self.rho_D_M) == 67

class test_normal_normal_01D(data_01D, normal_normal):
    """
    Tests :meth:`bet.calculateP.simpleFunP.normal_normal` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_normal_normal_01D, self).createData()
        super(test_normal_normal_01D, self).setUp()

class test_normal_normal_1D(data_1D, normal_normal):
    """
    Tests :meth:`bet.calculateP.simpleFunP.normal_normal` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_normal_normal_1D, self).createData()
        super(test_normal_normal_1D, self).setUp()


class test_normal_normal_2D(data_2D, normal_normal):
    """
    Tests :meth:`bet.calculateP.simpleFunP.normal_normal` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_normal_normal_2D, self).createData()
        super(test_normal_normal_2D, self).setUp()


class test_normal_normal_3D(data_3D, normal_normal):
    """
    Tests :meth:`bet.calculateP.simpleFunP.normal_normal` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_normal_normal_3D, self).createData()
        super(test_normal_normal_3D, self).setUp()


class uniform_hyperrectangle_base(prob_uniform):
    """
    Provides set up and a test to check the number of ``d_distr_samples`` for
    an exact simple function approximation of a hyperrectangle.
    """
    def test_M(self):
        """
        Test that the right number of d_distr_samples are used to create
        rho_D_M.
        """
        if not isinstance(self.center_pts_per_edge, collections.Iterable):
            assert len(self.rho_D_M) == (self.center_pts_per_edge+2)**self.mdim
        else:
            assert len(self.rho_D_M) == np.prod(self.center_pts_per_edge+2)


class uniform_hyperrectangle_int(uniform_hyperrectangle_base):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_*` with an
    int type value for ``center_pts_per_edge``.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.center_pts_per_edge = 2


class uniform_hyperrectangle_list(uniform_hyperrectangle_base):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_*` with an
    iterable type value for ``center_pts_per_edge``.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.center_pts_per_edge = 2*np.ones((self.mdim,), dtype=np.int)

class uniform_hyperrectangle_user_int(uniform_hyperrectangle_int):
    """
    Set up :met:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user` with an
    int type of value fo r``center_pts_per_edge``
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(uniform_hyperrectangle_user_int, self).setUp()
        if type(self.Q_ref) != np.array:
            Q_ref = np.array([self.Q_ref])
        else:
            Q_ref = self.Q_ref
        if len(self.data_domain.shape) == 1:
            data_domain = np.expand_dims(self.data_domain, axis=0)
        else:
            data_domain = self.data_domain

        self.rect_domain = np.zeros((data_domain.shape[0], 2))
        r_width = 0.1*data_domain[:, 1]

        self.rect_domain[:, 0] = Q_ref - .5*r_width
        self.rect_domain[:, 1] = Q_ref + .5*r_width

        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.uniform_hyperrectangle_user(self.data, 
                self.rect_domain.transpose(), self.center_pts_per_edge)

class uniform_hyperrectangle_user_list(uniform_hyperrectangle_list):
    """
    Set up :met:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user` with an
    int type of value fo r``center_pts_per_edge``
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(uniform_hyperrectangle_user_list, self).setUp()
        if type(self.Q_ref) != np.array:
            Q_ref = np.array([self.Q_ref])
        else:
            Q_ref = self.Q_ref
        if len(self.data_domain.shape) == 1:
            data_domain = np.expand_dims(self.data_domain, axis=0)
        else:
            data_domain = self.data_domain

        self.rect_domain = np.zeros((data_domain.shape[0], 2))
        r_width = 0.1*data_domain[:, 1]

        self.rect_domain[:, 0] = Q_ref - .5*r_width
        self.rect_domain[:, 1] = Q_ref + .5*r_width

        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.uniform_hyperrectangle_user(self.data, 
                self.rect_domain.transpose(), self.center_pts_per_edge)


class test_uniform_hyperrectangle_user_int_01D(data_01D, uniform_hyperrectangle_user_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user_int` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_user_int_01D, self).createData()
        super(test_uniform_hyperrectangle_user_int_01D, self).setUp()

class test_uniform_hyperrectangle_user_int_1D(data_1D, uniform_hyperrectangle_user_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user_int` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_user_int_1D, self).createData()
        super(test_uniform_hyperrectangle_user_int_1D, self).setUp()


class test_uniform_hyperrectangle_user_int_2D(data_2D, uniform_hyperrectangle_user_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user_int` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_user_int_2D, self).createData()
        super(test_uniform_hyperrectangle_user_int_2D, self).setUp()


class test_uniform_hyperrectangle_user_int_3D(data_3D, uniform_hyperrectangle_user_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user_int` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_user_int_3D, self).createData()
        super(test_uniform_hyperrectangle_user_int_3D, self).setUp()


class test_uniform_hyperrectangle_user_list_01D(data_01D, uniform_hyperrectangle_user_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user_list` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_user_list_01D, self).createData()
        super(test_uniform_hyperrectangle_user_list_01D, self).setUp()

class test_uniform_hyperrectangle_user_list_1D(data_1D, uniform_hyperrectangle_user_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user_list` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_user_list_1D, self).createData()
        super(test_uniform_hyperrectangle_user_list_1D, self).setUp()


class test_uniform_hyperrectangle_user_list_2D(data_2D, uniform_hyperrectangle_user_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user_list` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_user_list_2D, self).createData()
        super(test_uniform_hyperrectangle_user_list_2D, self).setUp()


class test_uniform_hyperrectangle_user_list_3D(data_3D, uniform_hyperrectangle_user_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_user_list` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_user_list_3D, self).createData()
        super(test_uniform_hyperrectangle_user_list_3D, self).setUp()


class uniform_hyperrectangle_size_int(uniform_hyperrectangle_int):
    """
    Set up :met:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size` with an
    int type of value fo r``center_pts_per_edge``
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(uniform_hyperrectangle_size_int, self).setUp()
        if type(self.Q_ref) != np.array:
            Q_ref = np.array([self.Q_ref])
        else:
            Q_ref = self.Q_ref
        if len(self.data_domain.shape) == 1:
            data_domain = np.expand_dims(self.data_domain, axis=0)
        else:
            data_domain = self.data_domain

        self.rect_domain = np.zeros((data_domain.shape[0], 2))
        binsize = 1.0
        r_width = binsize*np.ones(data_domain[:, 1].shape)

        self.rect_domain[:, 0] = Q_ref - .5*r_width
        self.rect_domain[:, 1] = Q_ref + .5*r_width

        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.uniform_hyperrectangle_binsize(self.data, 
                self.Q_ref, binsize, self.center_pts_per_edge)

class uniform_hyperrectangle_size_list(uniform_hyperrectangle_list):
    """
    Set up :met:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size` with an
    int type of value fo r``center_pts_per_edge``
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(uniform_hyperrectangle_size_list, self).setUp()
        if type(self.Q_ref) != np.array:
            Q_ref = np.array([self.Q_ref])
        else:
            Q_ref = self.Q_ref
        if len(self.data_domain.shape) == 1:
            data_domain = np.expand_dims(self.data_domain, axis=0)
        else:
            data_domain = self.data_domain

        self.rect_domain = np.zeros((data_domain.shape[0], 2))
        binsize = 1.0*np.ones((data_domain.shape[0],))
        r_width = binsize

        self.rect_domain[:, 0] = Q_ref - .5*r_width
        self.rect_domain[:, 1] = Q_ref + .5*r_width

        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.uniform_hyperrectangle_binsize(self.data, 
                self.Q_ref, binsize, self.center_pts_per_edge)


class test_uniform_hyperrectangle_size_int_01D(data_01D, uniform_hyperrectangle_size_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size_int` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_size_int_01D, self).createData()
        super(test_uniform_hyperrectangle_size_int_01D, self).setUp()

class test_uniform_hyperrectangle_size_int_1D(data_1D, uniform_hyperrectangle_size_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size_int` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_size_int_1D, self).createData()
        super(test_uniform_hyperrectangle_size_int_1D, self).setUp()


class test_uniform_hyperrectangle_size_int_2D(data_2D, uniform_hyperrectangle_size_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size_int` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_size_int_2D, self).createData()
        super(test_uniform_hyperrectangle_size_int_2D, self).setUp()


class test_uniform_hyperrectangle_size_int_3D(data_3D, uniform_hyperrectangle_size_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size_int` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_size_int_3D, self).createData()
        super(test_uniform_hyperrectangle_size_int_3D, self).setUp()


class test_uniform_hyperrectangle_size_list_01D(data_01D, uniform_hyperrectangle_size_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size_list` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_size_list_01D, self).createData()
        super(test_uniform_hyperrectangle_size_list_01D, self).setUp()

class test_uniform_hyperrectangle_size_list_1D(data_1D, uniform_hyperrectangle_size_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size_list` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_size_list_1D, self).createData()
        super(test_uniform_hyperrectangle_size_list_1D, self).setUp()


class test_uniform_hyperrectangle_size_list_2D(data_2D, uniform_hyperrectangle_size_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size_list` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_size_list_2D, self).createData()
        super(test_uniform_hyperrectangle_size_list_2D, self).setUp()


class test_uniform_hyperrectangle_size_list_3D(data_3D, uniform_hyperrectangle_size_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_size_list` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_size_list_3D, self).createData()
        super(test_uniform_hyperrectangle_size_list_3D, self).setUp()

class uniform_hyperrectangle_ratio_int(uniform_hyperrectangle_int):
    """
    Set up :met:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio` with an
    int type of value fo r``center_pts_per_edge``
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(uniform_hyperrectangle_ratio_int, self).setUp()
        if type(self.Q_ref) != np.array:
            Q_ref = np.array([self.Q_ref])
        else:
            Q_ref = self.Q_ref
        if len(self.data_domain.shape) == 1:
            data_domain = np.expand_dims(self.data_domain, axis=0)
        else:
            data_domain = self.data_domain

        self.rect_domain = np.zeros((data_domain.shape[0], 2))
        binratio = 0.1
        r_width = binratio*data_domain[:, 1]

        self.rect_domain[:, 0] = Q_ref - .5*r_width
        self.rect_domain[:, 1] = Q_ref + .5*r_width

        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.uniform_hyperrectangle(self.data, 
                self.Q_ref, binratio, self.center_pts_per_edge)

class uniform_hyperrectangle_ratio_list(uniform_hyperrectangle_list):
    """
    Set up :met:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio` with an
    int type of value fo r``center_pts_per_edge``
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(uniform_hyperrectangle_ratio_list, self).setUp()
        if type(self.Q_ref) != np.array:
            Q_ref = np.array([self.Q_ref])
        else:
            Q_ref = self.Q_ref
        if len(self.data_domain.shape) == 1:
            data_domain = np.expand_dims(self.data_domain, axis=0)
        else:
            data_domain = self.data_domain

        self.rect_domain = np.zeros((data_domain.shape[0], 2))
        binratio = 0.1*np.ones((data_domain.shape[0],))
        r_width = binratio*data_domain[:,1]

        self.rect_domain[:, 0] = Q_ref - .5*r_width
        self.rect_domain[:, 1] = Q_ref + .5*r_width

        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.uniform_hyperrectangle(self.data, 
                self.Q_ref, binratio, self.center_pts_per_edge)


class test_uniform_hyperrectangle_ratio_int_01D(data_01D, uniform_hyperrectangle_ratio_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio_int` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_ratio_int_01D, self).createData()
        super(test_uniform_hyperrectangle_ratio_int_01D, self).setUp()

class test_uniform_hyperrectangle_ratio_int_1D(data_1D, uniform_hyperrectangle_ratio_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio_int` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_ratio_int_1D, self).createData()
        super(test_uniform_hyperrectangle_ratio_int_1D, self).setUp()


class test_uniform_hyperrectangle_ratio_int_2D(data_2D, uniform_hyperrectangle_ratio_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio_int` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_ratio_int_2D, self).createData()
        super(test_uniform_hyperrectangle_ratio_int_2D, self).setUp()


class test_uniform_hyperrectangle_ratio_int_3D(data_3D, uniform_hyperrectangle_ratio_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio_int` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_ratio_int_3D, self).createData()
        super(test_uniform_hyperrectangle_ratio_int_3D, self).setUp()


class test_uniform_hyperrectangle_ratio_list_01D(data_01D, uniform_hyperrectangle_ratio_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio_list` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_ratio_list_01D, self).createData()
        super(test_uniform_hyperrectangle_ratio_list_01D, self).setUp()

class test_uniform_hyperrectangle_ratio_list_1D(data_1D, uniform_hyperrectangle_ratio_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio_list` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_ratio_list_1D, self).createData()
        super(test_uniform_hyperrectangle_ratio_list_1D, self).setUp()


class test_uniform_hyperrectangle_ratio_list_2D(data_2D, uniform_hyperrectangle_ratio_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio_list` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_ratio_list_2D, self).createData()
        super(test_uniform_hyperrectangle_ratio_list_2D, self).setUp()


class test_uniform_hyperrectangle_ratio_list_3D(data_3D, uniform_hyperrectangle_ratio_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_ratio_list` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_hyperrectangle_ratio_list_3D, self).createData()
        super(test_uniform_hyperrectangle_ratio_list_3D, self).setUp()

class uniform_data(prob_uniform):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_data` on data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.rho_D_M, self.d_distr_samples, self.d_Tree = sFun.uniform_data(self.data)

        if type(self.Q_ref) != np.array:
            self.Q_ref = np.array([self.Q_ref])
        if len(self.data_domain.shape) == 1:
            self.data_domain = np.expand_dims(self.data_domain, axis=0)
        self.rect_domain = self.data_domain
         
    def test_M(self):
        """
        Test that the right number of d_distr_samples are used to create
        rho_D_M.
        """
        assert len(self.rho_D_M) == self.data.shape[0]

class test_uniform_data_01D(data_01D, uniform_data):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_data` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_data_01D, self).createData()
        super(test_uniform_data_01D, self).setUp()

class test_uniform_data_1D(data_1D, uniform_data):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_data` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_data_1D, self).createData()
        super(test_uniform_data_1D, self).setUp()


class test_uniform_data_2D(data_2D, uniform_data):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_data` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_data_2D, self).createData()
        super(test_uniform_data_2D, self).setUp()


class test_uniform_data_3D(data_3D, uniform_data):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_data` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_data_3D, self).createData()
        super(test_uniform_data_3D, self).setUp()


