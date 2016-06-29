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
import bet.sample as samp

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
        #assert (self.d_Tree.n, self.d_Tree.m) == self.d_distr_samples.shape


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
        self.data = samp.sample_set(1)
        self.data.set_values(np.random.random((100,))*10.0)
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
        self.data = samp.sample_set(1)
        self.data.set_values(np.random.random((100,1))*10.0)
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
        self.data = samp.sample_set(2)
        self.data.set_values(np.random.random((100,2))*10.0)
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
        self.data = samp.sample_set(3)
        self.data.set_values(np.random.random((100,3))*10.0)
        self.Q_ref = np.array([5.0, 5.0, 5.0])
        self.data_domain = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
        self.mdim = 3

class uniform_partition_uniform_distribution_rectangle_scaled(prob_uniform):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled` on data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.data_prob = sFun.uniform_partition_uniform_distribution_rectangle_scaled(
            self.data, self.Q_ref, rect_scale=0.1, M=67, num_d_emulate=1E3)
        self.d_distr_samples = self.data_prob.get_values()
        self.rho_D_M = self.data_prob.get_probabilities()

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

class test_uniform_partition_uniform_distribution_rectangle_scaled_01D(data_01D,
                        uniform_partition_uniform_distribution_rectangle_scaled):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_scaled_01D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_scaled_01D, self).setUp()

class test_uniform_partition_uniform_distribution_rectangle_scaled_1D(data_1D,
                        uniform_partition_uniform_distribution_rectangle_scaled):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_scaled_1D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_scaled_1D, self).setUp()


class test_uniform_partition_uniform_distribution_rectangle_scaled_2D(data_2D,
                        uniform_partition_uniform_distribution_rectangle_scaled):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_scaled_2D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_scaled_2D, self).setUp()


class test_uniform_partition_uniform_distribution_rectangle_scaled_3D(data_3D,
                        uniform_partition_uniform_distribution_rectangle_scaled):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_scaled_3D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_scaled_3D, self).setUp()

class normal_partition_normal_distribution(prob):
    """
    Set up :meth:`bet.calculateP.simpleFunP.normal_partition_normal_distribution` on data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        if type(self.Q_ref) != np.array and type(self.Q_ref) != np.ndarray:
            std = 1.0
        else:
            std = np.ones(self.Q_ref.shape)
        self.data_prob = sFun.normal_partition_normal_distribution(None, self.Q_ref, std=std, M=67, num_d_emulate=1E3)
        self.d_distr_samples = self.data_prob.get_values()
        self.rho_D_M = self.data_prob.get_probabilities()
         
    def test_M(self):
        """
        Test that the right number of d_distr_samples are used to create
        rho_D_M.
        """
        assert len(self.rho_D_M) == 67

class test_normal_partition_normal_distribution_01D(data_01D, normal_partition_normal_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.normal_partition_normal_distribution` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_normal_partition_normal_distribution_01D, self).createData()
        super(test_normal_partition_normal_distribution_01D, self).setUp()

class test_normal_partition_normal_distribution_1D(data_1D, normal_partition_normal_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.normal_partition_normal_distribution` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_normal_partition_normal_distribution_1D, self).createData()
        super(test_normal_partition_normal_distribution_1D, self).setUp()


class test_normal_partition_normal_distribution_2D(data_2D, normal_partition_normal_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.normal_partition_normal_distribution` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_normal_partition_normal_distribution_2D, self).createData()
        super(test_normal_partition_normal_distribution_2D, self).setUp()


class test_normal_partition_normal_distribution_3D(data_3D, normal_partition_normal_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.normal_partition_normal_distribution` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_normal_partition_normal_distribution_3D, self).createData()
        super(test_normal_partition_normal_distribution_3D, self).setUp()


class uniform_partition_normal_distribution(prob):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_partition_normal_distribution` on data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        if type(self.Q_ref) != np.array and type(self.Q_ref) != np.ndarray:
            std = 1.0
        else:
            std = np.ones(self.Q_ref.shape)
        self.data_prob = sFun.uniform_partition_normal_distribution(None, self.Q_ref, std=std, M=67, num_d_emulate=1E3)
        self.d_distr_samples = self.data_prob.get_values()
        self.rho_D_M = self.data_prob.get_probabilities()

    def test_M(self):
        """
        Test that the right number of d_distr_samples are used to create
        rho_D_M.
        """
        assert len(self.rho_D_M) == 67


class test_uniform_partition_normal_distribution_01D(data_01D, uniform_partition_normal_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_normal_distribution` on 01D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_normal_distribution_01D, self).createData()
        super(test_uniform_partition_normal_distribution_01D, self).setUp()


class test_uniform_partition_normal_distribution_1D(data_1D, uniform_partition_normal_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_normal_distribution` on 1D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_normal_distribution_1D, self).createData()
        super(test_uniform_partition_normal_distribution_1D, self).setUp()


class test_uniform_partition_normal_distribution_2D(data_2D, uniform_partition_normal_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_normal_distribution` on 2D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_normal_distribution_2D, self).createData()
        super(test_uniform_partition_normal_distribution_2D, self).setUp()


class test_uniform_partition_normal_distribution_3D(data_3D, uniform_partition_normal_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_normal_distribution` on 3D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_normal_distribution_3D, self).createData()
        super(test_uniform_partition_normal_distribution_3D, self).setUp()


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
        assert len(self.rho_D_M) == 2

class uniform_hyperrectangle_int(uniform_hyperrectangle_base):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_*`.
    """
    def setUp(self):
        """
        Set up problem.
        """

class uniform_hyperrectangle_list(uniform_hyperrectangle_base):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_hyperrectangle_*.
    """
    def setUp(self):
        """
        Set up problem.
        """

class regular_partition_uniform_distribution_rectangle_domain_int(uniform_hyperrectangle_int):
    """
    Set up :met:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain`.
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(regular_partition_uniform_distribution_rectangle_domain_int, self).setUp()
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

        self.data_prob = sFun.regular_partition_uniform_distribution_rectangle_domain(
            self.data, self.rect_domain.transpose())
        self.rho_D_M = self.data_prob._probabilities
        self.d_distr_samples = self.data_prob._values
        
class regular_partition_uniform_distribution_rectangle_domain_list(uniform_hyperrectangle_list):
    """
    Set up :met:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain`.
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(regular_partition_uniform_distribution_rectangle_domain_list, self).setUp()
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

        self.data_prob = sFun.regular_partition_uniform_distribution_rectangle_domain(
            self.data, self.rect_domain.transpose())
        self.rho_D_M = self.data_prob._probabilities
        self.d_distr_samples = self.data_prob._values


class test_regular_partition_uniform_distribution_rectangle_domain_int_01D(data_01D,
                                        regular_partition_uniform_distribution_rectangle_domain_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_domain_int_01D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_domain_int_01D, self).setUp()

class test_regular_partition_uniform_distribution_rectangle_domain_int_1D(data_1D,
                                        regular_partition_uniform_distribution_rectangle_domain_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_domain_int_1D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_domain_int_1D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_domain_int_2D(data_2D,
                                        regular_partition_uniform_distribution_rectangle_domain_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_domain_int_2D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_domain_int_2D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_domain_int_3D(data_3D,
                                        regular_partition_uniform_distribution_rectangle_domain_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_domain_int_3D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_domain_int_3D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_domain_list_01D(data_01D,
                                        regular_partition_uniform_distribution_rectangle_domain_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_domain_list_01D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_domain_list_01D, self).setUp()

class test_regular_partition_uniform_distribution_rectangle_domain_list_1D(data_1D,
                                        regular_partition_uniform_distribution_rectangle_domain_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_domain_list_1D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_domain_list_1D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_domain_list_2D(data_2D,
                                        regular_partition_uniform_distribution_rectangle_domain_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_domain_list_2D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_domain_list_2D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_domain_list_3D(data_3D,
                                        regular_partition_uniform_distribution_rectangle_domain_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_domain` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_domain_list_3D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_domain_list_3D, self).setUp()


class regular_partition_uniform_distribution_rectangle_size_int(uniform_hyperrectangle_int):
    """
    Set up :met:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size``
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(regular_partition_uniform_distribution_rectangle_size_int, self).setUp()
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

        self.data_prob = sFun.regular_partition_uniform_distribution_rectangle_size(
            self.data, self.Q_ref, binsize)
        self.rho_D_M = self.data_prob._probabilities
        self.d_distr_samples = self.data_prob._values

class regular_partition_uniform_distribution_rectangle_size_list(uniform_hyperrectangle_list):
    """
    Set up :met:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` 
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(regular_partition_uniform_distribution_rectangle_size_list, self).setUp()
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

        self.data_prob = sFun.regular_partition_uniform_distribution_rectangle_size(
            self.data, self.Q_ref, binsize)
        self.rho_D_M = self.data_prob._probabilities
        self.d_distr_samples = self.data_prob._values


class test_regular_partition_uniform_distribution_rectangle_size_int_01D(data_01D,
                                    regular_partition_uniform_distribution_rectangle_size_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_size_int_01D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_size_int_01D, self).setUp()

class test_regular_partition_uniform_distribution_rectangle_size_int_1D(data_1D,
                                    regular_partition_uniform_distribution_rectangle_size_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_size_int_1D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_size_int_1D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_size_int_2D(data_2D,
                                    regular_partition_uniform_distribution_rectangle_size_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_size_int_2D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_size_int_2D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_size_int_3D(data_3D,
                                    regular_partition_uniform_distribution_rectangle_size_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_size_int_3D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_size_int_3D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_size_list_01D(data_01D,
                                    regular_partition_uniform_distribution_rectangle_size_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_size_list_01D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_size_list_01D, self).setUp()

class test_regular_partition_uniform_distribution_rectangle_size_list_1D(data_1D,
                                    regular_partition_uniform_distribution_rectangle_size_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_size_list_1D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_size_list_1D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_size_list_2D(data_2D,
                                    regular_partition_uniform_distribution_rectangle_size_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_size_list_2D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_size_list_2D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_size_list_3D(data_3D,
                                    regular_partition_uniform_distribution_rectangle_size_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_size` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_size_list_3D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_size_list_3D, self).setUp()

class regular_partition_uniform_distribution_rectangle_scaled_int(uniform_hyperrectangle_int):
    """
    Set up :met:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled`
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(regular_partition_uniform_distribution_rectangle_scaled_int, self).setUp()
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

        self.data_prob = sFun.regular_partition_uniform_distribution_rectangle_scaled(
            self.data, self.Q_ref, binratio)
        self.rho_D_M = self.data_prob._probabilities
        self.d_distr_samples = self.data_prob._values

class regular_partition_uniform_distribution_rectangle_scaled_list(uniform_hyperrectangle_list):
    """
    Set up :met:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled` 
    
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(regular_partition_uniform_distribution_rectangle_scaled_list, self).setUp()
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

        self.data_prob = sFun.regular_partition_uniform_distribution_rectangle_scaled(
            self.data, self.Q_ref, binratio)
        self.rho_D_M = self.data_prob._probabilities
        self.d_distr_samples = self.data_prob._values


class test_regular_partition_uniform_distribution_rectangle_scaled_int_01D(data_01D,
                                regular_partition_uniform_distribution_rectangle_scaled_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_scaled_int_01D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_scaled_int_01D, self).setUp()

class test_regular_partition_uniform_distribution_rectangle_scaled_int_1D(data_1D,
                                regular_partition_uniform_distribution_rectangle_scaled_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_scaled_int_1D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_scaled_int_1D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_scaled_int_2D(data_2D,
                                regular_partition_uniform_distribution_rectangle_scaled_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_scaled_int_2D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_scaled_int_2D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_scaled_int_3D(data_3D,
                                regular_partition_uniform_distribution_rectangle_scaled_int):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_scaled_int_3D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_scaled_int_3D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_scaled_list_01D(data_01D,
                                regular_partition_uniform_distribution_rectangle_scaled_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled_list` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_scaled_list_01D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_scaled_list_01D, self).setUp()

class test_regular_partition_uniform_distribution_rectangle_scaled_list_1D(data_1D,
                                regular_partition_uniform_distribution_rectangle_scaled_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_scaled_list_1D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_scaled_list_1D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_scaled_list_2D(data_2D,
                                regular_partition_uniform_distribution_rectangle_scaled_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_scaled_list_2D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_scaled_list_2D, self).setUp()


class test_regular_partition_uniform_distribution_rectangle_scaled_list_3D(data_3D,
                                regular_partition_uniform_distribution_rectangle_scaled_list):
    """
    Tests :meth:`bet.calculateP.simpleFunP.regular_partition_uniform_distribution_rectangle_scaled` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_regular_partition_uniform_distribution_rectangle_scaled_list_3D, self).createData()
        super(test_regular_partition_uniform_distribution_rectangle_scaled_list_3D, self).setUp()

class uniform_partition_uniform_distribution_data_samples(prob_uniform):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_data_samples` on data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.data_prob = sFun.uniform_partition_uniform_distribution_data_samples(self.data)
        self.d_distr_samples = self.data_prob.get_values()
        self.rho_D_M = self.data_prob.get_probabilities()
        self.data = self.data._values

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

class test_uniform_partition_uniform_distribution_data_samples_01D(data_01D,
                                        uniform_partition_uniform_distribution_data_samples):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_data_samples` on 01D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_data_samples_01D, self).createData()
        super(test_uniform_partition_uniform_distribution_data_samples_01D, self).setUp()

class test_uniform_partition_uniform_distribution_data_samples_1D(data_1D,
                                        uniform_partition_uniform_distribution_data_samples):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_data_samples` on 1D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_data_samples_1D, self).createData()
        super(test_uniform_partition_uniform_distribution_data_samples_1D, self).setUp()


class test_uniform_partition_uniform_distribution_data_samples_2D(data_2D,
                                        uniform_partition_uniform_distribution_data_samples):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_data_samples` on 2D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_data_samples_2D, self).createData()
        super(test_uniform_partition_uniform_distribution_data_samples_2D, self).setUp()


class test_uniform_partition_uniform_distribution_data_samples_3D(data_3D,
                                        uniform_partition_uniform_distribution_data_samples):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_data_samples` on 3D data domain.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_data_samples_3D, self).createData()
        super(test_uniform_partition_uniform_distribution_data_samples_3D, self).setUp()


class uniform_partition_uniform_distribution_rectangle_size(prob_uniform):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_size` on data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        self.data_prob = sFun.uniform_partition_uniform_distribution_rectangle_size(
            self.data, self.Q_ref, rect_size=1.0, M=67, num_d_emulate=1E3)
        self.d_distr_samples = self.data_prob.get_values()
        self.rho_D_M = self.data_prob.get_probabilities()

        if type(self.Q_ref) != np.array:
            self.Q_ref = np.array([self.Q_ref])
        if len(self.data_domain.shape) == 1:
            self.data_domain = np.expand_dims(self.data_domain, axis=0)

        self.rect_domain = np.zeros((self.data_domain.shape[0], 2))

        binsize = 1.0
        r_width = binsize * np.ones(self.data_domain[:, 1].shape)

        self.rect_domain[:, 0] = self.Q_ref - .5 * r_width
        self.rect_domain[:, 1] = self.Q_ref + .5 * r_width

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
        assert np.sum(self.rho_D_M[inside] >= 0.0) < 100
        print np.sum(self.rho_D_M[np.logical_not(inside)] == 0.0)
        assert np.sum(self.rho_D_M[np.logical_not(inside)] == 0.0) < 100


class test_uniform_partition_uniform_distribution_rectangle_size_01D(data_01D,
                                            uniform_partition_uniform_distribution_rectangle_size):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_size` on 01D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_size_01D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_size_01D, self).setUp()


class test_uniform_partition_uniform_distribution_rectangle_size_1D(data_1D,
                                            uniform_partition_uniform_distribution_rectangle_size):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_size` on 1D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_size_1D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_size_1D, self).setUp()


class test_uniform_partition_uniform_distribution_rectangle_size_2D(data_2D,
                                            uniform_partition_uniform_distribution_rectangle_size):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_size` on 2D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_size_2D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_size_2D, self).setUp()


class test_uniform_partition_uniform_distribution_rectangle_size_3D(data_3D,
                                            uniform_partition_uniform_distribution_rectangle_size):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_size` on 3D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_size_3D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_size_3D, self).setUp()


class uniform_partition_uniform_distribution_rectangle_domain(prob_uniform):
    """
    Set up :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_domain` on data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        if type(self.Q_ref) != np.array:
            Q_ref = np.array([self.Q_ref])
        else:
            Q_ref = self.Q_ref
        if len(self.data_domain.shape) == 1:
            data_domain = np.expand_dims(self.data_domain, axis=0)
        else:
            data_domain = self.data_domain

        self.rect_domain = np.zeros((data_domain.shape[0], 2))
        r_width = 0.1 * data_domain[:, 1]

        self.rect_domain[:, 0] = Q_ref - .5 * r_width
        self.rect_domain[:, 1] = Q_ref + .5 * r_width

        self.data_prob = sFun.uniform_partition_uniform_distribution_rectangle_domain(
            self.data, self.rect_domain.transpose(), M=67, num_d_emulate=1E3)
        self.d_distr_samples = self.data_prob.get_values()
        self.rho_D_M = self.data_prob.get_probabilities()

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
        assert np.sum(self.rho_D_M[inside] >= 0.0) < 100
        print np.sum(self.rho_D_M[np.logical_not(inside)] == 0.0)
        assert np.sum(self.rho_D_M[np.logical_not(inside)] == 0.0) < 100


class test_uniform_partition_uniform_distribution_rectangle_domain_01D(data_01D,
                                                uniform_partition_uniform_distribution_rectangle_domain):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_domain` on 01D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_domain_01D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_domain_01D, self).setUp()


class test_uniform_partition_uniform_distribution_rectangle_domain_1D(data_1D,
                                                uniform_partition_uniform_distribution_rectangle_domain):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_domain` on 1D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_domain_1D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_domain_1D, self).setUp()


class test_uniform_partition_uniform_distribution_rectangle_domain_2D(data_2D,
                                                uniform_partition_uniform_distribution_rectangle_domain):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_domain` on 2D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_domain_2D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_domain_2D, self).setUp()


class test_uniform_partition_uniform_distribution_rectangle_domain_3D(data_3D,
                                                uniform_partition_uniform_distribution_rectangle_domain):
    """
    Tests :meth:`bet.calculateP.simpleFunP.uniform_partition_uniform_distribution_rectangle_domain` on 3D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_uniform_partition_uniform_distribution_rectangle_domain_3D, self).createData()
        super(test_uniform_partition_uniform_distribution_rectangle_domain_3D, self).setUp()


class user_partition_user_distribution(prob):
    """
    Set up :meth:`bet.calculateP.simpleFunP.user_partition_user_distribution` on data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        self.data_prob = sFun.user_partition_user_distribution(self.data,
                                                               self.data,
                                                               self.data)
        self.rho_D_M = self.data_prob.get_probabilities()
        self.d_distr_samples = self.data_prob.get_values()

class test_user_partition_user_distribution_01D(data_01D,
                                                user_partition_user_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.user_partition_user_distribution` on 01D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_user_partition_user_distribution_01D, self).createData()
        super(test_user_partition_user_distribution_01D, self).setUp()


class test_user_partition_user_distribution_1D(data_1D,
                                               user_partition_user_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.user_partition_user_distribution` on 1D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_user_partition_user_distribution_1D, self).createData()
        super(test_user_partition_user_distribution_1D, self).setUp()


class test_user_partition_user_distribution_2D(data_2D,
                                               user_partition_user_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.user_partition_user_distribution` on 2D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_user_partition_user_distribution_2D, self).createData()
        super(test_user_partition_user_distribution_2D, self).setUp()


class test_user_partition_user_distribution_3D(data_3D,
                                               user_partition_user_distribution):
    """
    Tests :meth:`bet.calculateP.simpleFunP.user_partition_user_distribution` on 3D data domain.
    """

    def setUp(self):
        """
        Set up problem.
        """
        super(test_user_partition_user_distribution_3D, self).createData()
        super(test_user_partition_user_distribution_3D, self).setUp()
