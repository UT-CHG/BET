# Copyright (C) 2014-2016 The BET Development Team

# Steven Mattis and Lindley Graham 04/06/2015
# Steven Mattis 03/24/2016
"""
This module contains tests for :module:`bet.calculateP.calculateP`.


Most of these tests should make sure certain values are within a tolerance
rather than exact due to the stocastic nature of the algorithms being tested.
"""
import os
import unittest
import bet
import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as simpleFunP
import bet.sampling.basicSampling as bsam
import bet.sample as samp
import numpy as np
import numpy.testing as nptest
import bet.util as util
from bet.Comm import comm 

#data_path = os.path.dirname(bet.__file__) + "/../test/test_calculateP/datafiles"
data_path = "test/test_calculateP/datafiles"

class prob:
    def test_prob_sum_to_1(self):
        """
        Test to see if the prob. sums to 1.
        """
        nptest.assert_almost_equal(np.sum(self.inputs._probabilities), 1.0)
    #@unittest.skipIf(comm.size > 1, 'Only run in serial')
    def test_P_matches_true(self):
        """
        Test against reference probs. (Only in serial)
        """
        nptest.assert_almost_equal(self.P_ref, self.inputs._probabilities)
    def test_vol_sum_to_1(self):
        """
        Test that volume ratios sum to 1.
        """
        nptest.assert_almost_equal(np.sum(self.inputs._volumes), 1.0)
    def test_prob_pos(self):
        """
        Test that all probs are non-negative.
        """
        self.assertEqual(np.sum(np.less(self.inputs._probabilities, 0)), 0)

class prob_on_emulated_samples:
    def test_P_sum_to_1(self):
        """
        Test that prob. sums to 1.
        """
        self.inputs_emulated.local_to_global()
        nptest.assert_almost_equal(np.sum(self.inputs_emulated._probabilities), 1.0)
    def test_P_matches_true(self):
        """
        Test that probabilites match reference values.
        """
        self.inputs_emulated.local_to_global()
        if comm.size == 1:
            nptest.assert_almost_equal(self.P_emulate_ref, self.inputs_emulated._probabilities)
    def test_prob_pos(self):
        """
        Test that all probabilites are non-negative.
        """
        self.inputs_emulated.local_to_global()
        self.assertEqual(np.sum(np.less(self.inputs_emulated._probabilities, 0)), 0)


class prob_with_emulated_volumes:
    def test_P_sum_to_1(self):
        """
        Test that probs sum to 1.
        """
        nptest.assert_almost_equal(np.sum(self.inputs._probabilities), 1.0)
    def test_P_matches_true(self):
        """
        Test the probs. match reference values.
        """
        if comm.size == 1:
            nptest.assert_almost_equal(self.P_ref, self.inputs._probabilities)
    def test_vol_sum_to_1(self):
        """
        Test that volume ratios sum to 1.
        """
        nptest.assert_almost_equal(np.sum(self.inputs._volumes), 1.0)
    def test_prob_pos(self):
        """
        Test that all probs are non-negative.
        """
        self.assertEqual(np.sum(np.less(self.inputs._probabilities, 0)), 0)
        
    
class TestProbMethod_3to2(unittest.TestCase):
    """
    Sets up 3 to 2 map problem.
    """
    def setUp(self):
        self.inputs = samp.sample_set(3)
        self.outputs = samp.sample_set(2)
        self.inputs.set_values(np.loadtxt(data_path + "/3to2_samples.txt.gz"))
        self.outputs.set_values(np.loadtxt(data_path + "/3to2_data.txt.gz"))
        Q_ref = np.array([0.422, 0.9385])
        self.output_prob = simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
            self.outputs, Q_ref = Q_ref, rect_scale=0.2, center_pts_per_edge=1)

        self.inputs.set_domain(np.array([[0.0, 1.0],
                                        [0.0, 1.0],
                                        [0.0, 1.0]]))
        import numpy.random as rnd
        rnd.seed(1)
        self.inputs_emulated = bsam.random_sample_set('r',
                self.inputs.get_domain(), num_samples=1001, globalize=True)
        self.disc = samp.discretization(input_sample_set=self.inputs,
                                        output_sample_set=self.outputs,
                                        output_probability_set=self.output_prob,
                                        emulated_input_sample_set=self.inputs_emulated)

class Test_prob_3to2(TestProbMethod_3to2, prob):
    """
    Test :meth:`bet.calculateP.calculateP.prob` on 3 to 2 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_3to2, self).setUp()
        self.disc._input_sample_set.estimate_volume_mc()
        calcP.prob(self.disc)
        self.P_ref = np.loadtxt(data_path + "/3to2_prob.txt.gz")


class Test_prob_on_emulated_samples_3to2(TestProbMethod_3to2, prob_on_emulated_samples):
    """
    Test :meth:`bet.calculateP.calculateP.prob_on_emulated_samples` on a 3 to 2 map.
    """
    def setUp(self):
        """
        Set up 3 to 2 map.
        """
        super(Test_prob_on_emulated_samples_3to2, self).setUp()
        calcP.prob_on_emulated_samples(self.disc)
        self.P_emulate_ref = np.loadtxt(data_path+"/3to2_prob_emulated.txt.gz")
        #self.P_emulate = util.get_global_values(self.P_emulate)
        

class Test_prob_with_emulated_volumes_3to2(TestProbMethod_3to2, prob_with_emulated_volumes):
    """
    Test :meth:`bet.calculateP.calculateP.prob_with_emulated_volumes` on a 3 to 2 map.
    """
    def setUp(self):
        """
        Set up 3 to 2 problem.
        """
        super(Test_prob_with_emulated_volumes_3to2, self).setUp()
        calcP.prob_with_emulated_volumes(self.disc)
        self.P_ref = np.loadtxt(data_path + "/3to2_prob_mc.txt.gz")
 

class TestProbMethod_3to1(unittest.TestCase):
    """
    Sets up 3 to 1 map problem.
    """
    def setUp(self):
        self.inputs = samp.sample_set(3)
        self.outputs = samp.sample_set(1)
        self.inputs.set_values(np.loadtxt(data_path + "/3to2_samples.txt.gz"))
        self.outputs.set_values(np.loadtxt(data_path + "/3to2_data.txt.gz")[:,0])
        Q_ref = np.array([0.422])
        self.output_prob = simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
            self.outputs, Q_ref = Q_ref, rect_scale=0.2, center_pts_per_edge=1)

        self.inputs.set_domain(np.array([[0.0, 1.0],
                                        [0.0, 1.0],
                                        [0.0, 1.0]]))
        import numpy.random as rnd
        rnd.seed(1)
        self.inputs_emulated = bsam.random_sample_set('r',
                self.inputs.get_domain(), num_samples=1001, globalize=True)
        self.disc = samp.discretization(input_sample_set=self.inputs,
                                        output_sample_set=self.outputs,
                                        output_probability_set=self.output_prob,
                                        emulated_input_sample_set=self.inputs_emulated)

class Test_prob_3to1(TestProbMethod_3to1, prob):
    """
    Test :meth:`bet.calculateP.calculateP.prob` on a 3 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_3to1, self).setUp()
        self.disc._input_sample_set.estimate_volume_mc()
        calcP.prob(self.disc)
        self.P_ref = np.loadtxt(data_path + "/3to1_prob.txt.gz")


class Test_prob_on_emulated_samples_3to1(TestProbMethod_3to1, prob_on_emulated_samples):
    """
    Test :meth:`bet.calculateP.calculateP.prob_on_emulated_samples` on a 3 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_on_emulated_samples_3to1, self).setUp()
        calcP.prob_on_emulated_samples(self.disc)
        self.P_emulate_ref = np.loadtxt(data_path+"/3to1_prob_emulated.txt.gz")


class Test_prob_with_emulated_volumes_3to1(TestProbMethod_3to1, prob_with_emulated_volumes):
    """
    Test :meth:`bet.calculateP.calculateP.prob_with_emulated_volumes` on a 3 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_with_emulated_volumes_3to1, self).setUp()
        calcP.prob_with_emulated_volumes(self.disc)
        self.P_ref = np.loadtxt(data_path + "/3to1_prob_mc.txt.gz")

  
class TestProbMethod_10to4(unittest.TestCase):
    """
    Sets up 10 to 4 map problem.
    """
    def setUp(self):
        """
        Set up problem.
        """
        import numpy.random as rnd
        rnd.seed(1)
        self.inputs = samp.sample_set(10)
        self.outputs = samp.sample_set(4)
        self.lam_domain = np.zeros((10, 2))
        self.lam_domain[:, 0] = 0.0
        self.lam_domain[:, 1] = 1.0
        self.inputs.set_domain(self.lam_domain)
        self.inputs = bsam.random_sample_set('r',
                self.inputs.get_domain(), num_samples=101, globalize=True)
        self.outputs.set_values(np.dot(self.inputs._values, rnd.rand(10, 4)))
        Q_ref = np.mean(self.outputs._values, axis=0)
        self.inputs_emulated = bsam.random_sample_set('r',
                self.inputs.get_domain(), num_samples=1001, globalize=True)
        self.output_prob = simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
            self.outputs, Q_ref = Q_ref, rect_scale=0.2, center_pts_per_edge=1)
        self.disc = samp.discretization(input_sample_set=self.inputs,
                                        output_sample_set=self.outputs,
                                        output_probability_set=self.output_prob,
                                        emulated_input_sample_set=self.inputs_emulated)

    @unittest.skip("No reference data")
    def test_P_matches_true(self):
        pass

class Test_prob_10to4(TestProbMethod_10to4, prob):
    """
    Test :meth:`bet.calculateP.calculateP.prob` on a 10 to 4 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_10to4, self).setUp()
        self.disc._input_sample_set.estimate_volume_mc()
        calcP.prob(self.disc)


class Test_prob_on_emulated_samples_10to4(TestProbMethod_10to4, prob_on_emulated_samples):
    """
    Test :meth:`bet.calculateP.calculateP.prob_on_emulated_samples` on a 10 to 4 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_on_emulated_samples_10to4, self).setUp()

        calcP.prob_on_emulated_samples(self.disc)
        
class Test_prob_with_emulated_volumes_10to4(TestProbMethod_10to4, prob_with_emulated_volumes):
    """
    Test :meth:`bet.calculateP.calculateP.prob_with_emulated_volumes` on a 10 to 4 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_with_emulated_volumes_10to4, self).setUp()
        calcP.prob_with_emulated_volumes(self.disc)


class TestProbMethod_1to1(unittest.TestCase):
    """
    Sets up 1 to 1 map problem. Uses vectors instead of 2D arrays.
    """
    def setUp(self):
        """
        Set up problem.
        """

        import numpy.random as rnd
        rnd.seed(1)
        self.inputs = samp.sample_set(1)
        self.outputs = samp.sample_set(1)
        self.lam_domain = np.zeros((1, 2))
        self.lam_domain[:, 0] = 0.0
        self.lam_domain[:, 1] = 1.0
        self.inputs.set_domain(self.lam_domain)
        self.inputs.set_values(rnd.rand(100,))
        self.num_l_emulate = 1001
        self.inputs = bsam.random_sample_set('r',
                self.inputs.get_domain(), num_samples=1001, globalize=True)
        self.outputs.set_values(2.0*self.inputs._values)
        Q_ref = np.mean(self.outputs._values, axis=0)
        self.inputs_emulated = bsam.random_sample_set('r',
                self.inputs.get_domain(), num_samples=self.num_l_emulate,
                globalize=True) 
        self.output_prob = simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
            self.outputs, Q_ref = Q_ref, rect_scale=0.2, center_pts_per_edge=1)
        self.disc = samp.discretization(input_sample_set=self.inputs,
                                        output_sample_set=self.outputs,
                                        output_probability_set=self.output_prob,
                                        emulated_input_sample_set=self.inputs_emulated)
   
    @unittest.skip("No reference data")
    def test_P_matches_true(self):
        pass

class Test_prob_1to1(TestProbMethod_1to1, prob):
    """
    Test :meth:`bet.calculateP.calculateP.prob` on a 1 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_1to1, self).setUp()
        self.disc._input_sample_set.estimate_volume_mc()
        calcP.prob(self.disc)


class Test_prob_on_emulated_samples_1to1(TestProbMethod_1to1, prob_on_emulated_samples):
    """
    Test :meth:`bet.calculateP.calculateP.prob_on_emulated_samples` on a 1 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_on_emulated_samples_1to1, self).setUp()
        calcP.prob_on_emulated_samples(self.disc)


class Test_prob_with_emulated_volumes_1to1(TestProbMethod_1to1, prob_with_emulated_volumes):
    """
    Test :meth:`bet.calculateP.calculateP.prob_with_emulated_volumes` on a 1 to 1 map.
    """
    def setUp(self):
        """
        Set up problem.
        """
        super(Test_prob_with_emulated_volumes_1to1, self).setUp()
        calcP.prob_with_emulated_volumes(self.disc)

class Test_prob_from_sample_set(unittest.TestCase):
    """
    Test: method: `bet.calculateP.prob_from_sample_set`, 
    : method: `bet.calculateP.prob_from_sample_set_with_emulated_volumes`,
    and : method: `bet.calculateP.prob_from_discretization_input` 
    on a 2D domain.
    """
    def setUp(self):
        self.set_new = samp.rectangle_sample_set(dim=2)
        self.set_new.set_domain(np.array([[0.0, 1.0], [0.0, 1.0]]))
        self.set_new.setup(maxes=[[0.75, 0.75]], mins=[[0.25, 0.25]])
        
        self.set_old = samp.cartesian_sample_set(dim=2)
        self.set_old.set_domain(np.array([[0.0, 1.0], [0.0, 1.0]]))
        self.set_old.setup([np.linspace(0,1,21), np.linspace(0,1,21)])
        num_old = self.set_old.check_num()
        probs = np.zeros((num_old,))
        probs[0:-1] = 1.0/float(num_old-1)
        self.set_old.set_probabilities(probs)
        
        self.set_em = samp.cartesian_sample_set(dim=2)
        self.set_em.set_domain(np.array([[0.0, 1.0], [0.0, 1.0]]))
        self.set_em.setup([np.linspace(0,1,101), np.linspace(0,1,101)])
        

    def test_methods(self):
        calcP.prob_from_sample_set_with_emulated_volumes(self.set_old, self.set_new, self.set_em)
        nptest.assert_almost_equal(self.set_new._probabilities, [0.25, 0.75])
        calcP.prob_from_sample_set(self.set_old, self.set_new)
        nptest.assert_almost_equal(self.set_new._probabilities, [0.25, 0.75])
        disc = samp.discretization(input_sample_set=self.set_old,
                                   output_sample_set=self.set_old)
        calcP.prob_from_discretization_input(disc, self.set_new)
        nptest.assert_almost_equal(self.set_new._probabilities, [0.25, 0.75])
        num_em = self.set_em.check_num()
        probs = np.zeros((num_em,))
        probs[0:-1] = 1.0/float(num_em-1)
        self.set_em.set_probabilities(probs)
        self.set_em.global_to_local()
        disc = samp.discretization(input_sample_set=self.set_old,
                                   output_sample_set=self.set_old,
                                   emulated_input_sample_set=self.set_em)
        calcP.prob_from_discretization_input(disc, self.set_new)
        nptest.assert_almost_equal(self.set_new._probabilities, [0.25, 0.75])

        
