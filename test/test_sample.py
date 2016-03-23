# Copyright (C) 2016 The BET Development TEam

# Steve Mattis 03/23/2016

import unittest
import numpy as np
import numpy.testing as nptest
import bet.sample as sample
import bet.util as util
from bet.Comm import comm, MPI


    
class Test_sample_set(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.num = 100
        self.values = np.ones((self.num, self.dim))
        self.sam_set = sample.sample_set(dim=self.dim)
        self.sam_set.set_values(self.values)
    def test_check_dim(self):
        """
        Check set_dim
        """
        self.assertEqual(self.dim, self.sam_set.get_dim())
    def test_set_values(self):
        """
        Check set_values.
        """
        values = np.ones((150, self.dim))
        self.sam_set.set_values(values)
        nptest.assert_array_equal(util.fix_dimensions_data(values), self.sam_set.get_values())
    def test_get_values(self):
        """
        Check get_samples.
        """
        nptest.assert_array_equal(util.fix_dimensions_data(self.values), self.sam_set.get_values())
    def test_append_values(self):
        """
        Check appending of values.
        """
        new_values = np.zeros((10, self.dim))
        self.sam_set.append_values(new_values)
        nptest.assert_array_equal(util.fix_dimensions_data(new_values), self.sam_set.get_values()[self.num::,:])
    def test_get_dim(self):
        """
        Check to see if dimensions are correct.
        """
        self.assertEqual(self.dim, self.sam_set.get_dim())
    def test_probabilities(self):
        """
        Check probability methods
        """
        prob = 1.0/float(self.num)*np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        self.sam_set.check_num()
        nptest.assert_array_equal(prob, self.sam_set.get_probabilities())
    def test_volumes(self):
        """
        Check volume methods
        """
        vol = 1.0/float(self.num)*np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        self.sam_set.check_num()
        nptest.assert_array_equal(vol, self.sam_set.get_volumes())
        
    def test_error_estimates(self):
        """
        Check error estimate methods
        """
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        self.sam_set.check_num()
        nptest.assert_array_equal(ee, self.sam_set.get_error_estimates())

    def test_jacobian_methods(self):
        """
        Check jacobian methods.
        """
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.check_num()
        nptest.assert_array_equal(jac, self.sam_set.get_jacobians())

    def test_check_num(self):
        """
        Check check_num.
        """
        prob = 1.0/float(self.num)*np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0/float(self.num)*np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        num = self.sam_set.check_num()
        self.assertEqual(self.num, num)
        new_values = np.zeros((10, self.dim))
        self.sam_set.append_values(new_values)
        self.assertRaises(sample.length_not_matching,self.sam_set.check_num)

    def test_kd_tree(self):
        """
        Check features of the KD Tree
        """
        self.sam_set.set_kdtree()
        self.sam_set.get_kdtree()
        
    def test_parallel_features(self):
        """
        Check parallel features.
        """
        prob = 1.0/float(self.num)*np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0/float(self.num)*np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.assertNotEqual(self.sam_set._values_local, None)
        if comm.size > 1 :
            for array_name in self.sam_set._array_names:
                current_array = getattr(self.sam_set, array_name+"_local")
                if current_array is not None:
                    self.assertGreater(getattr(self.sam_set, array_name).shape[0], current_array.shape[0]) 
                    local_size = current_array.shape[0]
                    num = comm.allreduce(local_size, op=MPI.SUM)
                    self.assertEqual(num, self.num)
                    current_array_global = util.get_global_values(current_array)
                    nptest.assert_array_equal(getattr(self.sam_set, array_name), current_array_global)
        else:
            for array_name in self.sam_set._array_names:
                current_array = getattr(self.sam_set, array_name+"_local")
                if current_array is not None:
                    nptest.assert_array_equal(getattr(self.sam_set, array_name), current_array)
                    
        for array_name in self.sam_set._array_names:
            current_array = getattr(self.sam_set, array_name)
            if current_array is not None:
                setattr(self.sam_set, array_name + "_old", current_array)
                current_array = None
        self.sam_set.local_to_global()
        for array_name in self.sam_set._array_names:
            current_array = getattr(self.sam_set, array_name + "_local")
            if current_array is not None:
                nptest.assert_array_equal(getattr(self.sam_set, array_name),
                                          getattr(self.sam_set, array_name + "_old"))
                
                                    
class Test_sample_set_1d(Test_sample_set):
    def setUp(self):
        self.dim = 1
        self.num = 100
        self.values = np.ones((self.num, self.dim))
        self.sam_set = sample.sample_set(dim=self.dim)
        self.sam_set.set_values(self.values)
