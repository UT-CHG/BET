# Copyright (C) 2016 The BET Development TEam

# Steve Mattis 03/23/2016

import unittest, os
import numpy as np
import numpy.testing as nptest
import bet
import bet.sample as sample
import bet.util as util
from bet.Comm import comm, MPI

local_path = os.path.join(os.path.dirname(bet.__file__), "../test")
    
class Test_sample_set(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.num = 100
        self.values = np.ones((self.num, self.dim))
        self.sam_set = sample.sample_set(dim=self.dim)
        self.sam_set.set_values(self.values)
        self.domain = np.array([[0, 1],[0, 1]], dtype=np.float)
    def test_set_domain(self):
        """
        Test set domain.
        """
        self.sam_set.set_domain(self.domain)
        nptest.assert_array_equal(self.sam_set._domain, self.domain)
    def test_get_domain(self):
        """
        Test get domain.
        """
        self.sam_set.set_domain(self.domain)
        nptest.assert_array_equal(self.sam_set.get_domain(), self.domain)
    def test_save_load(self):
        """
        Check save_sample_set and load_sample_set.
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
        self.sam_set.set_domain(self.domain)
        self.sam_set.update_bounds()
        self.sam_set.update_bounds_local()

        sample.save_sample_set(self.sam_set, os.path.join(local_path, 
            'testfile.mat'), "TEST")

        loaded_set = sample.load_sample_set(os.path.join(local_path, 
            'testfile.mat'), "TEST")
        loaded_set_none = sample.load_sample_set(os.path.join(local_path, 
            'testfile.mat'))

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names+sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print attrname
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                        curr_attr)

        if os.path.exists(os.path.join(local_path, 'testfile.mat')):
            os.remove(os.path.join(local_path, 'testfile.mat'))
    def test_copy(self):
        """
        Check save_sample_set and load_sample_set.
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
        self.sam_set.set_domain(self.domain)
        self.sam_set.update_bounds()
        self.sam_set.update_bounds_local()
        self.sam_set.set_kdtree()

        copied_set = self.sam_set.copy()
        for attrname in sample.sample_set.vector_names+sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(copied_set, attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                        curr_attr)

        assert copied_set._kdtree is not None
    def test_update_bounds(self):
        """
        Check update_bounds
        """
        self.sam_set.set_domain(self.domain)
        self.sam_set.update_bounds()
        nptest.assert_array_equal(self.sam_set._left,
            np.repeat([self.domain[:, 0]], self.num, 0))
        nptest.assert_array_equal(self.sam_set._right,
            np.repeat([self.domain[:, 1]], self.num, 0))
        nptest.assert_array_equal(self.sam_set._width,
            np.repeat([self.domain[:, 1] - self.domain[:, 0]], self.num, 0))
        o_num = 35
        self.sam_set.update_bounds(o_num)
        nptest.assert_array_equal(self.sam_set._left,
            np.repeat([self.domain[:, 0]], o_num, 0))
        nptest.assert_array_equal(self.sam_set._right,
            np.repeat([self.domain[:, 1]], o_num, 0))
        nptest.assert_array_equal(self.sam_set._width,
            np.repeat([self.domain[:, 1] - self.domain[:, 0]], o_num, 0))
    def test_update_bounds_local(self):
        """
        Check update_bounds_local
        """
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)
        self.sam_set.update_bounds_local()
        local_size = self.sam_set.get_values_local().shape[0]
        nptest.assert_array_equal(self.sam_set._left_local,
            np.repeat([self.domain[:, 0]], local_size, 0))
        nptest.assert_array_equal(self.sam_set._right_local,
            np.repeat([self.domain[:, 1]], local_size, 0))
        nptest.assert_array_equal(self.sam_set._width_local,
            np.repeat([self.domain[:, 1] - self.domain[:, 0]], local_size,
                0))
        o_num = 35
        self.sam_set.update_bounds_local(o_num)
        nptest.assert_array_equal(self.sam_set._left_local,
            np.repeat([self.domain[:, 0]], o_num, 0))
        nptest.assert_array_equal(self.sam_set._right_local,
            np.repeat([self.domain[:, 1]], o_num, 0))
        nptest.assert_array_equal(self.sam_set._width_local,
            np.repeat([self.domain[:, 1] - self.domain[:, 0]], o_num, 0))

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
        nptest.assert_array_equal(util.fix_dimensions_data(values),
            self.sam_set.get_values())
    def test_set_values_local(self):
        """
        Check set_values_local.
        """
        values = np.ones((15, self.dim))
        self.sam_set.set_values_local(values)
        nptest.assert_array_equal(util.fix_dimensions_data(values),
            self.sam_set.get_values_local()) 
    def test_get_values(self):
        """
        Check get_samples.
        """
        nptest.assert_array_equal(util.fix_dimensions_data(self.values),
            self.sam_set.get_values()) 
    def test_get_shape(self):
        """
        Check get_samples.
        """
        nptest.assert_array_equal(util.fix_dimensions_data(self.values).shape,
                self.sam_set.shape())
    def test_append_values(self):
        """
        Check appending of values.
        """
        new_values = np.zeros((10, self.dim))
        self.sam_set.append_values(new_values)
        nptest.assert_array_equal(util.fix_dimensions_data(new_values),
            self.sam_set.get_values()[self.num::, :]) 
    def test_append_values_local(self):
        """
        Check appending of local values.
        """
        new_values = np.zeros((10, self.dim))
        self.sam_set.global_to_local()
        self.sam_set.append_values_local(new_values)
        nptest.assert_array_equal(util.fix_dimensions_data(new_values),
                self.sam_set.get_values_local()[self.num::, :])
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
        self.assertRaises(sample.length_not_matching, self.sam_set.check_num)

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
        if comm.size > 1:
            for array_name in sample.sample_set.array_names:
                current_array = getattr(self.sam_set, array_name+"_local")
                if current_array is not None:
                    self.assertGreater(getattr(self.sam_set,
                        array_name).shape[0], current_array.shape[0])
                    local_size = current_array.shape[0]
                    num = comm.allreduce(local_size, op=MPI.SUM)
                    self.assertEqual(num, self.num)
                    current_array_global = util.get_global_values(current_array)
                    nptest.assert_array_equal(getattr(self.sam_set,
                        array_name), current_array_global) 
                    if array_name is "_values":
                        assert self.sam_set.shape_local() == (local_size,
                                self.dim)
        else:
            for array_name in sample.sample_set.array_names:
                current_array = getattr(self.sam_set, array_name+"_local")
                if current_array is not None:
                    nptest.assert_array_equal(getattr(self.sam_set,
                        array_name), current_array)
                    if array_name is "_values":
                        assert self.sam_set.shape_local() == (self.num,
                                self.dim)
                    
        for array_name in sample.sample_set.array_names:
            current_array = getattr(self.sam_set, array_name)
            if current_array is not None:
                setattr(self.sam_set, array_name + "_old", current_array)
                current_array = None
        self.sam_set.local_to_global()
        for array_name in sample.sample_set.array_names:
            current_array = getattr(self.sam_set, array_name + "_local")
            if current_array is not None:
                nptest.assert_array_equal(getattr(self.sam_set, array_name),
                                          getattr(self.sam_set, array_name +
                                              "_old")) 
    def test_domain(self):
        """
        Test domain information.
        """
        domain = np.ones((self.dim, 2))
        self.sam_set.set_domain(domain)
        nptest.assert_array_equal(domain, self.sam_set.get_domain())
                
                                    
class Test_sample_set_1d(Test_sample_set):
    def setUp(self):
        self.dim = 1
        self.num = 100
        self.values = np.ones((self.num, self.dim))
        self.sam_set = sample.sample_set(dim=self.dim)
        self.sam_set.set_values(self.values)
        self.domain = np.array([[0, 1]], dtype=np.float)

class Test_discretization_simple(unittest.TestCase):
    def setUp(self):
        self.dim1 = 3
        self.num = 100
        self.dim2 = 1
        values1 = np.ones((self.num, self.dim1))
        values2 = np.ones((self.num, self.dim2))
        values3 = np.ones((self.num, self.dim2))
        self.input = sample.sample_set(dim=self.dim1)
        self.output = sample.sample_set(dim=self.dim2)
        self.output_probability_set = sample.sample_set(dim=self.dim2)
        self.input.set_values(values1)
        self.output.set_values(values2)
        self.output_probability_set.set_values(values3)
        self.disc = sample.discretization(input_sample_set=self.input,
                                          output_sample_set=self.output,
                                          output_probability_set=\
                                                  self.output_probability_set)
        
    def Test_check_nums(self):
        """
        Test number checking.
        """
        num = self.disc.check_nums()
        self.assertEqual(num, self.num)

    def Test_set_io_ptr(self):
        """
        Test setting io ptr
        """
        #TODO be careful if we change Kdtree
        self.disc.set_io_ptr(globalize=True)
        self.disc.get_io_ptr()
        self.disc.set_io_ptr(globalize=False)
        self.disc.get_io_ptr()

    def Test_set_emulated_ii_ptr(self):
        """
        Test setting emulated ii ptr
        """
        #TODO be careful if we change Kdtree
        values = np.ones((10, self.dim1))
        self.emulated = sample.sample_set(dim=self.dim1)
        self.emulated.set_values(values)
        self.disc._emulated_input_sample_set = self.emulated
        self.disc.set_emulated_ii_ptr(globalize=True)
        self.disc.get_emulated_ii_ptr()
        self.disc.set_emulated_ii_ptr(globalize=False)
        self.disc.get_emulated_ii_ptr()
        
    def Test_set_emulated_oo_ptr(self):
        """
        Test setting emulated oo ptr
        """
        #TODO be careful if we change Kdtree
        values = np.ones((10, self.dim2))
        self.emulated = sample.sample_set(dim=self.dim2)
        self.emulated.set_values(values)
        self.disc._emulated_output_sample_set = self.emulated
        self.disc.set_emulated_oo_ptr(globalize=True)
        self.disc.get_emulated_oo_ptr()
        self.disc.set_emulated_oo_ptr(globalize=False)
        self.disc.get_emulated_oo_ptr()

    def Test_save_load_discretization(self):
        """
        Test saving and loading of discretization
        """
        sample.save_discretization(self.disc, os.path.join(local_path, 
            'testfile.mat'), "TEST")

        loaded_disc = sample.load_discretization(os.path.join(local_path, 
            'testfile.mat'), "TEST")

        for attrname in sample.discretization.vector_names:
            curr_attr = getattr(loaded_disc, attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(curr_attr, getattr(self.disc,
                    attrname))

        for attrname in sample.discretization.sample_set_names:
            curr_set = getattr(loaded_disc, attrname)
            if curr_set is not None:
                for set_attrname in sample.sample_set.vector_names+\
                        sample.sample_set.all_ndarray_names:
                    curr_attr = getattr(curr_set, set_attrname)
                    if curr_attr is not None:
                        nptest.assert_array_equal(curr_attr, getattr(\
                                curr_set, set_attrname))

        if os.path.exists(os.path.join(local_path, 'testfile.mat')):
            os.remove(os.path.join(local_path, 'testfile.mat'))

    def Test_copy_discretization(self):
        """
        Test copying of discretization
        """
        copied_disc = self.disc.copy() 

        for attrname in sample.discretization.vector_names:
            curr_attr = getattr(copied_disc, attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(curr_attr, getattr(self.disc,
                    attrname))

        for attrname in sample.discretization.sample_set_names:
            curr_set = getattr(copied_disc, attrname)
            if curr_set is not None:
                for set_attrname in sample.sample_set.vector_names+\
                        sample.sample_set.all_ndarray_names:
                    curr_attr = getattr(curr_set, set_attrname)
                    if curr_attr is not None:
                        nptest.assert_array_equal(curr_attr, getattr(\
                                curr_set, set_attrname))
