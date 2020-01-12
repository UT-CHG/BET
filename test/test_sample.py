# Copyright (C) 2014-2019 The BET Development Team

import unittest
import os
import glob
import numpy as np
import numpy.testing as nptest
import bet
import bet.sample as sample
import bet.util as util
import bet.sampling.basicSampling as bsam
from bet.Comm import comm, MPI
import scipy.stats.distributions as dist

#local_path = os.path.join(os.path.dirname(bet.__file__), "/test")
local_path = ''


class Test_sample_set(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.num = 100
        self.values = np.ones((self.num, self.dim))
        self.sam_set = sample.sample_set(dim=self.dim)
        self.sam_set.set_values(self.values)
        self.domain = np.array([[0, 1], [0, 1]], dtype=np.float)

    def test_merge(self):
        """
        Test merge.
        """
        other_set = self.sam_set.copy()
        merge_set = self.sam_set.merge(other_set)
        nptest.assert_array_equal(self.sam_set._domain, merge_set._domain)
        nptest.assert_array_equal(self.sam_set._values,
                                  merge_set._values[0:self.num, :])

    def test_normalize(self):
        """
        Test normalize and undo normalize domain.
        """
        domain = 5.0 * self.domain - 1.0
        self.sam_set.set_domain(domain)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)

        self.sam_set.normalize_domain()
        nptest.assert_array_equal(self.sam_set._domain, self.domain)
        nptest.assert_array_almost_equal(self.sam_set._values, 0.4)
        nptest.assert_array_almost_equal(self.sam_set._error_estimates, 0.4)
        nptest.assert_array_almost_equal(self.sam_set._jacobians, 5.0)

        self.sam_set.undo_normalize_domain()
        nptest.assert_array_equal(self.sam_set._domain, domain)
        nptest.assert_array_almost_equal(self.sam_set._values, 1.0)
        nptest.assert_array_almost_equal(self.sam_set._error_estimates, 1.0)
        nptest.assert_array_almost_equal(self.sam_set._jacobians, 1.0)

    def test_clip(self):
        """
        Test clipping of sample set.
        """
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)

        cnum = int(0.5 * self.num)
        sam_set_clipped = self.sam_set.clip(cnum)

        num = sam_set_clipped.check_num()
        self.assertEqual(num, cnum)
        nptest.assert_array_equal(self.sam_set._values[0:cnum, :],
                                  sam_set_clipped._values)
        nptest.assert_array_equal(self.sam_set._error_estimates[0:cnum, :],
                                  sam_set_clipped._error_estimates)
        nptest.assert_array_equal(self.sam_set._jacobians[0:cnum, :],
                                  sam_set_clipped._jacobians)

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
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)
        self.sam_set.update_bounds()
        self.sam_set.update_bounds_local()

        file_name = os.path.join(local_path, 'testfile.mat')
        globalize = True
        sample.save_sample_set(self.sam_set, file_name, "TEST", globalize)
        comm.barrier()

        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_set = sample.load_sample_set(local_file_name, "TEST")
        loaded_set_none = sample.load_sample_set(local_file_name)

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print(attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)

        comm.barrier()

        file_name = os.path.join(local_path, 'testfile.mat')
        globalize = False
        sample.save_sample_set(self.sam_set, file_name, "TEST", globalize)
        comm.barrier()

        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_set = sample.load_sample_set(local_file_name, "TEST")
        loaded_set_none = sample.load_sample_set(local_file_name)

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print(attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)

    def test_copy(self):
        """
        Check copy.
        """
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num) * np.ones((self.num,))
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
        for attrname in sample.sample_set.vector_names + sample.sample_set.\
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
                                  np.repeat([self.domain[:, 1] -
                                             self.domain[:, 0]], self.num, 0))
        o_num = 35
        self.sam_set.update_bounds(o_num)
        nptest.assert_array_equal(self.sam_set._left,
                                  np.repeat([self.domain[:, 0]], o_num, 0))
        nptest.assert_array_equal(self.sam_set._right,
                                  np.repeat([self.domain[:, 1]], o_num, 0))
        nptest.assert_array_equal(self.sam_set._width,
                                  np.repeat([self.domain[:, 1] -
                                             self.domain[:, 0]], o_num, 0))

    def test_update_bounds_local(self):
        """
        Check update_bounds_local
        """
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)
        self.sam_set.update_bounds_local()
        local_size = self.sam_set.get_values_local().shape[0]
        nptest.assert_array_equal(self.sam_set._left_local,
                                  np.repeat([self.domain[:, 0]],
                                            local_size, 0))
        nptest.assert_array_equal(self.sam_set._right_local,
                                  np.repeat([self.domain[:, 1]],
                                            local_size, 0))
        nptest.assert_array_equal(self.sam_set._width_local,
                                  np.repeat([self.domain[:, 1] -
                                             self.domain[:, 0]],
                                            local_size, 0))
        o_num = 35
        self.sam_set.update_bounds_local(o_num)
        nptest.assert_array_equal(self.sam_set._left_local,
                                  np.repeat([self.domain[:, 0]], o_num, 0))
        nptest.assert_array_equal(self.sam_set._right_local,
                                  np.repeat([self.domain[:, 1]], o_num, 0))
        nptest.assert_array_equal(self.sam_set._width_local,
                                  np.repeat([self.domain[:, 1] -
                                             self.domain[:, 0]], o_num, 0))

    def test_check_dim(self):
        """
        Check set_dim.
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

        local_size = self.sam_set.get_values_local().shape[0]
        self.sam_set.append_values_local(new_values)
        nptest.assert_array_equal(util.fix_dimensions_data(new_values),
                                  self.sam_set.
                                  get_values_local()[local_size::, :])

    def test_get_dim(self):
        """
        Check to see if dimensions are correct.
        """
        self.assertEqual(self.dim, self.sam_set.get_dim())

    def test_probabilities(self):
        """
        Check probability methods.
        """
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        self.sam_set.check_num()
        nptest.assert_array_equal(prob, self.sam_set.get_probabilities())

    def test_densities(self):
        """
        Check density methods.
        """
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        self.sam_set.estimate_volume_mc()
        self.sam_set.set_densities()
        self.sam_set.check_num()
        vol = self.sam_set.get_volumes()
        nptest.assert_array_equal(prob / vol, self.sam_set.get_densities())
        den = np.ones((self.num,))
        self.sam_set.set_densities(den)
        nptest.assert_array_equal(den, self.sam_set.get_densities())

    def test_volumes(self):
        """
        Check volume methods.
        """
        vol = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        self.sam_set.check_num()
        nptest.assert_array_equal(vol, self.sam_set.get_volumes())

    def test_error_estimates(self):
        """
        Check error estimate methods.
        """
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        self.sam_set.check_num()
        nptest.assert_array_equal(ee, self.sam_set.get_error_estimates())

    def test_region(self):
        """
        Check region methods.
        """
        region = np.ones((self.num,), dtype=np.int)
        self.sam_set.set_region(region)
        self.sam_set.check_num()
        nptest.assert_array_equal(region, self.sam_set.get_region())

    def test_error_id(self):
        """
        Check error identifier methods.
        """
        error_id = np.ones((self.num,))
        self.sam_set.set_error_id(error_id)
        self.sam_set.check_num()
        nptest.assert_array_equal(error_id, self.sam_set.get_error_id())

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
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num) * np.ones((self.num,))
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
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.assertNotIn(None, self.sam_set._values_local)
        if comm.size > 1:
            for array_name in sample.sample_set.array_names:
                current_array = getattr(self.sam_set, array_name + "_local")
                if current_array is not None:
                    self.assertGreater(getattr(self.sam_set,
                                               array_name).shape[0],
                                       current_array.shape[0])
                    local_size = current_array.shape[0]
                    num = comm.allreduce(local_size, op=MPI.SUM)
                    self.assertEqual(num, self.num)
                    current_array_global = util.get_global_values(
                        current_array)
                    nptest.assert_array_equal(getattr(self.sam_set,
                                                      array_name),
                                              current_array_global)
                    if array_name is "_values":
                        assert self.sam_set.shape_local() == (local_size,
                                                              self.dim)
        else:
            for array_name in sample.sample_set.array_names:
                current_array = getattr(self.sam_set, array_name + "_local")
                if current_array is not None:
                    nptest.assert_array_equal(getattr(self.sam_set,
                                                      array_name),
                                              current_array)
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

    def test_distribution_and_domain(self):
        """
        Test set/get distribution and domain inference.
        """
        domain = np.array([[0, 1] for _ in range(self.dim)])
        import scipy
        from scipy.stats.distributions import uniform

        # default behavior is unit hypercube
        self.sam_set.set_distribution()
        assert isinstance(self.sam_set.get_distribution(),
                          scipy.stats.distributions.rv_frozen)

        # set frozen - should be able to interpret dimension.
        dist = uniform(loc=0, scale=1)
        self.sam_set.set_distribution(dist)
        assert isinstance(self.sam_set.get_distribution(),
                          scipy.stats.distributions.rv_frozen)

        # ensure domain was properly inferred
        nptest.assert_array_equal(domain, self.sam_set.get_domain())
        # set continuous and make sure it gets instantiated
        self.sam_set.set_distribution(uniform)
        assert isinstance(self.sam_set.get_distribution(),
                          scipy.stats.distributions.rv_frozen)
        nptest.assert_array_equal(domain, self.sam_set.get_domain())

        # try another domain
        new = [1 for _ in range(self.dim)]
        self.sam_set.set_distribution(uniform(loc=1,
                                              scale=new))
        assert isinstance(self.sam_set.get_distribution(),
                          scipy.stats.distributions.rv_frozen)
        nptest.assert_array_equal(domain + 1,
                                  self.sam_set.get_domain())
        # try another domain and handler.
        self.sam_set.set_dist(uniform(loc=new,
                                      scale=0.5))
        assert isinstance(self.sam_set.get_dist(),
                          scipy.stats.distributions.rv_frozen)
        nptest.assert_array_equal(1 + domain * 0.5,
                                  self.sam_set.get_domain())

        # incorrect dimension should raise error
        try:
            self.sam_set.set_distribution(uniform(loc=0,
                                                  scale=[1, 1, 1, 1]))
        except sample.dim_not_matching:
            pass

    def test_generate_samples(self):
        """
        Test random sample generation.
        """
        # default to uniform hypercube
        self.sam_set.set_distribution()
        for num in [1, 10, 50, 100]:
            self.sam_set.generate_samples(num)
            assert self.sam_set.check_num() == num
        from scipy.stats.distributions import uniform
        # make sure keyword gets passed to continuous distribution.
        self.sam_set.set_distribution(uniform, loc=2)
        for num in [1, 10, 50, 100]:
            self.sam_set.generate_samples(num)
            assert self.sam_set.check_num() == num
        # frozen distribution
        self.sam_set.set_distribution(uniform(loc=2))
        for num in [1, 10, 50, 100]:
            self.sam_set.generate_samples(num)
            assert self.sam_set.check_num() == num

    def test_rvs(self):
        """
        Test ability to use rvs generation with correct shape.
        """
        # passing continuous distribution => should convert to frozen.
        from scipy.stats.distributions import uniform
        x = self.sam_set.rvs(dist=uniform)
        assert len(x) == 1
        from scipy.stats.distributions import norm
        x = self.sam_set.rvs(dist=norm, loc=1)
        assert len(x) == 1
        # now test setting using frozen distributions
        for num in [1, 10, 50, 100]:
            self.sam_set.set_values(self.sam_set.rvs(num, dist=norm()))
            assert self.sam_set.check_num() == num

    def test_pdf(self):
        """
        Test pdf capabilities
        """
        import scipy.stats.distributions as dists
        self.sam_set.set_dist(dists.norm)
        self.sam_set.generate_samples(100)
        x = self.sam_set.get_values()
        for dst in [dists.norm, dists.uniform]:
            y = self.sam_set.pdf(x=x, dist=dst(loc=0, scale=1))
            tru = dst.pdf(x, loc=0, scale=1).prod(axis=1)
            nptest.assert_array_equal(tru, y)

    def test_cdf(self):
        """
        Test cdf capabilities
        """
        import scipy.stats.distributions as dists
        self.sam_set.set_dist(dists.norm)
        self.sam_set.generate_samples(100)
        x = self.sam_set.get_values()
        for dst in [dists.norm, dists.uniform]:
            tru = dst.cdf(x, loc=1, scale=2).prod(axis=1)
            z = self.sam_set.cdf(x=x, dist=dst(loc=1, scale=2))
            nptest.assert_array_equal(tru, z)

    def test_estimate_probabilities_mc(self):
        """
        Test MC assumption in probabilities.
        """
        import scipy.stats.distributions as dists
        self.sam_set.set_dist(dists.norm)
        for num in [1, 10, 50, 100]:
            self.sam_set.generate_samples(num)
            self.sam_set.estimate_probabilities_mc()
            assert len(np.unique(self.sam_set.get_probabilities())) == 1
            assert self.sam_set.get_probabilities()[0] == 1.0 / num


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
        self.input_set = sample.sample_set(dim=self.dim1)
        self.output_set = sample.sample_set(dim=self.dim2)
        self.output_probability_set = sample.sample_set(dim=self.dim2)
        self.input_set.set_values(values1)
        self.output_set.set_values(values2)
        self.output_probability_set.set_values(values3)
        self.disc = sample.discretization(input_sample_set=self.input_set,
                                          output_sample_set=self.output_set,
                                          output_probability_set=self.
                                          output_probability_set)

    def test_check_nums(self):
        """
        Test number checking.
        """
        num = self.disc.check_nums()
        self.assertEqual(num, self.num)

    def test_clip(self):
        """
        Test clipping of discretization.
        """
        cnum = int(0.5 * self.num)
        disc_clipped = self.disc.clip(cnum)
        nptest.assert_array_equal(self.disc._input_sample_set.
                                  _values[0:cnum, :],
                                  disc_clipped._input_sample_set._values)
        nptest.assert_array_equal(self.disc._output_sample_set.
                                  _values[0:cnum, :],
                                  disc_clipped._output_sample_set._values)

    def test_slicing(self):
        """
        Test `bet.sample.discretization.choose_inputs_outputs`
        """
        self.disc._output_sample_set.set_error_estimates(
            np.ones((self.num, self.dim2)))
        self.disc._input_sample_set.set_jacobians(
            np.ones((self.num, self.dim2, self.dim1)))
        self.disc._input_sample_set.set_reference_value(
            np.random.rand(self.dim1))
        self.disc._output_sample_set.set_reference_value(
            np.random.rand(self.dim2))
        self.disc._output_sample_set.set_domain(
            np.sort(np.random.rand(self.dim2, 2), axis=1))
        self.disc._input_sample_set.set_domain(
            np.sort(np.random.rand(self.dim1, 2), axis=1))

        disc_new = self.disc.choose_inputs_outputs(inputs=[0, 2], outputs=[0])
        nptest.assert_array_equal(self.disc._input_sample_set.
                                  _values[:, [0, 2]],
                                  disc_new._input_sample_set._values)
        nptest.assert_array_equal(self.disc._output_sample_set.
                                  _values[:, [0]],
                                  disc_new._output_sample_set._values)
        nptest.assert_array_equal(self.disc._output_sample_set.
                                  _error_estimates[:, [0]],
                                  disc_new._output_sample_set.
                                  _error_estimates)
        self.assertEqual(
            disc_new._input_sample_set._jacobians.shape, (self.num, 1, 2))
        nptest.assert_array_equal(
            disc_new._input_sample_set._reference_value,
            self.disc._input_sample_set._reference_value[[0, 2]])
        nptest.assert_array_equal(
            disc_new._output_sample_set._reference_value,
            self.disc._output_sample_set._reference_value[0])
        nptest.assert_array_equal(
            disc_new._input_sample_set._domain,
            self.disc._input_sample_set._domain[[0, 2], :])
        nptest.assert_array_equal(
            disc_new._output_sample_set._domain,
            self.disc._output_sample_set._domain[[0], :])

    def test_choose_outputs(self):
        """
        Test `bet.sample.discretization.choose_outputs`
        """
        self.disc._output_sample_set.set_domain(
            np.sort(np.random.rand(self.dim2, 2), axis=1))
        self.disc._output_sample_set.set_reference_value(
            np.random.rand(self.dim2))
        self.disc._output_sample_set.set_error_estimates(
            np.ones((self.num, self.dim2)))
        self.disc._input_sample_set.set_jacobians(
            np.ones((self.num, self.dim2, self.dim1)))

        disc_new = self.disc.choose_outputs(outputs=[0])
        nptest.assert_array_equal(self.disc._input_sample_set.
                                  _values,
                                  disc_new._input_sample_set._values)
        nptest.assert_array_equal(self.disc._output_sample_set.
                                  _values[:, [0]],
                                  disc_new._output_sample_set._values)
        nptest.assert_array_equal(self.disc._output_sample_set.
                                  _error_estimates[:, [0]],
                                  disc_new._output_sample_set.
                                  _error_estimates)
        nptest.assert_array_equal(
            disc_new._output_sample_set._domain,
            self.disc._output_sample_set._domain[[0], :])
        nptest.assert_array_equal(
            disc_new._output_sample_set._reference_value,
            self.disc._output_sample_set._reference_value[0])
        nptest.assert_array_equal(
            disc_new._input_sample_set._jacobians,
            self.disc._input_sample_set._jacobians)

    def test_set_io_ptr(self):
        """
        Test setting io ptr
        """
        # TODO be careful if we change Kdtree
        self.disc.set_io_ptr(globalize=True)
        self.disc.get_io_ptr()
        self.disc.set_io_ptr(globalize=False)
        self.disc.get_io_ptr()
        self.disc.globalize_ptrs()

    def test_set_emulated_ii_ptr(self):
        """
        Test setting emulated ii ptr
        """
        # TODO be careful if we change Kdtree
        values = np.ones((10, self.dim1))
        self.emulated = sample.sample_set(dim=self.dim1)
        self.emulated.set_values(values)
        self.disc._emulated_input_sample_set = self.emulated
        self.disc.set_emulated_ii_ptr(globalize=True)
        self.disc.get_emulated_ii_ptr()
        self.disc.set_emulated_ii_ptr(globalize=False)
        self.disc._emulated_input_sample_set.local_to_global()
        self.disc.get_emulated_ii_ptr()
        self.disc.globalize_ptrs()

    def test_set_emulated_oo_ptr(self):
        """
        Test setting emulated oo ptr
        """
        # TODO be careful if we change Kdtree
        values = np.ones((3, self.dim2))
        self.emulated = sample.sample_set(dim=self.dim2)
        self.emulated.set_values(values)
        self.disc._emulated_output_sample_set = self.emulated
        self.disc.set_emulated_oo_ptr(globalize=True)
        self.disc.get_emulated_oo_ptr()
        self.disc.set_emulated_oo_ptr(globalize=False)
        self.disc.get_emulated_oo_ptr()
        self.disc.globalize_ptrs()

    def test_set_input_sample_set(self):
        """
        Test setting input sample set
        """
        test_set = sample.sample_set(dim=self.dim1)
        self.disc.set_input_sample_set(test_set)

    def test_get_input_sample_set(self):
        """
        Test getting input sample set
        """
        self.disc.get_input_sample_set()

    def test_set_emulated_input_sample_set(self):
        """
        Test setting emulated input sample set
        """
        test_set = sample.sample_set(dim=self.dim1)
        self.disc.set_emulated_input_sample_set(test_set)

    def test_get_emulated_input_sample_set(self):
        """
        Test getting emulated input sample set
        """
        self.disc.get_emulated_input_sample_set()

    def test_set_output_sample_set(self):
        """
        Test setting output sample set
        """
        test_set = sample.sample_set(dim=self.dim2)
        self.disc.set_output_sample_set(test_set)

    def test_get_output_sample_set(self):
        """
        Test getting output sample set
        """
        self.disc.get_output_sample_set()

    def test_set_output_probability_set(self):
        """
        Test setting output probability sample set
        """
        test_set = sample.sample_set(dim=self.dim2)
        self.disc.set_output_probability_set(test_set)
        # test with existing values
        test_set.set_values(np.random.rand(100, self.dim2))
        self.disc.set_output_probability_set(test_set)
        # test with emulated
        self.disc.set_emulated_output_sample_set(test_set)
        self.disc.set_output_probability_set(test_set)
        # test without output samples
        self.disc._output_sample_set = None
        self.disc._emulated_output_sample_set = None
        self.disc.set_output_probability_set(test_set)
        # test error-handling
        self.disc.set_output_sample_set(test_set)
        test_set = sample.sample_set(dim=self.dim2 + 1)
        try:  # catch first possible dim error
            self.disc.set_output_probability_set(test_set)
        except sample.dim_not_matching:
            pass
        try:  # catch second error
            test_set = sample.sample_set(dim=self.dim2 + 1)
            test_set.set_values(np.random.rand(100, self.dim2 + 1))
            test_set.global_to_local()
            self.disc.set_data_driven()
            self.disc.set_output_probability_set(test_set)
        except sample.dim_not_matching:
            pass

    def test_get_output_probability_set(self):
        """
        Test getting output probability sample set
        """
        self.disc.get_output_probability_set()

    def test_set_emulated_output_sample_set(self):
        """
        Test setting emulated output sample set
        """
        test_set = sample.sample_set(dim=self.dim2)
        self.disc.set_emulated_output_sample_set(test_set)
        # test without output samples
        self.disc._output_sample_set = None
        self.disc._output_probability_set = None
        self.disc._emulated_output_sample_set = None
        self.disc.set_emulated_output_sample_set(test_set)
        # test error-handling
        self.disc.set_output_sample_set(test_set)
        test_set = sample.sample_set(dim=self.dim2 + 1)
        try:  # catch first possible dim error
            self.disc.set_emulated_output_sample_set(test_set)
        except sample.dim_not_matching:
            pass

    def test_get_emulated_output_sample_set(self):
        """
        Test getting emulated output sample set
        """
        self.disc.get_emulated_output_sample_set()

    def test_save_load_discretization(self):
        """
        Test saving and loading of discretization
        """
        file_name = os.path.join(local_path, 'testfile.mat')
        globalize = True
        sample.save_discretization(self.disc, file_name, "TEST", globalize)
        comm.barrier()
        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_disc = sample.load_discretization(local_file_name, "TEST")

        for attrname in sample.discretization.vector_names:
            curr_attr = getattr(loaded_disc, attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(curr_attr, getattr(self.disc,
                                                             attrname))

        for attrname in sample.discretization.sample_set_names:
            curr_set = getattr(loaded_disc, attrname)
            if curr_set is not None:
                for set_attrname in sample.sample_set.vector_names +\
                        sample.sample_set.all_ndarray_names:
                    curr_attr = getattr(curr_set, set_attrname)
                    if curr_attr is not None:
                        nptest.assert_array_equal(curr_attr,
                                                  getattr(curr_set,
                                                          set_attrname))
        comm.barrier()

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)

        # run test with globalize=False
        globalize = False
        sample.save_discretization(self.disc, file_name, "TEST", globalize)
        comm.barrier()
        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_disc = sample.load_discretization(local_file_name, "TEST")

        for attrname in sample.discretization.vector_names:
            curr_attr = getattr(loaded_disc, attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(curr_attr,
                                          getattr(self.disc, attrname))

        for attrname in sample.discretization.sample_set_names:
            curr_set = getattr(loaded_disc, attrname)
            if curr_set is not None:
                for set_attrname in sample.sample_set.vector_names +\
                        sample.sample_set.all_ndarray_names:
                    curr_attr = getattr(curr_set, set_attrname)
                    if curr_attr is not None:
                        nptest.assert_array_equal(curr_attr,
                                                  getattr(curr_set,
                                                          set_attrname))
        comm.barrier()

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)

    def test_copy_discretization(self):
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
                for set_attrname in sample.sample_set.vector_names +\
                        sample.sample_set.all_ndarray_names:
                    curr_attr = getattr(curr_set, set_attrname)
                    if curr_attr is not None:
                        nptest.assert_array_equal(curr_attr, getattr(
                            curr_set, set_attrname))

    def test_estimate_input_volume_emulated(self):
        """

        Testing :meth:`bet.discretization.estimate_input_volume_emulated`

        """
        lam_left = np.array([0.0, .25, .4])
        lam_right = np.array([1.0, 4.0, .5])
        lam_width = lam_right - lam_left

        lam_domain = np.zeros((3, 2))
        lam_domain[:, 0] = lam_left
        lam_domain[:, 1] = lam_right

        num_samples_dim = 2
        start = lam_left + lam_width / (2 * num_samples_dim)
        stop = lam_right - lam_width / (2 * num_samples_dim)
        d1_arrays = []

        for l, r in zip(start, stop):
            d1_arrays.append(np.linspace(l, r, num_samples_dim))

        s_set = sample.sample_set(util.meshgrid_ndim(d1_arrays).shape[1])
        s_set.set_domain(lam_domain)
        s_set.set_values(util.meshgrid_ndim(d1_arrays))

        volume_exact = 1.0 / s_set._values.shape[0]

        emulated_samples = s_set.copy()
        emulated_samples.update_bounds_local(1001)
        emulated_samples.set_values_local(emulated_samples._width_local
                                          * np.random.random((1001,
                                                              emulated_samples.
                                                              get_dim())) +
                                          emulated_samples._left_local)

        self.disc.set_input_sample_set(s_set)
        self.disc.set_emulated_input_sample_set(emulated_samples)
        self.disc.estimate_input_volume_emulated()

        lam_vol = self.disc._input_sample_set._volumes

        # Check the dimension.
        nptest.assert_array_equal(lam_vol.shape, (len(s_set._values), ))

        # Check that the volumes are within a tolerance for a regular grid of
        # samples.
        nptest.assert_array_almost_equal(lam_vol, volume_exact, 1)
        nptest.assert_almost_equal(np.sum(lam_vol), 1.0)

    def test_estimate_output_volume_emulated(self):
        """

        Testing :meth:`bet.discretization.estimate_output_volume_emulated`

        """
        lam_left = np.array([0.0])
        lam_right = np.array([1.0])
        lam_width = lam_right - lam_left

        lam_domain = np.zeros((1, 2))
        lam_domain[:, 0] = lam_left
        lam_domain[:, 1] = lam_right

        num_samples_dim = 2
        start = lam_left + lam_width / (2 * num_samples_dim)
        stop = lam_right - lam_width / (2 * num_samples_dim)
        d1_arrays = []

        for l, r in zip(start, stop):
            d1_arrays.append(np.linspace(l, r, num_samples_dim))

        s_set = sample.sample_set(util.meshgrid_ndim(d1_arrays).shape[1])
        s_set.set_domain(lam_domain)
        s_set.set_values(util.meshgrid_ndim(d1_arrays))

        volume_exact = 1.0 / s_set._values.shape[0]

        emulated_samples = s_set.copy()
        emulated_samples.update_bounds_local(1001)
        emulated_samples.set_values_local(emulated_samples._width_local
                                          * np.random.random((1001,
                                                              emulated_samples.
                                                              get_dim())) +
                                          emulated_samples._left_local)

        self.disc.set_output_sample_set(s_set)
        self.disc.set_emulated_output_sample_set(emulated_samples)
        self.disc.estimate_output_volume_emulated()

        lam_vol = self.disc._output_sample_set._volumes

        # Check the dimension.
        nptest.assert_array_equal(lam_vol.shape, (len(s_set._values), ))

        # Check that the volumes are within a tolerance for a regular grid of
        # samples.
        nptest.assert_array_almost_equal(lam_vol, volume_exact, 1)
        nptest.assert_almost_equal(np.sum(lam_vol), 1.0)


class TestEstimateVolume(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.estimate_volulme`.
    """

    def setUp(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.
        """
        lam_left = np.array([0.0, .25, .4])
        lam_right = np.array([1.0, 4.0, .5])
        lam_width = lam_right - lam_left

        self.lam_domain = np.zeros((3, 2))
        self.lam_domain[:, 0] = lam_left
        self.lam_domain[:, 1] = lam_right

        num_samples_dim = 2
        start = lam_left + lam_width / (2 * num_samples_dim)
        stop = lam_right - lam_width / (2 * num_samples_dim)
        d1_arrays = []

        for l, r in zip(start, stop):
            d1_arrays.append(np.linspace(l, r, num_samples_dim))

        self.s_set = sample.sample_set(util.meshgrid_ndim(d1_arrays).shape[1])
        self.s_set.set_domain(self.lam_domain)
        self.s_set.set_values(util.meshgrid_ndim(d1_arrays))
        print(util.meshgrid_ndim(d1_arrays).shape)
        self.volume_exact = 1.0 / self.s_set._values.shape[0]
        self.s_set.estimate_volume(n_mc_points=1001)
        self.lam_vol = self.s_set._volumes

    def test_dimension(self):
        """
        Check the dimension.
        """
        print(self.lam_vol.shape, self.s_set._values.shape)
        nptest.assert_array_equal(
            self.lam_vol.shape, (len(self.s_set._values), ))

    def test_volumes(self):
        """
        Check that the volumes are within a tolerance for a regular grid of
        samples.
        """
        nptest.assert_array_almost_equal(self.lam_vol, self.volume_exact, 1)
        nptest.assert_almost_equal(np.sum(self.lam_vol), 1.0)


class TestEstimateVolumeEmulated(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.estimate_volulme_emulated`.
    """

    def setUp(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.
        """
        lam_left = np.array([0.0, .25, .4])
        lam_right = np.array([1.0, 4.0, .5])
        lam_width = lam_right - lam_left

        self.lam_domain = np.zeros((3, 2))
        self.lam_domain[:, 0] = lam_left
        self.lam_domain[:, 1] = lam_right

        num_samples_dim = 2
        start = lam_left + lam_width / (2 * num_samples_dim)
        stop = lam_right - lam_width / (2 * num_samples_dim)
        d1_arrays = []

        for l, r in zip(start, stop):
            d1_arrays.append(np.linspace(l, r, num_samples_dim))

        self.s_set = sample.sample_set(util.meshgrid_ndim(d1_arrays).shape[1])
        self.s_set.set_domain(self.lam_domain)
        self.s_set.set_values(util.meshgrid_ndim(d1_arrays))
        print(util.meshgrid_ndim(d1_arrays).shape)
        self.volume_exact = 1.0 / self.s_set._values.shape[0]
        emulated_samples = self.s_set.copy()
        emulated_samples.update_bounds_local(1001)
        emulated_samples.set_values_local(emulated_samples._width_local
                                          * np.random.random((1001,
                                                              emulated_samples.
                                                              get_dim())) +
                                          emulated_samples._left_local)
        self.s_set.estimate_volume_emulated(emulated_samples)
        self.lam_vol = self.s_set._volumes

    def test_dimension(self):
        """
        Check the dimension.
        """
        print(self.lam_vol.shape, self.s_set._values.shape)
        nptest.assert_array_equal(
            self.lam_vol.shape, (len(self.s_set._values), ))

    def test_volumes(self):
        """
        Check that the volumes are within a tolerance for a regular grid of
        samples.
        """
        nptest.assert_array_almost_equal(self.lam_vol, self.volume_exact, 1)
        nptest.assert_almost_equal(np.sum(self.lam_vol), 1.0)


class TestEstimateLocalVolume(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.estimate_local_volulme`.
    """

    def setUp(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.

        """
        lam_left = np.array([0.0, .25, .4])
        lam_right = np.array([1.0, 4.0, .5])
        lam_width = lam_right - lam_left

        self.lam_domain = np.zeros((3, 2))
        self.lam_domain[:, 0] = lam_left
        self.lam_domain[:, 1] = lam_right

        num_samples_dim = 2
        start = lam_left + lam_width / (2 * num_samples_dim)
        stop = lam_right - lam_width / (2 * num_samples_dim)
        d1_arrays = []

        for l, r in zip(start, stop):
            d1_arrays.append(np.linspace(l, r, num_samples_dim))

        self.s_set = sample.sample_set(util.meshgrid_ndim(d1_arrays).shape[1])
        self.s_set.set_domain(self.lam_domain)
        self.s_set.set_values(util.meshgrid_ndim(d1_arrays))
        self.volume_exact = 1.0 / self.s_set._values.shape[0]
        self.s_set.estimate_local_volume()
        self.lam_vol = self.s_set._volumes

    def test_dimension(self):
        """
        Check the dimension.
        """
        nptest.assert_array_equal(
            self.lam_vol.shape, (len(self.s_set._values), ))

    def test_volumes(self):
        """
        Check that the volumes are within a tolerance for a regular grid of
        samples.
        """
        nptest.assert_array_almost_equal(self.lam_vol, self.volume_exact, 2)
        nptest.assert_almost_equal(np.sum(self.lam_vol), 1.0)


class TestExactVolume1D(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.exact_volume_1D`.
    """

    def setUp(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.
        """
        num_samples = 10
        self.lam_domain = np.array([[.0, .1]])
        edges = np.linspace(self.lam_domain[:, 0], self.lam_domain[:, 1],
                            num_samples + 1)
        self.samples = (edges[1:] + edges[:-1]) * .5
        np.random.shuffle(self.samples)
        self.volume_exact = 1. / self.samples.shape[0]
        self.volume_exact = self.volume_exact * np.ones((num_samples,))
        s_set = sample.voronoi_sample_set(dim=1)
        s_set.set_domain(self.lam_domain)
        s_set.set_values(self.samples)
        s_set.exact_volume_1D()
        self.lam_vol = s_set.get_volumes()

    def test_dimension(self):
        """
        Check the dimension.
        """
        nptest.assert_array_equal(self.lam_vol.shape, (len(self.samples), ))

    def test_volumes(self):
        """
        Check that the volumes are within a tolerance for a regular grid of
        samples.
        """
        nptest.assert_array_almost_equal(self.lam_vol, self.volume_exact)
        nptest.assert_almost_equal(np.sum(self.lam_vol), 1.0)


class TestExactVolume2D(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.exact_volume_2D`.
    """

    def setUp(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.
        """
        sampler = bsam.sampler(None)
        self.input_samples = sample.sample_set(2)
        self.input_samples.set_domain(np.array([[0.0, 1.0], [0.0, 1.0]]))
        self.input_samples = sampler.regular_sample_set(
            self.input_samples, num_samples_per_dim=[10, 9])
        self.input_samples.exact_volume_2D()
        self.vol1 = np.copy(self.input_samples._volumes)
        self.input_samples.estimate_volume_mc()
        self.vol2 = np.copy(self.input_samples._volumes)

    def test_volumes(self):
        """
        Check that the volumes are within a tolerance for a regular grid of
        samples.
        """
        nptest.assert_array_almost_equal(self.vol1, self.vol2)
        nptest.assert_almost_equal(np.sum(self.vol1), 1.0)


class TestEstimateRadii(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.estimate_radii`.
    """

    def setUp(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.

        """
        lam_left = np.array([0.0, 0.5, 0.5])
        lam_right = np.array([1.0, 1.5, 1.5])
        lam_width = lam_right - lam_left

        self.lam_domain = np.zeros((3, 2))
        self.lam_domain[:, 0] = lam_left
        self.lam_domain[:, 1] = lam_right

        num_samples_dim = 2
        start = lam_left + lam_width / (2 * num_samples_dim)
        stop = lam_right - lam_width / (2 * num_samples_dim)
        d1_arrays = []

        for l, r in zip(start, stop):
            d1_arrays.append(np.linspace(l, r, num_samples_dim))

        self.s_set = sample.sample_set(util.meshgrid_ndim(d1_arrays).shape[1])
        self.s_set.set_domain(self.lam_domain)
        self.s_set.set_values(util.meshgrid_ndim(d1_arrays))

        self.radii_exact = np.sqrt(3 * .25**2)

        self.s_set.estimate_radii(normalize=False)
        self.s_set.estimate_radii()
        self.rad = self.s_set._radii
        self.norm_rad = self.s_set._normalized_radii

    def test_dimension(self):
        """
        Check the dimension.
        """
        nptest.assert_array_equal(self.rad.shape, (len(self.s_set._values), ))
        nptest.assert_array_equal(
            self.norm_rad.shape, (len(self.s_set._values), ))

    def test_radii(self):
        """
        Check that the radii are within a tolerance for a regular grid of
        samples.
        """
        nptest.assert_array_almost_equal(self.rad, self.radii_exact, 1)
        nptest.assert_array_almost_equal(self.norm_rad, self.radii_exact, 1)


class TestEstimateRadiiAndVolume(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.estimate_radii_and_volume`.
    """

    def setUp(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.

        """
        lam_left = np.array([0.0, 0.5, 0.5])
        lam_right = np.array([1.0, 1.5, 1.5])
        lam_width = lam_right - lam_left

        self.lam_domain = np.zeros((3, 2))
        self.lam_domain[:, 0] = lam_left
        self.lam_domain[:, 1] = lam_right

        num_samples_dim = 2
        start = lam_left + lam_width / (2 * num_samples_dim)
        stop = lam_right - lam_width / (2 * num_samples_dim)
        d1_arrays = []

        for l, r in zip(start, stop):
            d1_arrays.append(np.linspace(l, r, num_samples_dim))

        self.s_set = sample.sample_set(util.meshgrid_ndim(d1_arrays).shape[1])
        self.s_set.set_domain(self.lam_domain)
        self.s_set.set_values(util.meshgrid_ndim(d1_arrays))
        self.volume_exact = 1.0 / self.s_set._values.shape[0]

        self.radii_exact = np.sqrt(3 * .25**2)

        self.s_set.estimate_radii_and_volume(normalize=False)
        self.s_set.estimate_radii_and_volume()
        self.lam_vol = self.s_set._volumes
        self.rad = self.s_set._radii
        self.norm_rad = self.s_set._normalized_radii

    def test_dimension(self):
        """
        Check the dimension.
        """
        nptest.assert_array_equal(self.rad.shape, (len(self.s_set._values), ))
        nptest.assert_array_equal(
            self.norm_rad.shape, (len(self.s_set._values), ))
        nptest.assert_array_equal(
            self.lam_vol.shape, (len(self.s_set._values), ))

    def test_radii(self):
        """
        Check that the volumes are within a tolerance for a regular grid of
        samples.
        """
        nptest.assert_array_almost_equal(self.rad, self.radii_exact, 1)
        nptest.assert_array_almost_equal(self.norm_rad, self.radii_exact, 1)
        nptest.assert_array_almost_equal(self.lam_vol, self.volume_exact, 1)
        nptest.assert_almost_equal(np.sum(self.lam_vol), 1.0)


class Test_rectangle_sample_set(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.sam_set = sample.rectangle_sample_set(dim=self.dim)
        # maximum number
        nprocs = 8
        self.nprocs = nprocs
        n = np.linspace(0.1, 0.9, nprocs)
        maxes = [[n[i], n[i]] for i in range(1, nprocs)]
        mins = [[n[i], n[i]] for i in range(nprocs - 1)]
        self.sam_set.setup(maxes, mins)
        self.domain = np.array([[0, 1], [0, 1]], dtype=np.float)
        self.sam_set.set_domain(self.domain)
        self.num = self.sam_set.check_num()

    def test_save_load(self):
        """
        Check save_sample_set and load_sample_set.
        """
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)

        file_name = os.path.join(local_path, 'testfile.mat')
        globalize = True
        sample.save_sample_set(self.sam_set, file_name, "TEST", globalize)
        comm.barrier()

        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_set = sample.load_sample_set(local_file_name, "TEST")
        loaded_set_none = sample.load_sample_set(local_file_name)

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print(attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)
        comm.barrier()

        file_name = os.path.join(local_path, 'testfile.mat')
        globalize = False
        sample.save_sample_set(self.sam_set, file_name, "TEST", globalize)
        comm.barrier()

        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_set = sample.load_sample_set(local_file_name, "TEST")
        loaded_set_none = sample.load_sample_set(local_file_name)

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print(attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)

    def test_copy(self):
        """
        Check copy.
        """
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)
        self.sam_set.set_kdtree()

        copied_set = self.sam_set.copy()
        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(copied_set, attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        assert copied_set._kdtree is not None

    def test_query(self):
        """
        Check querying
        """
        n = np.linspace(0.1, 0.9, self.nprocs)
        x = np.array([[n[i] + 1E-5, n[i] + 1E-5]
                      for i in range(self.nprocs - 1)])
        (d, ptr) = self.sam_set.query(x)
        nptest.assert_array_equal(ptr, np.arange(self.nprocs - 1))

    def test_volumes(self):
        """
        Check volume calculation
        """
        self.sam_set.exact_volume_lebesgue()
        volumes = self.sam_set.get_volumes()
        volumes_exact = (0.8 / (self.nprocs - 1))**2
        total_vol_exact = 1 - (self.nprocs - 1) * volumes_exact
        nptest.assert_array_almost_equal(volumes[:-1], volumes_exact)
        nptest.assert_array_almost_equal(volumes[-1], total_vol_exact)


class Test_ball_sample_set(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.sam_set = sample.ball_sample_set(dim=self.dim)
        # max number of processors supported in test.
        nprocs = 8
        self.nprocs = nprocs
        n = np.linspace(0.1, 0.9, nprocs)
        centers = [[n[i], n[i]] for i in range(nprocs - 1)]
        radii = [0.1] * (nprocs - 1)
        self.sam_set.setup(centers, radii)
        self.domain = np.array([[0, 1], [0, 1]], dtype=np.float)
        self.sam_set.set_domain(self.domain)
        self.num = self.sam_set.check_num()

    def test_save_load(self):
        """
        Check save_sample_set and load_sample_set.
        """
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)

        # Do serial tests
        globalize = True
        file_name = os.path.join(local_path, 'testfile.mat')
        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        print(os.path.exists(local_file_name))

        sample.save_sample_set(self.sam_set, file_name, "TEST", globalize)
        comm.barrier()

        loaded_set = sample.load_sample_set(local_file_name, "TEST")
        loaded_set_none = sample.load_sample_set(local_file_name)

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print(attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)
        comm.barrier()

        # Do parallel tests
        file_name = os.path.join(local_path, 'testfile.mat')
        globalize = False
        sample.save_sample_set(self.sam_set, file_name, "TEST", globalize)
        comm.barrier()

        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_set = sample.load_sample_set(local_file_name, "TEST")
        loaded_set_none = sample.load_sample_set(local_file_name)

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print(attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)

    def test_copy(self):
        """
        Check copy.
        """
        prob = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num) * np.ones((self.num,))
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)
        self.sam_set.set_kdtree()

        copied_set = self.sam_set.copy()
        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(copied_set, attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        assert copied_set._kdtree is not None

    def test_query(self):
        """
        Check querying
        """
        n = np.linspace(0.1, 0.9, self.nprocs)
        x = np.array([[n[i] + 1E-5, n[i] + 1E-5]
                      for i in range(self.nprocs - 1)])
        (d, ptr) = self.sam_set.query(x)
        nptest.assert_array_equal(ptr, np.arange(self.nprocs - 1))

    def test_volumes(self):
        """
        Check volume calculation
        """
        self.sam_set.exact_volume()
        volumes = self.sam_set.get_volumes()
        nptest.assert_array_almost_equal(volumes[:-1], np.pi / 100)
        leftover_volume = 1 - (self.nprocs - 1) * np.pi / 100
        nptest.assert_array_almost_equal(volumes[-1], leftover_volume)


class Test_cartesian_sample_set(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.sam_set = sample.cartesian_sample_set(dim=self.dim)
        # number of processors test should support (for dim<=2 only)
        nprocs = 10
        self.nprocs = int(np.ceil(np.sqrt(nprocs))**2)
        # equispaced grid in each dimension
        vi = np.linspace(0, 1, 1 + int(np.sqrt(self.nprocs)))
        xi = [vi, vi]
        self.sam_set.setup(xi)
        self.domain = np.array([[0, 1], [0, 1]], dtype=np.float)
        self.sam_set.set_domain(self.domain)
        self.num = self.sam_set.check_num()

    def test_save_load(self):
        """
        Check save_sample_set and load_sample_set.
        """
        prob = 1.0 / float(self.num - 1) * np.ones((self.num,))
        prob[-1] = 0
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num - 1) * np.ones((self.num,))
        vol[-1] = 0
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)

        globalize = True
        file_name = os.path.join(local_path, 'testfile.mat')
        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        print(os.path.exists(local_file_name))

        sample.save_sample_set(self.sam_set, file_name, "TEST", globalize)
        comm.barrier()

        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_set = sample.load_sample_set(local_file_name, "TEST")
        loaded_set_none = sample.load_sample_set(local_file_name)

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print(attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)
        comm.barrier()

        file_name = os.path.join(local_path, 'testfile.mat')
        globalize = False
        sample.save_sample_set(self.sam_set, file_name, "TEST", globalize)
        comm.barrier()

        if comm.size > 1 and not globalize:
            local_file_name = os.path.os.path.join(os.path.dirname(file_name),
                                                   "proc{}_{}".
                                                   format(comm.rank, os.path.
                                                          basename(file_name)))
        else:
            local_file_name = file_name

        loaded_set = sample.load_sample_set(local_file_name, "TEST")
        loaded_set_none = sample.load_sample_set(local_file_name)

        assert loaded_set_none is None

        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(loaded_set, attrname)
            print(attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        if comm.rank == 0 and globalize:
            os.remove(local_file_name)
        elif not globalize:
            os.remove(local_file_name)

    def test_copy(self):
        """
        Check copy.
        """
        prob = 1.0 / float(self.num - 1) * np.ones((self.num,))
        prob[-1] = 0
        self.sam_set.set_probabilities(prob)
        vol = 1.0 / float(self.num - 1) * np.ones((self.num,))
        vol[-1] = 0
        self.sam_set.set_volumes(vol)
        ee = np.ones((self.num, self.dim))
        self.sam_set.set_error_estimates(ee)
        jac = np.ones((self.num, 3, self.dim))
        self.sam_set.set_jacobians(jac)
        self.sam_set.global_to_local()
        self.sam_set.set_domain(self.domain)
        self.sam_set.set_kdtree()

        copied_set = self.sam_set.copy()
        for attrname in sample.sample_set.vector_names + sample.sample_set.\
                all_ndarray_names:
            curr_attr = getattr(copied_set, attrname)
            if curr_attr is not None:
                nptest.assert_array_equal(getattr(self.sam_set, attrname),
                                          curr_attr)

        assert copied_set._kdtree is not None

    def test_query(self):
        """
        Check querying
        """
        x = self.sam_set.get_values()[:-1, :] + 1E-5
        (d, ptr) = self.sam_set.query(x)
        nptest.assert_array_equal(ptr, np.arange(self.nprocs))

    def test_volumes(self):
        """
        Check volume calculation
        """
        self.sam_set.exact_volume_lebesgue()
        volumes = self.sam_set.get_volumes()
        nptest.assert_array_almost_equal(volumes[:-1], 1. / self.nprocs)
        assert volumes[-1] == 0


class Test_sampling_discretization(unittest.TestCase):
    r"""
    Test the sampling-based approach.
    """

    def setUp(self):
        self.dim1 = 3
        self.num = 300
        self.dim2 = 2
        values1 = np.random.rand(self.num, self.dim1)
        values2 = np.random.randn(self.num, self.dim2)
        self.input_values = values1
        self.output_values = values2
        values3 = np.ones((self.num, self.dim2))
        self.input_set = sample.sample_set(dim=self.dim1)
        self.output_set = sample.sample_set(dim=self.dim2)
        self.output_probability_set = sample.sample_set(dim=self.dim2)
        self.input_set.set_values(values1)
        self.output_set.set_values(values2)
        self.output_probability_set.set_values(values3)
        self.disc = sample.discretization(input_sample_set=self.input_set,
                                          output_sample_set=self.output_set,
                                          output_probability_set=self.
                                          output_probability_set)

    def test_format_indices(self):
        """
        Test multitude of index-formatting options.
        """
        # Single Float Test
        a = 0.3
        b = a - 1
        n = 50
        set_inds = self.disc.format_indices  # shortened function handle
        nptest.assert_array_equal(np.array(set_inds(n, a) +
                                           set_inds(n, b)),
                                  np.arange(n))

        # Integer Test
        nptest.assert_array_equal(np.array(set_inds(n, int(a * n)) +
                                           set_inds(n, int(b * n))),
                                  np.arange(n))

        # List Test
        nptest.assert_array_equal(np.array(set_inds(n, np.arange(n))),
                                  np.arange(n))
        nptest.assert_array_equal(np.array(set_inds(n, list(np.arange(n)))),
                                  np.arange(n))

        # Tuple Test(s)
        for i in range(1, n):
            nptest.assert_array_equal(np.array(set_inds(n + 1, (i,))),
                                      np.arange(0, n + 1, i))
            nptest.assert_array_equal(np.array(set_inds(n, None)),
                                      np.arange(n))
            nptest.assert_array_equal(np.array(set_inds(n, (i, 1))),
                                      np.arange(n)[i::])
            nptest.assert_array_equal(np.array(set_inds(n, (i, 2))),
                                      np.arange(n)[np.arange(i, n, 2)])
            nptest.assert_array_equal(np.array(set_inds(n, (i, 1.0, 3))),
                                      np.arange(n)[np.arange(i, n, 3)])
            nptest.assert_array_equal(np.array(set_inds(n, (i, n, 1))),
                                      np.arange(n)[i::])
            nptest.assert_array_equal(np.array(set_inds(n, (i, n, 2))),
                                      np.arange(n)[np.arange(i, n, 2)])
            nptest.assert_array_equal(np.array(set_inds(n, (i, n, 0.2))),
                                      np.arange(n)[np.arange(i, n,
                                                             int(0.2 * n))])
            nptest.assert_array_equal(np.array(set_inds(n, (0, 0.2, i))),
                                      np.arange(n)[np.arange(0,
                                                             int(0.2 * n), i)])
            nptest.assert_array_equal(np.array(set_inds(n, (1, 0.2, i))),
                                      np.arange(n)[np.arange(1,
                                                             int(0.2 * n), i)])
            nptest.assert_array_equal(np.array(set_inds(n, (0.2, 1.0, 1))),
                                      np.arange(n)[np.arange(int(0.2 * n),
                                                             n, 1)])
            nptest.assert_array_equal(np.array(set_inds(n, (0.2, n, 2))),
                                      np.arange(n)[np.arange(int(0.2 * n),
                                                             n, 2)])

    def test_empty_problem(self):
        """
        Test problem setup syntaxes.
        """
        D = self.disc

        def mymodel(input_values):
            try:
                return 2 * input_values[:, np.arange(self.dim2) % self.dim1]
            except IndexError:  # handle 1-d arrays (for reference vals)
                return 2 * input_values[np.arange(self.dim2) % self.dim1]

        D.set_model(mymodel)
        D.set_initial()  # uniform [0,1]
        # D.set_observed(dist.norm())  # set_output_probability_set [TK - fix]
        # D.get_output().set_reference_value(2*np.ones(self.dim2))  # only for
        # size inference?
        D.set_observed()
        D.set_data_from_observed()
        # nptest.assert_array_equal(D.get_data(), 2)
        # D.set_predicted()  # should be optional...

        # evaluate on existing samples
        D.updated_pdf()
        D.initial_pdf()
        D.observed_pdf()
        D.predicted_pdf()

        # evaluate at a given set of points
        D.initial_pdf(self.input_values)
        D.updated_pdf(self.input_values)
        D.observed_pdf(self.output_values)
        D.predicted_pdf(self.output_values)

    def test_set_observed_rv_continuous(self):
        """
        Test default behavior of set_observed.
        """
        D = self.disc
        D.set_observed(dist.norm)
        D.set_data_from_observed()
        nptest.assert_array_equal(D.get_data(), 0)
        D.observed_pdf()
        D.set_data_driven_mode('SSE')
        D.observed_pdf()
        D.set_data_driven_mode('MSE')

    def test_set_observed_rv_frozen(self):
        """
        Test default behavior of set_observed with frozen dist.
        """
        D = self.disc
        D.set_observed(dist.norm(loc=2 * np.ones(self.dim2)))
        D.set_data_from_observed()
        nptest.assert_array_equal(D.get_data(), 2)

    def test_solve_problem(self):
        """
        Solve inverse problem (input dim == output dim)
        """
        D = self.disc

        def mymodel(input_values):
            try:
                return 2 * input_values[:, np.arange(self.dim2) % self.dim1]
            except IndexError:  # handle 1-d arrays (for reference vals)
                return 2 * input_values[np.arange(self.dim2) % self.dim1]

        D.set_model(mymodel)
        D.set_initial(dist.uniform(loc=[0] * self.dim1,
                                   scale=[1] * self.dim1))  # uniform [0,1]
        # set_output_probability_set
        D.set_observed(dist.uniform(loc=[0.5] * self.dim2,
                                    scale=[1] * self.dim2))
        D.set_predicted(dist.uniform(loc=[0] * self.dim2,
                                     scale=[2] * self.dim2))

        updated_pdf = D.updated_pdf()

        # check that correct samples received positive probability
        pos_vals = D.get_input_values()[updated_pdf > 0, :]
        nptest.assert_array_equal(pos_vals[:, :self.dim2] < 0.75, True)
        nptest.assert_array_equal(
            pos_vals[:, :self.dim2] > 0.25, True)  # greater than
        # check validity of solution against (simple-function) analytical one
        assert np.max(updated_pdf[updated_pdf > 0] - 2**self.dim1) < 1E-14
        D.mud_index()
        D.mud_point()

    def test_set_observed_no_reference(self):
        """
        Test how observed behaves without reference output.
        """
        D = self.disc
        for l in [1, 2, 3]:
            D.set_observed(loc=l)  # infer dimension correctly
            D.get_data()
            assert np.linalg.norm(np.array(D.get_std()) - 1) == 0
        for s in [1, 2, 3]:
            D.set_observed(scale=s)  # infer dimension correctly
            D.get_data()
            assert np.linalg.norm(np.array(D.get_std()) - s) == 0

    def test_get_std_from_obs(self):
        """
        Test inferring std from observed distribution.
        """
        D = self.disc
        for l in [1, 2, 3]:
            D.set_observed(scale=l)  # infer dimension correctly
            D._setup[0]['std'] = None
            assert np.linalg.norm(np.array(D.get_std()) - l) == 0

    def test_set_data(self):
        """
        Test straight-forward data-setting.
        """
        D = self.disc.copy()
        ref_val = 21 * np.random.rand(self.dim2)
        # set manually
        D._output_probability_set._reference_value = np.copy(ref_val)
        nptest.assert_array_equal(D.get_data(), ref_val)
        D = self.disc.copy()
        # set using function
        D.set_data(ref_val)
        nptest.assert_array_equal(D.get_data(), ref_val)
        # inherit std from observed
        D.set_observed()
        nptest.assert_array_equal(D.get_std(), np.ones(self.dim2))
        # test perturbation in-place
        D._output_probability_set._reference_value += 1
        nptest.assert_array_equal(D.get_data(), ref_val + 1)
        # test inferring std from data
        if self.dim2 > 1:
            D._setup[0]['std'] = None
            D._setup[0]['obs'] = None
            assert np.linalg.norm(np.array(D.get_std()) -
                                  D.get_data().std()) == 0
            # change data
            D.set_data(1 + 2 * ref_val)
            assert np.linalg.norm(np.array(D.get_std()) -
                                  D.get_data().std()) == 0

    def test_set_data_from_observed(self):
        """
        Test using mean/median/bound of observed as data.
        """
        D = self.disc
        D.set_observed()

        a, b = 1, 2
        loc, scale = np.zeros(self.dim2), np.ones(self.dim2)

        obs_dist = dist.beta(a=a, b=b, loc=loc, scale=scale)
        D.set_observed(obs_dist)
        # take draw from distribution
        D.set_data_from_observed()
        nptest.assert_array_equal(obs_dist.mean() -
                                  D.get_data(), 0)
        D.set_data_from_observed('mean')
        nptest.assert_array_equal(obs_dist.mean() -
                                  D.get_data(), 0)
        D.set_data_from_observed('median')
        nptest.assert_array_equal(obs_dist.median() -
                                  D.get_data(), 0)
        # interval around mean value using our keywords: min/max
        D.set_data_from_observed('min')
        nptest.assert_array_equal(obs_dist.interval(0.99)[0] -
                                  D.get_data(), 0)
        D.set_data_from_observed('min', alpha=0.15)
        nptest.assert_array_equal(obs_dist.interval(0.15)[0] -
                                  D.get_data(), 0)
        D.set_data_from_observed('max')
        nptest.assert_array_equal(obs_dist.interval(0.99)[1] -
                                  D.get_data(), 0)
        D.set_data_from_observed('max', alpha=0.25)
        nptest.assert_array_equal(obs_dist.interval(0.25)[1] -
                                  D.get_data(), 0)

    def test_set_data_from_reference(self):
        """
        Test using reference and observed to perturb
        """
        D = self.disc
        ref_val = np.ones(self.dim2)
        D._output_sample_set.set_reference_value(ref_val)

        std = 0.1
        noise_dist = dist.norm(loc=0, scale=std)

        D.set_noise_model(noise_dist)
        D.set_data_from_reference()

        assert np.max(np.abs(D.get_data() - ref_val)) < std * 6

        # should be able to pass scalar values and have normal assumption
        D.set_noise_model(std * 2)  # double error level
        D.set_data_from_reference()
        assert np.max(np.abs(D.get_data() - ref_val)) < std * 12

        if D._setup[0]['model'] is not None:
            # recover missing output reference if input present.
            D._input_sample_set.set_reference_value(ref_val / 2)
            D._output_sample_set._reference_value = None
            D.set_noise_model(std / 2)  # half error level
            D.set_data_from_reference()
            assert np.max(np.abs(D.get_data() - ref_val)) < std * 3

        # use draw directly from observed if reference is empty
        D._input_sample_set._reference_value = None
        D._output_sample_set._reference_value = None
        D.set_noise_model(std / 2)  # half error level
        D.set_data_from_reference()
        assert np.max(np.abs(D.get_data())) < std * 3

    def test_set_initial_no_model(self):
        """
        Test setting initial without model.
        """
        import scipy.stats as sstats
        D = self.disc
        D.set_initial(dist.norm)
        assert isinstance(D.get_initial().dist,
                          sstats._continuous_distns.norm_gen)
        assert D.get_output_values().size == 0
        # test that re-setting initial works as expected (deletes output
        # values)
        D.set_initial(dist.uniform)
        assert D.get_output_values().size == 0
        assert isinstance(D.get_initial().dist,
                          sstats._continuous_distns.uniform_gen)
        D.set_initial(dist.norm(loc=np.zeros(self.dim1)))
        assert D.get_output_values().size == 0
        assert isinstance(D.get_initial().dist,
                          sstats._continuous_distns.norm_gen)
        D.get_input().set_reference_value(np.ones(self.dim1))
        D.set_initial(dist.norm(loc=np.zeros(self.dim1)))
        assert D.get_output_values().size == 0
        # test that this syntax works
        D.get_initial_distribution()

    def test_set_initial_with_model(self):
        """
        Test setting initial with model.
        """
        import scipy.stats as sstats
        D = self.disc
        # set model
        for dim in range(1, 5):
            self.dim2 = dim

            def mymodel(input_values):
                try:
                    return 2 * input_values[:, np.arange(dim) % self.dim1]
                except IndexError:  # handle 1-d arrays (for reference vals)
                    return 2 * input_values[np.arange(dim) % self.dim1]

            D.set_model(mymodel)
            D.set_initial(dist.norm)
            assert isinstance(D.get_initial_distribution().dist,
                              sstats._continuous_distns.norm_gen)
            # set_output_probability_set
            assert D.get_output_values().shape == (self.num, dim)
            # test that re-setting initial works as expected (deletes output
            # values)
            D.set_initial(dist.uniform)
            assert D.get_output_values().size == self.num * dim
            assert isinstance(D.get_initial().dist,
                              sstats._continuous_distns.uniform_gen)
            D.set_initial(dist.norm(loc=np.zeros(self.dim1)))
            assert D.get_output_values().shape == (self.num, dim)
            assert isinstance(D.get_initial().dist,
                              sstats._continuous_distns.norm_gen)
            D.get_input().set_reference_value(np.ones(self.dim1))
            D.set_initial(dist.norm(loc=np.zeros(self.dim1)))
            assert D.get_output_values().shape == (self.num, dim)
            assert D.get_output_values().size == self.num * dim

    def test_set_observed(self):
        """
        Test setting observed.
        """
        import scipy.stats as sstats
        D = self.disc
        D.set_observed(dist.norm(loc=np.arange(self.dim2)))
        assert isinstance(D.get_observed().dist,
                          sstats._continuous_distns.norm_gen)
        try:
            D.set_output_probability_set(None)
        except AttributeError:
            pass
        D._output_probability_set = None
        D.set_observed(dist.uniform(loc=np.arange(self.dim2)))
        assert isinstance(D.get_observed().dist,
                          sstats._continuous_distns.uniform_gen)

    def test_set_predicted(self):
        """
        Test setting predicted.
        """
        import scipy.stats as sstats
        from scipy.stats import gaussian_kde as gkde
        D = self.disc
        num = 21

        def mymodel(input_values):
            try:
                return 2 * input_values[:, np.arange(self.dim2) % self.dim1]
            except IndexError:  # handle 1-d arrays (for reference vals)
                return 2 * input_values[np.arange(self.dim2) % self.dim1]

        D.set_model(mymodel)
        D.set_initial(num=num)
        assert D.check_nums() == num
        D.set_predicted()
        assert isinstance(D.get_predicted(), gkde)
        D.set_predicted(dist.uniform(loc=0, scale=1))
        assert isinstance(D.get_predicted().dist,
                          sstats._continuous_distns.uniform_gen)

    def test_set_std(self):
        """
        Test set/get for standard deviation parameter.
        """
        D = self.disc
        D.set_data(np.ones(self.dim2))
        D.set_std(0.1)
        nptest.assert_array_equal(D.get_std(), 0.1 * np.ones(self.dim2))
        # make sure vector of correct length works
        vec = np.random.rand(self.dim2)
        D.set_std(vec)
        nptest.assert_array_equal(D.get_std(), vec)
        vec = np.random.rand(self.dim2 - 1)
        try:  # should throw dim_not_matching error.
            D.set_std(vec)
            assert len(D.get_std()) == self.dim2
        except sample.dim_not_matching:
            pass

    def test_likelihood(self):
        """
        Test likelihood function in default mode.
        """
        D = self.disc

        def mymodel(input_values):
            try:
                return 2 * input_values[:, np.arange(self.dim2) % self.dim1]
            except IndexError:  # handle 1-d arrays (for reference vals)
                return 2 * input_values[np.arange(self.dim2) % self.dim1]

        D.set_model(mymodel)
        D.set_initial(dist.uniform(loc=[0] * self.dim1,
                                   scale=[1] * self.dim1))  # uniform [0,1]
        D.set_observed(dist.norm(loc=[0.5] * self.dim2,
                                 scale=[0.1] * self.dim2))
        D.set_predicted(dist.uniform(loc=[0] * self.dim2,
                                     scale=[2] * self.dim2))
        mud_point = D.mud_point()
        D.set_likelihood()
        map_point = D.map_point()
        assert np.linalg.norm(mud_point - map_point) / \
            np.linalg.norm(map_point) < 0.05
        # nptest.assert_array_equal(mud_point, map_point)

    def test_shorthand(self):
        """
        Test short-hand versions of methods.
        """
        assert self.disc.get_input() == self.disc.get_input_sample_set()
        assert self.disc.get_output() == self.disc.get_output_sample_set()

    def test_set_data_driven_mode(self):
        """
        Test data-driven functional setting.
        """
        D = self.disc
        for mode in ['SSE', 'MSE', 'SWE']:
            D.set_data_driven_mode(mode)
            assert D.get_setup()['qoi'] == mode
        # test error-catching
        try:
            D.set_data_driven('FAKE')
        except ValueError:
            pass

    def test_set_data_driven_status(self):
        """
        Test data-driven mode options.
        """
        D = self.disc
        D.set_data_driven()
        # test toggling modes
        assert D.get_setup(0)['col'] is True
        D.set_data_driven(False)
        assert D.get_setup(0)['col'] is False
        # test inheriting mode
        D.iterate()
        assert D.get_setup()['col'] is False
        # change current mode
        D.set_data_driven()
        assert D.get_setup()['col'] is True
        # change previous mode
        D.set_data_driven(True, iteration=0)
        assert D.get_setup(0)['col'] is True

    # def test_set_initial_densities(self):
    #     """
    #     TK - fix this test.
    #     """
    #     self.disc.set_initial_densities()


class Test_sampling_one_dim(Test_sampling_discretization):
    def setUp(self):
        self.dim1 = 1
        self.num = 100
        self.dim2 = 1
        values1 = np.random.rand(self.num, self.dim1)
        values2 = np.random.randn(self.num, self.dim2)
        self.input_values = values1
        self.output_values = values2
        values3 = np.ones((self.num, self.dim2))
        self.input_set = sample.sample_set(dim=self.dim1)
        self.output_set = sample.sample_set(dim=self.dim2)
        self.output_probability_set = sample.sample_set(dim=self.dim2)
        self.input_set.set_values(values1)
        self.output_set.set_values(values2)
        self.output_probability_set.set_values(values3)
        self.disc = sample.discretization(input_sample_set=self.input_set,
                                          output_sample_set=self.output_set,
                                          output_probability_set=self.
                                          output_probability_set)


class Test_sampling_data_driven(Test_sampling_discretization):
    """
    If the pushforward is constant, MUD/MAP must match if there
    is only a single observation.
    """

    def setUp(self):
        self.dim1 = 1
        self.num = 101
        self.dim2 = 1
        # values1 = np.random.rand(self.num, self.dim1)
        values1 = np.linspace(0, 1, self.num).reshape(-1, 1)

        def mymodel(input_values):
            try:
                return 2 * input_values[:, np.arange(self.dim2) % self.dim1]
            except IndexError:  # handle 1-d arrays (for reference vals)
                return 2 * input_values[np.arange(self.dim2) % self.dim1]
        values2 = mymodel(values1)

        self.input_values = values1
        self.output_values = values2
        values3 = np.ones((self.num, self.dim2))
        self.input_set = sample.sample_set(dim=self.dim1)
        self.output_set = sample.sample_set(dim=self.dim2)
        self.input_set.set_values(values1)
        self.output_set.set_values(values2)
        self.disc = sample.discretization(input_sample_set=self.input_set,
                                          output_sample_set=self.output_set)

        self.disc.set_initial(dist.uniform(loc=[0] * self.dim1,
                                           scale=[1] * self.dim1), gen=False)

        self.model = mymodel
        self.disc.set_data_driven()
        self.std = 0.01

        true_value = np.array([0.5] * self.dim1)
        self.ans = true_value
        target_noise = self.std * np.random.randn(self.dim2)
        true_target = mymodel(true_value)
        target = true_target + target_noise
        self.disc.set_data(target, std=self.std)

    def test_solve_problem(self):
        """
        Solve inverse problem (input dim != output dim)
        """
        D = self.disc
        mymodel = self.model

        D.set_model(mymodel)
        # make sure this function can be called.
        updated_pdf = D.updated_pdf()

        # check that correct samples received positive probability
        expected = 0.005 * updated_pdf.max()
        # get relatively high probability samples
        pos_vals = D.get_input_values()[updated_pdf > expected, :]
        # ensure they are within 3 std deviations of true value
        nptest.assert_array_equal(pos_vals < self.ans + self.std * 3, True)
        nptest.assert_array_equal(pos_vals > self.ans - self.std * 3, True)
        # check validity of solution against # TK - SOMETIMES FAILS
        assert np.linalg.norm(D.mud_point() - self.ans) <= 1E-2 + 1E-6

    def test_set_predicted(self):
        """
        Test setting predicted.
        """
        import scipy.stats as sstats
        from scipy.stats import gaussian_kde as gkde
        D = self.disc
        num = 21
        mymodel = self.model

        D.set_model(mymodel)
        D.set_initial(num=num)
        assert D.check_nums() == num
        D.set_predicted()
        assert isinstance(D.get_predicted(), gkde)
        D.set_predicted(dist.uniform(loc=0, scale=2))
        assert isinstance(D.get_predicted().dist,
                          sstats._continuous_distns.uniform_gen)

    def test_likelihood(self):
        """
        Test likelihood function with data-driven.
        """
        D = self.disc
        mymodel = self.model

        D.set_model(mymodel)

        D.set_predicted()

        mud_point = D.mud_point()
        # set_output_probability_set
        # D.set_observed(scale=[self.std] * self.dim2)
        D.set_noise_model(self.std)  # is this necessary?
        D.set_likelihood()
        map_point = D.map_point()
        # nptest.assert_array_almost_equal(mud_point, map_point, 2)
        assert np.linalg.norm(mud_point - map_point) / \
            np.linalg.norm(map_point) < 0.05


class Test_sampling_data_driven_alt(Test_sampling_data_driven):
    """
    With enough data points, we should get the same solution
    despite havin more error in observations.
    """

    def setUp(self):
        self.dim1 = 1
        self.num = 200
        self.dim2 = 50
        # values1 = np.random.rand(self.num, self.dim1)
        values1 = np.linspace(0, 1, self.num).reshape(-1, 1)

        def mymodel(input_values):
            try:
                return 2 * input_values[:, np.arange(self.dim2) % self.dim1]
            except IndexError:  # handle 1-d arrays (for reference vals)
                return 2 * input_values[np.arange(self.dim2) % self.dim1]
        values2 = mymodel(values1)

        self.input_values = values1
        self.output_values = values2
        values3 = np.ones((self.num, self.dim2))
        self.input_set = sample.sample_set(dim=self.dim1)
        self.output_set = sample.sample_set(dim=self.dim2)
        self.input_set.set_values(values1)
        self.output_set.set_values(values2)
        self.disc = sample.discretization(input_sample_set=self.input_set,
                                          output_sample_set=self.output_set)

        self.disc.set_initial(dist.uniform(loc=[0] * self.dim1,
                                           scale=[1] * self.dim1), gen=False)

        self.model = mymodel
        self.disc.set_data_driven()
        self.std = 0.05

        true_value = np.array([0.5] * self.dim1)
        self.ans = true_value
        target_noise = self.std * np.random.randn(self.dim2)
        true_target = mymodel(true_value)
        target = true_target + target_noise
        self.disc.set_data(target, std=self.std)


class Test_sampling_repeated(Test_sampling_data_driven):
    def setUp(self):
        self.dim1 = 1
        self.num = 300
        self.dim2 = 1
        self.num_obs = 50
        # values1 = np.random.rand(self.num, self.dim1)
        values1 = np.linspace(0, 1, self.num).reshape(-1, 1)

        def mymodel(input_values):
            return 2 * input_values
        values2 = mymodel(values1)
        self.input_values = values1
        self.output_values = values2
        values3 = np.ones((self.num, self.dim2))
        self.input_set = sample.sample_set(dim=self.dim1)
        self.output_set = sample.sample_set(dim=self.dim1)
        self.input_set.set_values(values1)
        self.output_set.set_values(values2)
        self.disc = sample.discretization(input_sample_set=self.input_set,
                                          output_sample_set=self.output_set)

        self.disc.set_initial(dist.uniform(loc=[0] * self.dim1,
                                           scale=[1] * self.dim1), gen=False)

        self.model = mymodel
        self.disc.set_data_driven()
        self.std = 0.05

        true_value = np.array([0.5])
        self.ans = true_value
        target_noise = self.std * np.random.randn(self.num_obs)
        true_target = mymodel(true_value)
        target = true_target + target_noise
        self.disc.set_data(target, std=self.std)
        self.disc.set_repeated()
