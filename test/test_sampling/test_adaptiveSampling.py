# -*- coding: utf-8 -*-
# Lindley Graham 04/07/2015

"""
This module contains unittests for :mod:`~bet.sampling.adaptiveSampling`
"""

import unittest, os
import numpy.testing as nptest
import numpy as np
import bet.sampling.adaptiveSampling as asam
import scipy.io as sio
from bet.Comm import *
import bet

local_path = os.path.join(os.path.dirname(bet.__file__),
    "../test/test_sampling")

@unittest.skipIf(size > 1, 'Only run in serial')
def test_loadmat_init():
    """
    Tests :meth:`bet.sampling.adaptiveSampling.loadmat` and
    :meth:`bet.sampling.adaptiveSampling.sampler.init`.
    """
    np.random.seed(1)
    chain_length = 10
    mdat1 = {'samples':np.random.random((50, 1)),
            'data':np.random.random((50, 1)), 'num_samples':50,
            'chain_length':chain_length} 
    mdat2 = {'samples':np.random.random((60, 1)),
                    'num_samples':60, 'chain_length':chain_length}
    model = "this is not a model"
    
    num_samples = np.array([50, 60])
    num_chains_pproc1, num_chains_pproc2 = np.ceil(num_samples/float(chain_length*size)).astype('int')
    num_chains1, num_chains2 = size * np.array([num_chains_pproc1,
        num_chains_pproc2])
    num_samples1, num_samples2 = chain_length * np.array([num_chains1,
        num_chains2])
    
    sio.savemat(os.path.join(local_path, 'testfile1'), mdat1)
    sio.savemat(os.path.join(local_path, 'testfile2'), mdat2)

    (loaded_sampler1, samples1, data1) = asam.loadmat(os.path.join(local_path,
        'testfile1'))
    nptest.assert_array_equal(samples1, mdat1['samples'])
    nptest.assert_array_equal(data1, mdat1['data'])
    assert loaded_sampler1.num_samples == num_samples1
    assert loaded_sampler1.chain_length == chain_length
    assert loaded_sampler1.num_chains_pproc == num_chains_pproc1
    assert loaded_sampler1.num_chains == num_chains1
    nptest.assert_array_equal(np.repeat(range(num_chains1), chain_length, 0),
            loaded_sampler1.sample_batch_no)
    assert loaded_sampler1.lb_model == None

    (loaded_sampler2, samples2, data2) = asam.loadmat(os.path.join(local_path,
        'testfile2'), model)
    nptest.assert_array_equal(samples2, mdat2['samples'])
    nptest.assert_array_equal(data2, None)
    assert loaded_sampler2.num_samples == num_samples2
    assert loaded_sampler2.chain_length == chain_length
    assert loaded_sampler2.num_chains_pproc == num_chains_pproc2
    assert loaded_sampler2.num_chains == num_chains2
    nptest.assert_array_equal(np.repeat(range(num_chains2), chain_length, 0),
            loaded_sampler2.sample_batch_no)
    if os.path.exists(os.path.join(local_path, 'testfile1.mat')):
        os.remove(os.path.join(local_path, 'testfile1.mat'))
    if os.path.exists(os.path.join(local_path, 'testfile2.mat')):
        os.remove(os.path.join(local_path, 'testfile2.mat'))

def verify_samples(QoI_range, sampler, param_min, param_max,
        t_set, savefile, initial_sample_type):
    """
    Run :meth:`bet.sampling.adaptiveSampling.sampler.generalized_chains` and
    verify that the samples have the correct dimensions and are containted in
    the bounded parameter space.
    """

    # create indicator function
    Q_ref = QoI_range*0.5
    bin_size = 0.15*QoI_range
    maximum = 1/np.product(bin_size)
    def ifun(outputs):
        """
        Indicator function
        """
        left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
        right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
        left = np.all(np.greater_equal(outputs, left), axis=1)
        right = np.all(np.less_equal(outputs, right), axis=1)
        inside = np.logical_and(left, right)
        max_values = np.repeat(maximum, outputs.shape[0], 0)
        return inside.astype('float64')*max_values
        
    # create rhoD_kernel
    kernel_rD = asam.rhoD_kernel(maximum, ifun)

    # run generalized chains
    (samples, data, all_step_ratios) = sampler.generalized_chains(param_min,
            param_max, t_set, kernel_rD, savefile, initial_sample_type)

    # check dimensions of samples
    assert samples.shape == (sampler.num_samples, len(param_min))

    # are the samples in bounds?
    param_left = np.repeat([param_min], sampler.num_samples, 0)
    param_right = np.repeat([param_max], sampler.num_samples, 0)
    assert np.all(samples <= param_right)
    assert np.all(samples >= param_left)

    # check dimensions of data
    assert data.shape == (sampler.num_samples, len(QoI_range))

    # check dimensions of all_step_ratios
    assert all_step_ratios.shape == (sampler.num_chains, sampler.chain_length)

    # are all the step ratios of an appropriate size?
    assert np.all(all_step_ratios >= t_set.min_ratio)
    assert np.all(all_step_ratios <= t_set.max_ratio)
    
    # did the savefiles get created? (proper number, contain proper keys)
    mdat = {}
    if size > 1:
        mdat = sio.loadmat(os.path.join(local_path, os.path.dirname(savefile),
            "proc{}{}".format(rank, os.path.basename(savefile))))
    else:
        mdat = sio.loadmat(savefile)
    nptest.assert_array_equal(samples, mdat['samples'])
    nptest.assert_array_equal(data, mdat['data'])
    nptest.assert_array_equal(all_step_ratios, mdat['step_ratios'])
    assert sampler.chain_length == mdat['chain_length']
    assert sampler.num_samples == mdat['num_samples']
    assert sampler.num_chains == mdat['num_chains']
    nptest.assert_array_equal(sampler.sample_batch_no,
            np.squeeze(mdat['sample_batch_no']))

class Test_adaptive_sampler(unittest.TestCase):
    """
    Test :class:`bet.sampling.adaptiveSampling.sampler`.
    """

    def setUp(self):
        """
        Set up
        """

        # create 1-1 map
        self.param_min1 = np.zeros((1, ))
        self.param_max1 = np.zeros((1, ))
        def map_1t1(x):
            """
            1 to 1 map
            """
            return x*2.0
        # create 3-1 map
        self.param_min3 = np.zeros((3, ))
        self.param_max3 = np.ones((3, ))
        def map_3t1(x):
            """
            3 to 1 map
            """
            return np.expand_dims(np.sum(x, 1), axis=1)
        # create 3-2 map
        def map_3t2(x):
            """
            3 to 2 map
            """
            return np.vstack(([x[:, 0]+x[:, 1], x[:, 2]])).transpose()
        # create 10-4 map
        self.param_min10 = np.zeros((10, ))
        self.param_max10 = np.ones((10, ))
        def map_10t4(x):
            """
            10 to 4 map
            """
            x1 = x[:, 0] + x[:, 1]
            x2 = x[:, 2] + x[:, 3]
            x3 = x[:, 4] + x[:, 5]
            x4 = np.sum(x[:, [6, 7, 8, 9]], 1)
            return np.vstack([x1, x2, x3, x4]).transpose()
        self.savefiles = ["11t11", "1t1", "3to1", "3to2", "10to4"]
        self.models = [map_1t1, map_1t1, map_3t1, map_3t2, map_10t4]
        self.QoI_range = [np.array([2.0]), np.array([2.0]), np.array([3.0]),
                np.array([2.0, 1.0]), np.array([2.0, 2.0, 2.0, 4.0])]

        # define parameters for the adaptive sampler

        num_samples = 1000
        chain_length = 100
        num_chains_pproc = int(np.ceil(num_samples/float(chain_length*size)))
        num_chains = size * num_chains_pproc
        num_samples = chain_length * np.array(num_chains)

        self.samplers = []
        for model in self.models:
            self.samplers.append(asam.sampler(num_samples, chain_length,
                model))

        self.param_min_list = [self.param_min1, self.param_min1,
                self.param_min3, self.param_min3, self.param_min10]
        self.param_max_list = [self.param_max1, self.param_max1,
                self.param_max3, self.param_max3, self.param_max10]

        self.test_list = zip(self.models, self.QoI_range, self.samplers,
                self.param_min_list, self.param_max_list, self.savefiles)


    def tearDown(self):
        for f in self.savefiles:
            if os.path.exists(f+".mat"):
                os.remove(f+".mat")
        if size > 1:
            for f in self.savefiles:
                proc_savefile = os.path.join(local_path, os.path.dirname(f),
                        "proc{}{}.mat".format(rank, os.path.basename(f)))
                print proc_savefile
                if os.path.exists(proc_savefile):
                    os.remove(proc_savefile)
                proc_savefile = os.path.join(local_path, os.path.dirname(f),
                        "p{}proc{}{}.mat".format(rank, rank, os.path.basename(f)))
                if os.path.exists(proc_savefile):
                    os.remove(proc_savefile)
                print proc_savefile

    def test_update(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.save`
        """
        mdict = {"frog":3, "moose":2}
        self.samplers[0].update_mdict(mdict)
        assert self.samplers[0].num_samples == mdict["num_samples"]
        assert self.samplers[0].chain_length == mdict["chain_length"]
        assert self.samplers[0].num_chains == mdict["num_chains"]
        nptest.assert_array_equal(self.samplers[0].sample_batch_no,
                np.repeat(range(self.samplers[0].num_chains),
                    self.samplers[0].chain_length, 0))
    
    def test_run_gen(self):
        """
        Run :meth:`bet.sampling.adaptiveSampling.sampler.run_gen` and verify
        that the output has the correct dimensions.
        """
        # sampler.run_gen(kern_list, rho_D, maximum, param_min, param_max,
        # t_set, savefile, initial_sample_type)
        # returns list where each member is a tuple ((samples, data),
        # all_step_ratios, num_high_prob_samples,
        # sorted_indices_of_num_high_prob_samples, average_step_ratio)
            # create indicator function
        inputs = self.test_list[3]
        _, QoI_range, sampler, param_min, param_max, savefile = inputs
        
        Q_ref = QoI_range*0.5
        bin_size = 0.15*QoI_range
        maximum = 1/np.product(bin_size)
        def ifun(outputs):
            """
            Indicator function
            """
            inside = np.logical_and(np.all(np.greater_equal(outputs,
                Q_ref-.5*bin_size), axis=1), np.all(np.less_equal(outputs,
                    Q_ref+.5*bin_size), axis=1)) 
            max_values = np.repeat(maximum, outputs.shape[0], 0)
            return inside.astype('float64')*max_values

        # create rhoD_kernel
        kernel_rD = asam.rhoD_kernel(maximum, ifun)
        kern_list = [kernel_rD]*2

        # create t_set
        t_set = asam.transition_set(.5, .5**5, 1.0) 

        # run run_gen
        output = sampler.run_gen(kern_list, ifun, maximum, param_min,
                param_max, t_set, savefile)

        results, r_step_size, results_rD, sort_ind, mean_ss = output

        for out in output:
            assert len(out) == 2

        for samples, data in results:
            assert samples.shape == (sampler.num_samples, len(param_min))
            assert data.shape == (sampler.num_samples, len(QoI_range))
        for step_sizes in r_step_size:
            assert step_sizes.shape == (sampler.num_chains,
                    sampler.chain_length) 
        for num_hps in results_rD:
            assert type(num_hps) == int
        for inds in sort_ind:
            assert np.issubdtype(type(inds), int)
        for asr in mean_ss:
            assert asr > t_set.min_ratio
            assert asr < t_set.max_ratio
    
    def test_run_tk(self):
        """
        Run :meth:`bet.sampling.adaptiveSampling.sampler.run_tk` and verify
        that the output has the correct dimensions.
        """
        # sampler.run_tk(init_ratio, min_raio, max_ratio, rho_D, maximum,
        # param_min, param_max, kernel, savefile, intial_sample_type)
        # returns list where each member is a tuple ((samples, data),
        # all_step_ratios, num_high_prob_samples,
        # sorted_indices_of_num_high_prob_samples, average_step_ratio)
        inputs = self.test_list[3]
        _, QoI_range, sampler, param_min, param_max, savefile = inputs
        
        Q_ref = QoI_range*0.5
        bin_size = 0.15*QoI_range
        maximum = 1/np.product(bin_size)
        def ifun(outputs):
            """
            Indicator function
            """
            inside = np.logical_and(np.all(np.greater_equal(outputs,
                Q_ref-.5*bin_size), axis=1), np.all(np.less_equal(outputs,
                    Q_ref+.5*bin_size), axis=1)) 
            max_values = np.repeat(maximum, outputs.shape[0], 0)
            return inside.astype('float64')*max_values

        # create rhoD_kernel
        kernel_rD = asam.rhoD_kernel(maximum, ifun)

        # create t_set
        init_ratio = [1.0, .5, .25]
        min_ratio = [.5**2, .5**5, .5**7]
        max_ratio = [1.0, .75, .5]

        # run run_gen
        output = sampler.run_tk(init_ratio, min_ratio, max_ratio, ifun, maximum, param_min,
                param_max, kernel_rD, savefile)
        
        results, r_step_size, results_rD, sort_ind, mean_ss = output

        for out in output:
            assert len(out) == 3

        for samples, data in results:
            assert samples.shape == (sampler.num_samples, len(param_min))
            assert data.shape == (sampler.num_samples, len(QoI_range))
        for step_sizes in r_step_size:
            assert step_sizes.shape == (sampler.num_chains,
                    sampler.chain_length) 
        for num_hps in results_rD:
            assert type(num_hps) == int
        for inds in sort_ind:
            assert np.issubdtype(type(inds), int)
        for asr, mir, mar in zip(mean_ss, min_ratio, max_ratio):
            assert asr > mir
            assert asr < mar

    def test_run_inc_dec(self):
        """
        Run :meth:`bet.sampling.adaptiveSampling.sampler.run_inc_dec` and verify
        that the output has the correct dimensions.
        """
        # sampler.run_inc_dec(increase, decrease, tolerance, rho_D, maximum,
        # param_min, param_max, t_set, savefile, initial_sample_type)
        # returns list where each member is a tuple ((samples, data),
        # all_step_ratios, num_high_prob_samples,
        # sorted_indices_of_num_high_prob_samples, average_step_ratio)
        inputs = self.test_list[3]
        _, QoI_range, sampler, param_min, param_max, savefile = inputs
        
        Q_ref = QoI_range*0.5
        bin_size = 0.15*QoI_range
        maximum = 1/np.product(bin_size)
        def ifun(outputs):
            """
            Indicator function
            """
            inside = np.logical_and(np.all(np.greater_equal(outputs,
                Q_ref-.5*bin_size), axis=1), np.all(np.less_equal(outputs,
                    Q_ref+.5*bin_size), axis=1)) 
            max_values = np.repeat(maximum, outputs.shape[0], 0)
            return inside.astype('float64')*max_values

        # create rhoD_kernel
        increase = [2.0, 3.0, 5.0]
        decrease = [.7, .5, .2]
        tolerance = [1e-3, 1e-4, 1e-7]

        # create t_set
        t_set = asam.transition_set(.5, .5**5, 1.0) 

        # run run_gen
        output = sampler.run_inc_dec(increase, decrease, tolerance, ifun, maximum, param_min,
                param_max, t_set, savefile)

        results, r_step_size, results_rD, sort_ind, mean_ss = output

        for out in output:
            assert len(out) == 3

        for samples, data in results:
            assert samples.shape == (sampler.num_samples, len(param_min))
            assert data.shape == (sampler.num_samples, len(QoI_range))
        for step_sizes in r_step_size:
            assert step_sizes.shape == (sampler.num_chains,
                    sampler.chain_length) 
        for num_hps in results_rD:
            assert type(num_hps) == int
        for inds in sort_ind:
            assert np.issubdtype(type(inds), int)
        for asr in mean_ss:
            assert asr > t_set.min_ratio
            assert asr < t_set.max_ratio

    def test_generalized_chains(self):
        """
        Test :met:`bet.sampling.adaptiveSampling.sampler.generalized_chains`
        for three different QoI maps (1 to 1, 3 to 1, 3 to 2, 10 to 4).
        """
        # create a transition set
        t_set = asam.transition_set(.5, .5**5, 1.0) 

        for _, QoI_range, sampler, param_min, param_max, savefile in self.test_list:
            for initial_sample_type in ["random", "r", "lhs"]:
                verify_samples(QoI_range, sampler, param_min, param_max, t_set,
                        savefile, initial_sample_type)

class test_kernels(unittest.TestCase):
    """
    Tests kernels for a 1d, 2d, 4d data space.
    """
    def setUp(self):
        """
        Set up
        """
        self.QoI_range = [np.array([3.0]),
                np.array([2.0, 1.0]), np.array([2.0, 2.0, 2.0, 4.0])]

    def test_list(self):
        """
        Run test for a 1d, 2d, and 4d data space.
        """
        for QoI_range in self.QoI_range:
            Q_ref = QoI_range*0.5
            bin_size = 0.15*QoI_range
            maximum = 1/np.product(bin_size)
            def ifun(outputs):
                """
                Indicator function
                """
                inside = np.logical_and(np.all(np.greater_equal(outputs,
                    Q_ref-.5*bin_size), axis=1), np.all(np.less_equal(outputs,
                        Q_ref+.5*bin_size), axis=1)) 
                max_values = np.repeat(maximum, outputs.shape[0], 0)
                return inside.astype('float64')*max_values
            self.verify_indiv(Q_ref, ifun, maximum)

    def verify_indiv(self, Q_ref, rhoD, maximum):
        """
        Test that the list of kernels is correctly created.
        """
        kern_list = asam.kernels(Q_ref, rhoD, maximum)
        assert len(kern_list) == 3
        assert type(kern_list[0]) == asam.maxima_mean_kernel
        assert type(kern_list[1]) == asam.rhoD_kernel
        assert type(kern_list[2]) == asam.maxima_kernel

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
        bin_size = 0.15*self.data_domain[:, 1]
        self.maximum = 1/np.product(bin_size)
        def ifun(outputs):
            """
            Indicator function
            """
            inside = np.logical_and(np.all(np.greater_equal(outputs,
                self.Q_ref-.5*bin_size), axis=1), np.all(np.less_equal(outputs,
                    self.Q_ref+.5*bin_size), axis=1)) 
            max_values = np.repeat(self.maximum, outputs.shape[0], 0)
            return inside.astype('float64')*max_values
        self.rho_D = ifun

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
        bin_size = 0.15*self.data_domain[:, 1]
        self.maximum = 1/np.product(bin_size)
        def ifun(outputs):
            """
            Indicator function
            """
            inside = np.logical_and(np.all(np.greater_equal(outputs,
                self.Q_ref-.5*bin_size), axis=1), np.all(np.less_equal(outputs,
                    self.Q_ref+.5*bin_size), axis=1)) 
            max_values = np.repeat(self.maximum, outputs.shape[0], 0)
            return inside.astype('float64')*max_values
        self.rho_D = ifun


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
        bin_size = 0.15*self.data_domain[:, 1]
        self.maximum = 1/np.product(bin_size)
        def ifun(outputs):
            """
            Indicator function
            """
            inside = np.logical_and(np.all(np.greater_equal(outputs,
                self.Q_ref-.5*bin_size), axis=1), np.all(np.less_equal(outputs,
                    self.Q_ref+.5*bin_size), axis=1)) 
            max_values = np.repeat(self.maximum, outputs.shape[0], 0)
            return inside.astype('float64')*max_values
        self.rho_D = ifun


class kernel(object):
    """
    Test :class:`bet.sampling.adaptiveSampling.kernel`
    """
    def setUp(self):
        """
        Set up
        """
        self.kernel = asam.kernel()

    def test_init(self):
        """
        Test the initalization of :class:`bet.sampling.adaptiveSampling.kernel`
        """
        assert self.kernel.TOL == 1e-8
        assert self.kernel.increase == 1.0
        assert self.kernel.decrease == 1.0

    def test_delta_step(self):
        """
        Test the delta_step method of
        :class:`bet.sampling.adaptiveSampling.kernel`
        """
        kern_new, proposal = self.kernel.delta_step(self.data)
        assert kern_new == None
        assert proposal.shape == (self.data.shape[0],)
        

class test_kernel_1D(kernel, data_1D):
    """
    Test :class:`bet.sampling.adaptiveSampling.kernel` on a 1D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_kernel_1D, self).createData()
        super(test_kernel_1D, self).setUp()
      
class test_kernel_2D(kernel, data_2D):
    """
    Test :class:`bet.sampling.adaptiveSampling.kernel` on a 2D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_kernel_2D, self).createData()
        super(test_kernel_2D, self).setUp()
      
class test_kernel_3D(kernel, data_3D):
    """
    Test :class:`bet.sampling.adaptiveSampling.kernel` on a 3D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_kernel_3D, self).createData()
        super(test_kernel_3D, self).setUp()
      

class rhoD_kernel(kernel):
    """
    Test :class:`bet.sampling.adaptiveSampling.rhoD_kernel`
    """
    def setUp(self):
        """
        Set up
        """
        self.kernel = asam.rhoD_kernel(self.maximum, self.rho_D)

    def test_init(self):
        """
        Test the initalization of :class:`bet.sampling.adaptiveSampling.rhoD_kernel`
        """
        assert self.kernel.TOL == 1e-8
        assert self.kernel.increase == 2.0
        assert self.kernel.decrease == 0.5
        assert self.kernel.MAX == self.maximum
        assert self.kernel.rho_D == self.rho_D
        assert self.kernel.sort_ascending == False

    def test_delta_step(self):
        """
        Test the delta_step method of
        :class:`bet.sampling.adaptiveSampling.rhoD_kernel`
        """
        kern_new, proposal = self.kernel.delta_step(self.data)
        nptest.assert_array_equal(kern_new, self.rho_D(self.data))
        assert proposal == None 
        
        data = np.vstack([self.Q_ref+3.0, self.Q_ref, self.Q_ref-3.0])
        data_new = np.vstack([self.Q_ref, self.Q_ref+3.0, self.Q_ref-3.0])
        kern_old = self.rho_D(data)
        kern_new, proposal = self.kernel.delta_step(data_new, kern_old)
        nptest.assert_array_equal(proposal, [0.5, 2.0, 1.0])

class test_rhoD_kernel_1D(rhoD_kernel, data_1D):
    """
    Test :class:`bet.sampling.adaptiveSampling.rhoD_kernel` on a 1D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_rhoD_kernel_1D, self).createData()
        super(test_rhoD_kernel_1D, self).setUp()
      
class test_rhoD_kernel_2D(rhoD_kernel, data_2D):
    """
    Test :class:`bet.sampling.adaptiveSampling.rhoD_kernel` on a 2D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_rhoD_kernel_2D, self).createData()
        super(test_rhoD_kernel_2D, self).setUp()
      
class test_rhoD_kernel_3D(rhoD_kernel, data_3D):
    """
    Test :class:`bet.sampling.adaptiveSampling.rhoD_kernel` on a 3D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_rhoD_kernel_3D, self).createData()
        super(test_rhoD_kernel_3D, self).setUp()

class maxima_kernel(kernel):
    """
    Test :class:`bet.sampling.adaptiveSampling.maxima_kernel`
    """
    def setUp(self):
        """
        Set up
        """
        self.kernel = asam.maxima_kernel(np.vstack([self.Q_ref, self.Q_ref+.5]), self.rho_D)

    def test_init(self):
        """
        Test the initalization of :class:`bet.sampling.adaptiveSampling.maxima_kernel`
        """
        assert self.kernel.TOL == 1e-8
        assert self.kernel.increase == 2.0
        assert self.kernel.decrease == 0.5
        nptest.assert_equal(self.kernel.MAXIMA, np.vstack([self.Q_ref,
            self.Q_ref+.5]))
        assert self.kernel.num_maxima == 2
        nptest.assert_equal(self.kernel.rho_max,
                self.rho_D(np.vstack([self.Q_ref, self.Q_ref+.5])))
        assert self.kernel.sort_ascending == True

    def test_delta_step(self):
        """
        Test the delta_step method of
        :class:`bet.sampling.adaptiveSampling.maxima_kernel`
        """
        data_old = np.vstack([self.Q_ref+3.0, self.Q_ref, self.Q_ref-3.0])
        kern_old, proposal = self.kernel.delta_step(data_old)

        # TODO: check kern_old
        #nptest.assert_array_equal(kern_old, np.zeros((self.data.shape[0],))
        assert proposal == None 
        
        data_new = np.vstack([self.Q_ref, self.Q_ref+3.0, self.Q_ref-3.0])
        kern_new, proposal = self.kernel.delta_step(data_new, kern_old)

        #TODO: check kern_new
        #nptest.assert_array_eqyal(kern_new, something)
        nptest.assert_array_equal(proposal, [0.5, 2.0, 1.0])

class test_maxima_kernel_1D(maxima_kernel, data_1D):
    """
    Test :class:`bet.sampling.adaptiveSampling.maxima_kernel` on a 1D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_maxima_kernel_1D, self).createData()
        super(test_maxima_kernel_1D, self).setUp()
      
class test_maxima_kernel_2D(maxima_kernel, data_2D):
    """
    Test :class:`bet.sampling.adaptiveSampling.maxima_kernel` on a 2D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_maxima_kernel_2D, self).createData()
        super(test_maxima_kernel_2D, self).setUp()
      
class test_maxima_kernel_3D(maxima_kernel, data_3D):
    """
    Test :class:`bet.sampling.adaptiveSampling.maxima_kernel` on a 3D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_maxima_kernel_3D, self).createData()
        super(test_maxima_kernel_3D, self).setUp()


class maxima_mean_kernel(maxima_kernel):
    """
    Test :class:`bet.sampling.adaptiveSampling.maxima_mean_kernel`
    """
    def setUp(self):
        """
        Set up
        """
        self.kernel = asam.maxima_mean_kernel(np.vstack([self.Q_ref, self.Q_ref+.5]), self.rho_D)

    def test_init(self):
        """
        Test the initalization of
        :class:`bet.sampling.adaptiveSampling.maxima_mean_kernel`
        """
        assert self.kernel.radius == None
        assert self.kernel.mean == None
        assert self.kernel.current_clength == 0
        super(maxima_mean_kernel, self).test_init()

    def test_reset(self):
        """
        Test the method
        :meth:`bet.sampling.adaptiveSampling.maxima_mean_kernel.reset`
        """
        self.kernel.reset()
        assert self.kernel.radius == None
        assert self.kernel.mean == None
        assert self.kernel.current_clength == 0

    def test_delta_step(self):
        """
        Test the delta_step method of
        :class:`bet.sampling.adaptiveSampling.maxima_mean_kernel`
        """
        super(maxima_mean_kernel, self).test_delta_step()
        # TODO
        # check self.current_clength
        # check self.radius
        # check self.mean
        
class test_maxima_mean_kernel_1D(maxima_mean_kernel, data_1D):
    """
    Test :class:`bet.sampling.adaptiveSampling.maxima_mean_kernel` on a 1D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_maxima_mean_kernel_1D, self).createData()
        super(test_maxima_mean_kernel_1D, self).setUp()
      
class test_maxima_mean_kernel_2D(maxima_mean_kernel, data_2D):
    """
    Test :class:`bet.sampling.adaptiveSampling.maxima_mean_kernel` on a 2D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_maxima_mean_kernel_2D, self).createData()
        super(test_maxima_mean_kernel_2D, self).setUp()
      
class test_maxima_mean_kernel_3D(maxima_mean_kernel, data_3D):
    """
    Test :class:`bet.sampling.adaptiveSampling.maxima_mean_kernel` on a 3D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_maxima_mean_kernel_3D, self).createData()
        super(test_maxima_mean_kernel_3D, self).setUp()


class transition_set(object):
    """
    Tests :class:`bet.sampling.adaptiveSamplinng.transition_set`
    """
    def setUp(self):
        """
        Set Up
        """
        self.t_set = asam.transition_set(.5, .5**5, 1.0) 

    def test_init(self):
        """
        Tests the initialization of :class:`bet.sampling.adaptiveSamplinng.transition_set`
        """
        assert self.t_set.init_ratio == .5
        assert self.t_set.min_ratio == .5**5
        assert self.t_set.max_ratio == 1.0

    def test_step(self):
        """
        Tests the method
        :meth:`bet.sampling.adaptiveSampling.transition_set.step`
        """
        # define step_ratio, param_width, param_left, param_right, samples_old
        # from data
        param_left = np.repeat([self.data_domain[:, 0]], self.data.shape[0], 0)
        param_right = np.repeat([self.data_domain[:, 1]], self.data.shape[0], 0)
        param_width = param_right - param_left

        step_ratio = 0.5*np.ones(self.data.shape[0],)
        step_ratio[self.data.shape[0]/2:] = .1
        step_size = np.repeat([step_ratio], param_width.shape[1],
                0).transpose()*param_width
        # take a step
        samples_new = self.t_set.step(step_ratio, param_width, param_left,
                param_right, self.data)

        # make sure the proposed steps are inside the domain
        # check dimensions of samples
        assert samples_new.shape == self.data.shape

        # are the samples in bounds?
        assert np.all(samples_new <= param_right)
        assert np.all(samples_new >= param_left)

        # make sure the proposed steps are inside the box defined around their
        # generating old samples
        assert np.all(samples_new <= self.data+0.5*step_size)
        assert np.all(samples_new >= self.data-0.5*step_size)


class test_transition_set_1D(transition_set, data_1D):
    """
    Test :class:`bet.sampling.adaptiveSampling.transition_set` on a 1D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_transition_set_1D, self).createData()
        super(test_transition_set_1D, self).setUp()
      
class test_transition_set_2D(transition_set, data_2D):
    """
    Test :class:`bet.sampling.adaptiveSampling.transition_set` on a 2D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_transition_set_2D, self).createData()
        super(test_transition_set_2D, self).setUp()
      
class test_transition_set_3D(transition_set, data_3D):
    """
    Test :class:`bet.sampling.adaptiveSampling.transition_set` on a 3D data space.
    """
    def setUp(self):
        """
        Set up
        """
        super(test_transition_set_3D, self).createData()
        super(test_transition_set_3D, self).setUp()

