"""
This module contains unittests for :mod:`~bet.util`
"""

import unittest, os
import numpy.testing as nptest
import numpy as np
import bet.sampling.adaptiveSampling as asam
import scipy.io as sio

"""
TODO: rewrite loadmat test, rewrite init test, rewrite update_mdict test, write
tests for run_BLAHBLAH, write test for generalized_chains, write a test stub
for kernels but do not implement
"""

def test_loadmat_init():
    """
    Tests :meth:`bet.sampling.adaptiveSampling.loadmat` and
    :meth:`bet.sampling.adaptiveSampling.sampler.init`.
    """
    chain_length = 10
    mdat1 = {'samples':np.random.random((5,1)),
            'data':np.random.random((5,1)), 'num_samples':50,
            'chain_length':chain_length} 
    mdat2 = {'samples':np.random.random((6,1)),
                    'num_samples':60, 'chain_length':chain_length}
    model = "this is not a model"
    
    num_samples = np.array([50, 60])
    num_chains_pproc1, num_chains_pproc2 = int(np.ceil(num_samples/float(chain_length*size)))
    num_chains1, num_chains2 = size * num_chains_pproc
    num_samples1, num_samples2 = chain_length * np.array([num_chains1,
        num_chains2])

    sio.savemat('testfile1', mdat1)
    sio.savemat('testfile2', mdat2)

    (loaded_sampler1, samples1, data1) = asam.loadmat('testfile1')
    nptest.assert_array_equal(samples1, mdat1['samples'])
    nptest.assert_array_equal(data1, mdat1['data'])
    assert loaded_sampler1.num_samples == num_samples1
    assert loaded_sampler1.chain_length == chain_length
    assert loaded_sampler1.num_chains_pproc == num_chains_pproc1
    assert loaded_sampler1.num_chains = num_chains1
    nptest.assert_array_equal(np.repeat(range(num_chains1), chain_length, 0),
            loaded_sampler1.sample_batch_no)
    assert loaded_sampler1.lb_model == None

    (loaded_sampler2, samples2, data2) = asam.loadmat('testfile2', model)
    nptest.assert_array_equal(samples2, mdat2['samples'])
    nptest.assert_array_equal(data2, None)
    assert loaded_sampler2.num_samples == num_samples2
    assert loaded_sampler2.chain_length == chain_length
    assert loaded_sampler2.num_chains_pproc == num_chains_pproc2
    assert loaded_sampler2.num_chains = num_chains2
    nptest.assert_array_equal(np.repeat(range(num_chains2), chain_length, 0),
            loaded_sampler2.sample_batch_no)
    os.remove('testfile1.mat')
    os.remove('testfile2.mat')

def verify_random_samples(model, QoI_range, sampler, param_min, param_max,
        t_set, savefile, initial_sample_type):

    # create indicator function
    Q_ref = QoI_range*0.5
    bin_size = 0.15*QoI_range
    maxiumum = 1/np.product(bin_size)
    def ifun(outputs):
        left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
        right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
        left = np.all(np.greater_equal(outputs, left), axis=1)
        right = np.all(np.less_equal(outputs, right), axis=1)
        inside = np.logial_and(left, right)
        max_values = np.repeate(maxium, outputs.shape[0], 0)
        return inside.astype('float64')*max_values
        
    # create rhoD_kernel
    kernel_rD = asam.rhoD_kernel(maxium, ifun)

    # run generalized chains
    (samples, data, all_step_ratios) = sampler.generalized_chains(param_min,
            param_max, t_set, kernel_rD, savefile, inital_sample_type)

    # check dimensions of samples
    assert samples.shape == (sampler.num_samples, len(param_min))

    # are the samples in bounds?
    param_left = np.repeat([param_min], num_samples, 0)
    param_right = np.repeat([param_max], num_samples, 0)
    assert np.all(my_samples <= param_right)
    assert np.all(my_samples >= param_left)

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
        mdat =
        sio.loadmat(os.path.join(os.path.dirname(savefile),"proc{}{}".format(rank,
            os.path.basename(savefile))))
    else:
        mdat = sio.loadmat(savefile)
    nptest.assert_array_equal(samples, mdat['samples'])
    nptest.assert_array_equal(data, mdat['data'])
    nptest.assert_array_equal(all_step_ratios, mdat['step_ratios'])
    assert sampler.chain_length == mdat['chain_length']
    assert sampler.num_samples == mdat['num_samples']
    assert sampler.num_chains == mdat['num_chains']
    nptest.assert_array_equal(sampler.sample_batch_no,
            mdat['sampler_batch_no'])

class Test_basic_sampler(unittest.TestCase):
    """
    Test :class:`bet.sampling.adaptiveSampling.sampler`.
    """

    def setUp(self):
        # create 1-1 map
        self.param_min1 = np.zeros((1, ))
        self.param_max1 = np.zeros((1, ))
        def map_1t1(x):
            return x*2.0
        # create 3-1 map
        self.param_min3 = np.zeros((3, ))
        self.param_max3 = np.ones((3, ))
        def map_3t1(x):
            return np.sum(x, 1)
        # create 3-2 map
        def map_3t2(x):
            return np.vstack(([x[:, 0]+x[:, 1], x[:, 2]])).transpose()
        # create 10-4 map
        self.param_min10 = np.zeros((10, ))
        self.param_min10 = np.ones((10, ))
        def map_10t4(x):
            x1 = x[:, 0] + x[:, 1]
            x2 = x[:, 2] + x[:, 3]
            x3 = x[:, 4] + x[:, 5]
            x4 = np.sum(x[:, [6, 7, 8, 9]], 1)
            return np.vstack([x1, x2, x3, x4]).transpose()
        self.savefiles = ["11t11", "1t1", "3to1", "3to2", "10to4"]
        self.models = [map_1t1, map_1t1, map_3t1, map_3t2, map_10t4]
        self.QoI_range = [np.array([2.0]), np.array([2.0]), np.array([3.0]),
                np.array([2.0, 1.0]), np.array([2.0, 2.0, 2.0, 4.0])

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

    def tearDown(self):
        for f in self.savefiles:
            if os.path.exists(f+".mat"):
                os.remove(f+".mat")
        if size > 1:
            for f in self.savefiles:
                for proc in range(size):
                    proc_savefile = os.path.join(os.path.dirname(f),
                            "proc{}{}".format(rank,
                                os.path.basename(savefile)))

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
                np.repeat(range(num_chains), chain_length, 0))
    
    @unittest.skip("Implement me")
    def test_run_gen(self):
        # sampler.run_gen(kern_list, rho_D, maximum, param_min, param_max,
        # t_set, savefile, initial_sample_type)
        # returns list where each member is a tuple ((samples, data),
        # all_step_ratios, num_high_prob_samples,
        # sorted_indices_of_num_high_prob_samples, average_step_ratio)
        pass
    @unittest.skip("Implement me")
    def test_run_tk(self):
        # sampler.run_tk(init_ratio, min_raio, max_ratio, rho_D, maximum,
        # param_min, param_max, kernel, savefile, intial_sample_type)
        # returns list where each member is a tuple ((samples, data),
        # all_step_ratios, num_high_prob_samples,
        # sorted_indices_of_num_high_prob_samples, average_step_ratio)
        pass
    @unittest.skip("Implement me")
    def test_run_inc_dec(self):
        # sampler.run_inc_dec(increase, decrease, tolerance, rho_D, maximum,
        # param_min, param_max, t_set, savefile, initial_sample_type)
        # returns list where each member is a tuple ((samples, data),
        # all_step_ratios, num_high_prob_samples,
        # sorted_indices_of_num_high_prob_samples, average_step_ratio)
        pass

    def test_generalized_chains(self):
        """
        Test :met:`bet.sampling.adaptiveSampling.sampler.generalized_chains`
        for three different QoI maps (1 to 1, 3 to 1, 3 to 2, 10 to 4).
        """
        param_min_list = [self.param_min1, self.param_min1, self.param_min3,
            self.param_min3, self.param_min10]
        param_max_list = [self.param_max1, self.param_max1, self.param_max3,
            self.param_max3, self.param_max10]
        # create a transition set
        t_set = asam.transition_set(.5, .5**5, 1.0) 

        test_list = zip(self.models, self.QoI_range, self.samplers,
                param_min_list, param_max_list, self.savefiles)

        for model, sampler, param_min, param_max, savefile, rho_D in test_list:
            for initial_sample_type in ["random", "r", "lhs"]:
                        yield verify_samples, model, QoI_range, sampler,
                        param_min, param_max, t_set, savefile, initial_sample_type

@unittest.skip("Implement me")
def test_kernels():
        pass

@unittest.skip("Implement me")
class test_transition_set(unittest):
    @unittest.skip("Implement me")
    def test_init():
        pass
    @unittest.skip("Implement me")
    def test_step():
        pass

@unittest.skip("Implement me")
class kernel(unittest):
    @unittest.skip("Implement me")
    def test_init():
        pass
    def test_delta_step():
        pass

@unittest.skip("Implement me")
class test_rhoD_kernsl(kernel):
    pass

@unittest.skip("Implement me")
class test_maxima_kernsl(kernel):
    pass

@unittest.skip("Implement me")
class test_maxima_mean_kernsl(kernel):
    pass

@unittest.skip("Implement me")
class test_multi_dist_kernsl(kernel):
    pass

