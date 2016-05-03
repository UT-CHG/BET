# Copyright (C) 2014-2015 The BET Development Team

# Lindley Graham 04/07/2015
"""
This module contains unittests for :mod:`~bet.sampling.basicSampling:`
"""

import unittest, os, pyDOE
import numpy.testing as nptest
import numpy as np
import scipy.io as sio
import bet
import bet.sampling.basicSampling as bsam
from bet.Comm import comm 
import bet.sample
from bet.sample import sample_set
from bet.sample import discretization as disc 

local_path = os.path.join(os.path.dirname(bet.__file__),
    "../test/test_sampling")


@unittest.skipIf(comm.size > 1, 'Only run in serial')
def test_loadmat():
    """
    Tests :meth:`bet.sampling.basicSampling.loadmat`
    """
    np.random.seed(1)
    mdat1 = {'num_samples':5}
    mdat2 = {'num_samples':6}
    model = "this is not a model"

    my_input1 = sample_set(1)
    my_input1.set_values(np.random.random((5,1)))
    my_output = sample_set(1)
    my_output.set_values(np.random.random((5,1)))
    my_input2 = sample_set(1)
    my_input2.set_values(np.random.random((6,1)))


    sio.savemat(os.path.join(local_path, 'testfile1'), mdat1)
    sio.savemat(os.path.join(local_path, 'testfile2'), mdat2)

    
    bet.sample.save_discretization(disc(my_input1, my_output),
            (os.path.join(local_path, 'testfile1'))
    bet.sample.save_discretization(disc(my_input2, None),
            os.path.join(local_path, 'testfile2'), "NAME")

    (loaded_sampler1, discretization1) = bsam.loadmat(os.path.join(local_path,
        'testfile1'))
    nptest.assert_array_equal(discretization1._input_sample_set.get_values(),
            my_input1.get_values())
    nptest.assert_array_equal(discretization1._output_sample_set.get_values(),
            my_output.get_values())
    assert loaded_sampler1.num_samples == 5
    assert loaded_sampler1.lb_model is None

    (loaded_sampler2, discretization2) = bsam.loadmat(os.path.join(local_path,
        'testfile2'), disc_name="NAME", model=model)
    nptest.assert_array_equal(discretization2._input_sample_set.get_values(),
            my_input2.get_values())
    assert discretization2._output_sample_set is None
    assert loaded_sampler2.num_samples == 6
    assert loaded_sampler2.lb_model == model
    if os.path.exists(os.path.join(local_path, 'testfile1.mat')):
        os.remove(os.path.join(local_path, 'testfile1.mat'))
    if os.path.exists(os.path.join(local_path, 'testfile2.mat')):
        os.remove(os.path.join(local_path, 'testfile2.mat'))

def verify_user_samples(model, sampler, input_sample_set, savefile, parallel):
    """
    Verify that the user samples are correct.
    """
    # evalulate the model at the samples directly
    output_values = (model(input_sample_set._values))
    if len(output_values.shape) == 1:
        output_sample_set = sample_set(1)
    else:
        output_sample_set = sample_set(output_values.shape[1])
    output_sample_set.set_values(output_values)
    discretization = disc(input_sample_set, output_sample_set)

    # evaluate the model at the samples
    my_discretization = sampler.user_samples(input_sample_set, savefile,
            parallel) 
    my_num = my_discretization.check_nums() 

    # compare the samples
    nptest.assert_array_equal(my_discretization._input_sample_set.get_values(),
            discretization._input_sample_set.get_values())
    # compare the data
    nptest.assert_array_equal(my_discretization._output_sample_set.get_values(),
            discretization._output_sample_set.get_values())

    # did num_samples get updated?
    assert my_num == sampler.num_samples
    
    # did the file get correctly saved?
    if comm.rank == 0:
        saved_disc = bet.sample.load_discretization(savefile)
        
        # compare the samples
        nptest.assert_array_equal(my_discretization._input_sample_set.get_values(),
            saved_disc._input_sample_set.get_values())
        # compare the data
        nptest.assert_array_equal(my_discretization._output_sample_set.get_values(),
           saved_disc._output_sample_set.get_values())
        
    comm.Barrier()

def verify_random_samples(model, sampler, sample_type, input_domain,
        num_samples, savefile, parallel):
    np.random.seed(1)
    # recreate the samples
    if num_samples is None:
        num_samples = sampler.num_samples
    
    input_sample_set = sample_set(input_domain.shape[0])
    input_sample_set.set_domain(input_domain)
    
    input_left = np.repeat([input_domain[:, 0]], num_samples, 0)
    input_right = np.repeat([input_domain[:, 1]], num_samples, 0)
    
    input_values = (input_right-input_left)
    if sample_type == "lhs":
        input_values = input_values * pyDOE.lhs(input_sample_set.get_dim(),
                num_samples, 'center') 
    elif sample_type == "random" or "r":
        input_values = input_values * np.random.random(input_left.shape)
    input_values = input_values + input_left
    input_sample_set.set_values(input_values)
    
    # evalulate the model at the samples directly
    output_values = (model(input_sample_set._values))
    if len(output_values.shape) == 1:
        output_sample_set = sample_set(1)
    else:
        output_sample_set = sample_set(output_values.shape[1])
    output_sample_set.set_values(output_values)

    # evaluate the model at the samples
    # reset the random seed
    np.random.seed(1)

    # evaluate the model at the samples
    my_discretization = sampler.random_samples(sample_type, input_domain,
            savefile, num_samples=num_samples, parallel=parallel)
    my_num = my_discretization.check_nums() 
    
    # make sure that the samples are within the boundaries
    assert np.all(my_discretization._input_sample_set._values <= input_right)
    assert np.all(my_discretization._input_sample_set._values >= input_left)

    # compare the samples
    nptest.assert_array_equal(input_sample_set._values,
            my_discretization._input_sample_set._values)
    # compare the data
    nptest.assert_array_equal(output_sample_set._values,
            my_discretization._output_sample_set._values)

    # did num_samples get updated?
    assert my_num == sampler.num_samples
    
    # did the file get correctly saved?
    if comm.rank == 0:
        saved_disc = bet.sample.load_discretization(savefile)
        
        # compare the samples
        nptest.assert_array_equal(my_discretization._input_sample_set.get_values(),
            saved_disc._input_sample_set.get_values())
        # compare the data
        nptest.assert_array_equal(my_discretization._output_sample_set.get_values(),
           saved_disc._output_sample_set.get_values())
    comm.Barrier()


class Test_basic_sampler(unittest.TestCase):
    """
    Test :class:`bet.sampling.basicSampling.sampler`.
    """

    def setUp(self):
        # create 1-1 map
        self.input_domain1 = np.column_stack((np.zeros((1,)), np.ones((1,))))
        def map_1t1(x):
            return np.sin(x)
        # create 3-1 map
        self.input_domain3 = np.column_stack((np.zeros((3,)), np.ones((3,))))
        def map_3t1(x):
            return np.sum(x, 1)
        # create 3-2 map
        def map_3t2(x):
            return np.vstack(([x[:, 0]+x[:, 1], x[:, 2]])).transpose()
        # create 10-4 map
        self.input_domain10 = np.column_stack((np.zeros((10,)), np.ones((10,))))
        def map_10t4(x):
            x1 = x[:, 0] + x[:, 1]
            x2 = x[:, 2] + x[:, 3]
            x3 = x[:, 4] + x[:, 5]
            x4 = np.sum(x[:, [6, 7, 8, 9]], 1)
            return np.vstack([x1, x2, x3, x4]).transpose()
        num_samples = 100
        self.savefiles = ["11t11", "1t1", "3to1", "3to2", "10to4"]
        self.models = [map_1t1, map_1t1, map_3t1, map_3t2, map_10t4]
        self.samplers = []
        for model in self.models:
            self.samplers.append(bsam.sampler(model, num_samples))

    def tearDown(self):
        """
        Clean up extra files
        """
        if comm.rank == 0:
            for f in self.savefiles:
                if os.path.exists(f+".mat"):
                    os.remove(f+".mat")
        if comm.size > 1:
            for f in self.savefiles:
                proc_savefile = os.path.join(local_path, os.path.dirname(f),
                        "proc{}{}.mat".format(comm.rank, os.path.basename(f)))
                print proc_savefile
                if os.path.exists(proc_savefile):
                    os.remove(proc_savefile)
                proc_savefile = os.path.join(local_path, os.path.dirname(f),
                        "p{}proc{}{}.mat".format(comm.rank, comm.rank,
                            os.path.basename(f)))
                if os.path.exists(proc_savefile):
                    os.remove(proc_savefile)
                print proc_savefile

    def test_init(self):
        """
        Test initalization of :class:`bet.sampling.basicSampling.sampler`
        """
        assert self.samplers[0].num_samples == 100
        assert self.samplers[0].lb_model == self.models[0]
        assert bsam.sampler(self.models[0], None).num_samples is None

    def test_update(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.save`
        """
        mdict = {"frog":3, "moose":2}
        self.samplers[0].update_mdict(mdict)
        assert self.samplers[0].num_samples == mdict["num_samples"]

    def test_user_samples(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.user_samples` for
        three different QoI maps (1 to 1, 3 to 1, 3 to 2, 10 to 4).
        """
        # create a list of different sets of samples
        list_of_samples = [np.ones((4, )), np.ones((4, 1)), np.ones((4, 3)),
                np.ones((4, 3)), np.ones((4, 10))]
        list_of_dims = [1, 1, 3, 3, 10]
        
        list_of_sample_sets = [None]*len(list_of_samples)

        for i, array in enumerate(list_of_samples):
            list_of_sample_sets[i] = sample_set(list_of_dims[i])
            list_of_sample_sets[i].set_values(array)

        test_list = zip(self.models, self.samplers, list_of_sample_sets, 
                self.savefiles)
        
        for model, sampler, input_sample_set, savefile in test_list: 
            for parallel in [False, True]:
                verify_user_samples(model, sampler, input_sample_set, savefile,
                        parallel)
   
    def test_random_samples(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.random_samples` for three
        different QoI maps (1 to 1, 3 to 1, 3 to 2, 10 to 4).
        """
        input_domain_list = [self.input_domain1, self.input_domain1,
                self.input_domain3, self.input_domain3, self.input_domain10]

        test_list = zip(self.models, self.samplers, input_domain_list,
                self.savefiles)

        for model, sampler, input_domain, savefile in test_list:
            for sample_type in ["random", "r", "lhs"]:
                for num_samples in [None, 25]:
                    for parallel in [False, True]:
                        verify_random_samples(model, sampler, sample_type,
                                input_domain, num_samples, savefile,
                                parallel)

