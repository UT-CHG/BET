# Copyright (C) 2014-2019 The BET Development Team

"""
This module contains unittests for :mod:`~bet.sampling.basicSampling:`
"""

import unittest
import os
import pyDOE
import numpy.testing as nptest
import numpy as np
import bet
import bet.sampling.basicSampling as bsam
from bet.Comm import comm
import bet.sample
from bet.sample import sample_set
from bet.sample import discretization as disc
import collections

local_path = os.path.join(".")


@unittest.skipIf(comm.size > 1, 'Only run in serial')
def test_loadmat():
    """
    Tests :meth:`bet.sampling.basicSampling.loadmat`
    """
    np.random.seed(1)
    mdat1 = {'num_samples': 5}
    mdat2 = {'num_samples': 6}
    model = "this is not a model"

    my_input1 = sample_set(1)
    my_input1.set_values(np.random.random((5, 1)))
    my_output = sample_set(1)
    my_output.set_values(np.random.random((5, 1)))
    my_input2 = sample_set(1)
    my_input2.set_values(np.random.random((6, 1)))

    bet.sample.savemat(os.path.join(local_path, 'testfile1'), mdat1)
    bet.sample.savemat(os.path.join(local_path, 'testfile2'), mdat2)

    bet.sample.save_discretization(disc(my_input1, my_output),
                                   (os.path.join(local_path, 'testfile1')), globalize=True)
    bet.sample.save_discretization(disc(my_input2, None),
                                   os.path.join(local_path, 'testfile2'), "NAME", globalize=True)

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


def test_loadmat_parallel():
    """

    Tests :class:`bet.sampling.basicSampling.sampler.loadmat`.

    """
    np.random.seed(1)
    mdat1 = {'num_samples': 10}
    mdat2 = {'num_samples': 20}
    model = "this is not a model"

    my_input1 = sample_set(1)
    my_input1.set_values_local(np.array_split(np.random.random((10, 1)),
                                              comm.size)[comm.rank])
    my_output1 = sample_set(1)
    my_output1.set_values_local(np.array_split(np.random.random((10, 1)),
                                               comm.size)[comm.rank])
    my_input2 = sample_set(1)
    my_input2.set_values_local(np.array_split(np.random.random((20, 1)),
                                              comm.size)[comm.rank])
    my_output2 = sample_set(1)
    my_output2.set_values_local(np.array_split(np.random.random((20, 1)),
                                               comm.size)[comm.rank])

    file_name1 = 'testfile1.mat'
    file_name2 = 'testfile2.mat'

    if comm.size > 1:
        local_file_name1 = os.path.os.path.join(os.path.dirname(file_name1),
                                                "proc{}_{}".format(comm.rank, os.path.basename(file_name1)))
        local_file_name2 = os.path.os.path.join(os.path.dirname(file_name2),
                                                "proc{}_{}".format(comm.rank, os.path.basename(file_name2)))
    else:
        local_file_name1 = file_name1
        local_file_name2 = file_name2

    bet.sample.savemat(local_file_name1, mdat1)
    bet.sample.savemat(local_file_name2, mdat2)
    comm.barrier()

    bet.sample.save_discretization(disc(my_input1, my_output1),
                                   file_name1, globalize=False)
    bet.sample.save_discretization(disc(my_input2, my_output2),
                                   file_name2, "NAME", globalize=False)

    (loaded_sampler1, discretization1) = bsam.loadmat(file_name1)
    nptest.assert_array_equal(discretization1._input_sample_set.get_values(),
                              my_input1.get_values())
    nptest.assert_array_equal(discretization1._output_sample_set.get_values(),
                              my_output1.get_values())
    assert loaded_sampler1.num_samples == 10
    assert loaded_sampler1.lb_model is None

    (loaded_sampler2, discretization2) = bsam.loadmat(file_name2,
                                                      disc_name="NAME", model=model)
    nptest.assert_array_equal(discretization2._input_sample_set.get_values(),
                              my_input2.get_values())
    nptest.assert_array_equal(discretization2._output_sample_set.get_values(),
                              my_output2.get_values())

    assert loaded_sampler2.num_samples == 20
    assert loaded_sampler2.lb_model == model
    if comm.size == 1:
        os.remove(file_name1)
        os.remove(file_name2)
    else:
        os.remove(local_file_name1)
        os.remove(local_file_name2)


def verify_compute_QoI_and_create_discretization(model, sampler,
                                                 input_sample_set,
                                                 savefile):
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

    # evaluate the model at the sample
    print(savefile, input_sample_set.get_dim())
    my_discretization = sampler.compute_QoI_and_create_discretization(
        input_sample_set, savefile, globalize=True)
    # comm.barrier()

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
    saved_disc = bet.sample.load_discretization(savefile)
    mdat = bet.sample.loadmat(savefile)
    print("HERE HERE", mdat, my_num)
    # comm.barrier()
    # compare the samples
    nptest.assert_array_equal(my_discretization._input_sample_set.get_values(),
                              saved_disc._input_sample_set.get_values())
    # compare the data
    nptest.assert_array_equal(my_discretization._output_sample_set.get_values(),
                              saved_disc._output_sample_set.get_values())


def verify_add_qoi(model, sampler,
                   input_sample_set,
                   savefile):
    """
    Verify that the user samples are correct.
    """
    # evalulate the model at the samples directly
    output_values = (np.column_stack([model(input_sample_set._values),
                                      model(input_sample_set._values)]))

    if len(output_values.shape) == 1:
        output_sample_set = sample_set(1)
    else:
        output_sample_set = sample_set(output_values.shape[1])
    output_sample_set.set_values(output_values)
    discretization = disc(input_sample_set, output_sample_set)

    # evaluate the model at the sample
    print(savefile, input_sample_set.get_dim())
    my_discretization = sampler.compute_QoI_and_create_discretization(
        input_sample_set, savefile, globalize=True)
    # check add_qoi
    my_discretization = sampler.add_qoi(my_discretization,
                                        savefile=savefile, globalize=True)
    # comm.barrier()

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
    saved_disc = bet.sample.load_discretization(savefile)
    mdat = bet.sample.loadmat(savefile)
    print("HERE HERE", mdat, my_num)
    # comm.barrier()
    # compare the samples
    nptest.assert_array_equal(my_discretization.
                              _input_sample_set.get_values(),
                              saved_disc._input_sample_set.get_values())
    # compare the data
    nptest.assert_array_equal(my_discretization.
                              _output_sample_set.get_values(),
                              saved_disc._output_sample_set.get_values())


def verify_add_qoi_with_data(model, sampler,
                             input_sample_set,
                             savefile):
    """
    Verify that the adding data and QoI is done correctly.
    """
    # evalulate the model at the samples directly
    output_values = (np.column_stack([model(input_sample_set._values),
                                      model(input_sample_set._values)]))

    if len(output_values.shape) == 1:
        output_sample_set = sample_set(1)
        prob_set = sample_set(1)
    else:
        output_sample_set = sample_set(output_values.shape[1])
        prob_set = sample_set(output_values.shape[1] // 2)

    output_sample_set.set_values(output_values)
    discretization = disc(input_sample_set, output_sample_set)

    # evaluate the model at the sample
    print(savefile, input_sample_set.get_dim())
    my_discretization = sampler.compute_QoI_and_create_discretization(
        input_sample_set, savefile, globalize=True)

    # set output_probability_set to not be empty (but with data)
    my_discretization.set_output_probability_set(prob_set)
    data_orig = np.random.rand(prob_set.get_dim())
    my_discretization.set_data(data_orig)
    data = np.random.rand(prob_set.get_dim())
    # check add_qoi
    my_discretization = sampler.add_qoi(my_discretization, data=data,
                                        savefile=savefile, globalize=True)
    # comm.barrier()

    my_num = my_discretization.check_nums()

    # compare the samples
    MD = my_discretization
    D = discretization
    nptest.assert_array_equal(MD.get_input().get_values(),
                              D.get_input().get_values())

    # compare the data
    nptest.assert_array_equal(MD.get_output().get_values(),
                              D.get_output().get_values())

    nptest.assert_array_equal(MD.get_data(),
                              np.concatenate((data_orig, data)))

    # did num_samples get updated?
    assert my_num == sampler.num_samples

    # did the file get correctly saved?
    saved_disc = bet.sample.load_discretization(savefile)
    mdat = bet.sample.loadmat(savefile)
    print("HERE HERE", mdat, my_num)
    # comm.barrier()
    # compare the samples
    nptest.assert_array_equal(my_discretization.
                              _input_sample_set.get_values(),
                              saved_disc._input_sample_set.get_values())
    # compare the data
    nptest.assert_array_equal(my_discretization.
                              _output_sample_set.get_values(),
                              saved_disc._output_sample_set.get_values())


def verify_create_random_discretization(model, sampler,
                                        sample_type, input_domain,
                                        num_samples, savefile):

    np.random.seed(1)
    # recreate the samples
    if num_samples is None:
        num_samples = sampler.num_samples

    input_sample_set = sample_set(input_domain.shape[0])
    input_sample_set.set_domain(input_domain)

    input_left = np.repeat([input_domain[:, 0]], num_samples, 0)
    input_right = np.repeat([input_domain[:, 1]], num_samples, 0)

    input_values = (input_right - input_left)
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

    # reset the random seed
    np.random.seed(1)
    comm.barrier()
    # create the random discretization using a specified input domain
    my_discretization = sampler.create_random_discretization(sample_type,
                                                             input_domain, savefile, num_samples=num_samples, globalize=True)
    # comm.barrier()
    my_num = my_discretization.check_nums()

    # make sure that the samples are within the boundaries
    assert np.all(my_discretization._input_sample_set._values <= input_right)
    assert np.all(my_discretization._input_sample_set._values >= input_left)

    if comm.size == 0:
        # compare the samples
        nptest.assert_array_equal(input_sample_set._values,
                                  my_discretization._input_sample_set._values)
        # compare the data
        nptest.assert_array_equal(output_sample_set._values,
                                  my_discretization._output_sample_set._values)

    # did num_samples get updated?
    assert my_num == sampler.num_samples

    # did the file get correctly saved?
    saved_disc = bet.sample.load_discretization(savefile)

    # compare the samples
    nptest.assert_array_equal(my_discretization._input_sample_set.get_values(),
                              saved_disc._input_sample_set.get_values())
    # compare the data
    nptest.assert_array_equal(my_discretization._output_sample_set.get_values(),
                              saved_disc._output_sample_set.get_values())

    # reset the random seed
    np.random.seed(1)

    my_sample_set = sample_set(input_domain.shape[0])
    my_sample_set.set_domain(input_domain)
    # comm.barrier()
    # create the random discretization using an initialized sample_set
    my_discretization = sampler.create_random_discretization(sample_type,
                                                             my_sample_set, savefile, num_samples=num_samples,
                                                             globalize=True)
    my_num = my_discretization.check_nums()

    # make sure that the samples are within the boundaries
    assert np.all(my_discretization._input_sample_set._values <= input_right)
    assert np.all(my_discretization._input_sample_set._values >= input_left)

    if comm.size == 0:
        # compare the samples
        nptest.assert_array_equal(input_sample_set._values,
                                  my_discretization._input_sample_set._values)
        # compare the data
        nptest.assert_array_equal(output_sample_set._values,
                                  my_discretization._output_sample_set._values)

    # reset the random seed
    np.random.seed(1)
    # recreate the samples to test default choices with unit hypercube domain
    if num_samples is None:
        num_samples = sampler.num_samples

    my_dim = input_domain.shape[0]
    input_sample_set = sample_set(my_dim)
    input_sample_set.set_domain(np.repeat([[0.0, 1.0]], my_dim, axis=0))

    input_left = np.repeat([input_domain[:, 0]], num_samples, 0)
    input_right = np.repeat([input_domain[:, 1]], num_samples, 0)

    input_values = (input_right - input_left)
    if sample_type == "lhs":
        input_values = input_values * pyDOE.lhs(input_sample_set.get_dim(),
                                                num_samples, 'center')
    elif sample_type == "random" or "r":
        input_values = input_values * np.random.random(input_left.shape)
    input_values = input_values + input_left
    input_sample_set.set_values(input_values)

    # reset random seed
    np.random.seed(1)
    comm.barrier()
    # create the random discretization using a specified input_dim
    my_discretization = sampler.create_random_discretization(sample_type,
                                                             my_dim, savefile, num_samples=num_samples, globalize=True)
    # comm.barrier()
    my_num = my_discretization.check_nums()

    # make sure that the samples are within the boundaries
    assert np.all(my_discretization._input_sample_set._values <= input_right)
    assert np.all(my_discretization._input_sample_set._values >= input_left)

    if comm.size == 0:
        # compare the samples
        nptest.assert_array_equal(input_sample_set._values,
                                  my_discretization._input_sample_set._values)
        # compare the data
        nptest.assert_array_equal(output_sample_set._values,
                                  my_discretization._output_sample_set._values)


def verify_random_sample_set_domain(sampler, sample_type, input_domain,
                                    num_samples):
    np.random.seed(1)
    # recreate the samples
    if num_samples is None:
        num_samples = sampler.num_samples

    input_sample_set = sample_set(input_domain.shape[0])
    input_sample_set.set_domain(input_domain)

    input_left = np.repeat([input_domain[:, 0]], num_samples, 0)
    input_right = np.repeat([input_domain[:, 1]], num_samples, 0)

    input_values = (input_right - input_left)
    if sample_type == "lhs":
        input_values = input_values * pyDOE.lhs(input_sample_set.get_dim(),
                                                num_samples, 'center')
    elif sample_type == "random" or "r":
        input_values = input_values * np.random.random(input_left.shape)
    input_values = input_values + input_left
    input_sample_set.set_values(input_values)

    # reset the random seed
    np.random.seed(1)

    # create the sample set from the domain
    print(sample_type)
    my_sample_set = sampler.random_sample_set(sample_type, input_domain,
                                              num_samples=num_samples)

    # make sure that the samples are within the boundaries
    assert np.all(my_sample_set._values <= input_right)
    assert np.all(my_sample_set._values >= input_left)

    # compare the samples
    if comm.size == 0:
        nptest.assert_array_equal(input_sample_set._values,
                                  my_sample_set._values)


def verify_random_sample_set_dimension(sampler, sample_type, input_dim,
                                       num_samples):

    np.random.seed(1)
    # recreate the samples
    if num_samples is None:
        num_samples = sampler.num_samples

    input_domain = np.repeat([[0, 1]], input_dim, axis=0)
    input_sample_set = sample_set(input_dim)
    input_sample_set.set_domain(input_domain)

    input_left = np.repeat([input_domain[:, 0]], num_samples, 0)
    input_right = np.repeat([input_domain[:, 1]], num_samples, 0)

    input_values = (input_right - input_left)
    if sample_type == "lhs":
        input_values = input_values * pyDOE.lhs(input_sample_set.get_dim(),
                                                num_samples, 'center')
    elif sample_type == "random" or "r":
        input_values = input_values * np.random.random(input_left.shape)
    input_values = input_values + input_left
    input_sample_set.set_values(input_values)

    # reset the random seed
    np.random.seed(1)

    # create the sample set from the domain
    my_sample_set = sampler.random_sample_set(sample_type, input_dim,
                                              num_samples=num_samples)

    # make sure that the samples are within the boundaries
    assert np.all(my_sample_set._values <= input_right)
    assert np.all(my_sample_set._values >= input_left)

    # compare the samples
    if comm.size == 0:
        nptest.assert_array_equal(input_sample_set._values,
                                  my_sample_set._values)


def verify_random_sample_set(sampler, sample_type, input_sample_set,
                             num_samples):
    test_sample_set = input_sample_set
    np.random.seed(1)
    # recreate the samples
    if num_samples is None:
        num_samples = sampler.num_samples

    input_domain = input_sample_set.get_domain()
    if input_domain is None:
        input_domain = np.repeat([[0, 1]], input_sample_set.get_dim(), axis=0)

    input_left = np.repeat([input_domain[:, 0]], num_samples, 0)
    input_right = np.repeat([input_domain[:, 1]], num_samples, 0)

    input_values = (input_right - input_left)
    if sample_type == "lhs":
        input_values = input_values * pyDOE.lhs(input_sample_set.get_dim(),
                                                num_samples, 'center')
    elif sample_type == "random" or "r":
        input_values = input_values * np.random.random(input_left.shape)
    input_values = input_values + input_left
    test_sample_set.set_values(input_values)

    # reset the random seed
    np.random.seed(1)

    # create the sample set from the domain
    print(sample_type)
    my_sample_set = sampler.random_sample_set(sample_type, input_sample_set,
                                              num_samples=num_samples)

    # make sure that the samples are within the boundaries
    assert np.all(my_sample_set._values <= input_right)
    assert np.all(my_sample_set._values >= input_left)

    # compare the samples
    if comm.size == 0:
        nptest.assert_array_equal(test_sample_set._values,
                                  my_sample_set._values)


def verify_regular_sample_set(sampler, input_sample_set,
                              num_samples_per_dim):

    test_sample_set = input_sample_set
    dim = input_sample_set.get_dim()
    # recreate the samples
    if num_samples_per_dim is None:
        num_samples_per_dim = 5

    if not isinstance(num_samples_per_dim, collections.Iterable):
        num_samples_per_dim = num_samples_per_dim * \
            np.ones((dim,), dtype='int')

    sampler.num_samples = np.product(num_samples_per_dim)

    test_domain = test_sample_set.get_domain()
    if test_domain is None:
        test_domain = np.repeat([[0, 1]], test_sample_set.get_dim(), axis=0)

    test_values = np.zeros((sampler.num_samples, test_sample_set.get_dim()))

    vec_samples_dimension = np.empty((dim), dtype=object)
    for i in np.arange(0, dim):
        bin_width = (test_domain[i, 1] - test_domain[i, 0]) / \
            np.float(num_samples_per_dim[i])
        vec_samples_dimension[i] = list(np.linspace(
            test_domain[i, 0] - 0.5 * bin_width,
            test_domain[i, 1] + 0.5 * bin_width,
            num_samples_per_dim[i] + 2))[1:num_samples_per_dim[i] + 1]

    arrays_samples_dimension = np.meshgrid(
        *[vec_samples_dimension[i] for i in np.arange(0, dim)], indexing='ij')

    for i in np.arange(0, dim):
        test_values[:, i:i +
                    1] = np.vstack(arrays_samples_dimension[i].flat[:])

    test_sample_set.set_values(test_values)

    # create the sample set from sampler
    my_sample_set = sampler.regular_sample_set(input_sample_set,
                                               num_samples_per_dim=num_samples_per_dim)

    # compare the samples
    nptest.assert_array_equal(test_sample_set._values,
                              my_sample_set._values)


def verify_regular_sample_set_domain(sampler, input_domain,
                                     num_samples_per_dim):

    input_sample_set = sample_set(input_domain.shape[0])
    input_sample_set.set_domain(input_domain)

    test_sample_set = input_sample_set
    dim = input_sample_set.get_dim()
    # recreate the samples
    if num_samples_per_dim is None:
        num_samples_per_dim = 5

    if not isinstance(num_samples_per_dim, collections.Iterable):
        num_samples_per_dim = num_samples_per_dim * \
            np.ones((dim,), dtype='int')

    sampler.num_samples = np.product(num_samples_per_dim)

    test_domain = test_sample_set.get_domain()
    if test_domain is None:
        test_domain = np.repeat([[0, 1]], test_sample_set.get_dim(), axis=0)

    test_values = np.zeros((sampler.num_samples, test_sample_set.get_dim()))

    vec_samples_dimension = np.empty((dim), dtype=object)
    for i in np.arange(0, dim):
        bin_width = (test_domain[i, 1] - test_domain[i, 0]) / \
            np.float(num_samples_per_dim[i])
        vec_samples_dimension[i] = list(np.linspace(
            test_domain[i, 0] - 0.5 * bin_width,
            test_domain[i, 1] + 0.5 * bin_width,
            num_samples_per_dim[i] + 2))[1:num_samples_per_dim[i] + 1]

    arrays_samples_dimension = np.meshgrid(
        *[vec_samples_dimension[i] for i in np.arange(0, dim)], indexing='ij')

    for i in np.arange(0, dim):
        test_values[:, i:i +
                    1] = np.vstack(arrays_samples_dimension[i].flat[:])

    test_sample_set.set_values(test_values)

    # create the sample set from sampler
    my_sample_set = sampler.regular_sample_set(input_domain,
                                               num_samples_per_dim=num_samples_per_dim)

    # compare the samples
    nptest.assert_array_equal(test_sample_set._values,
                              my_sample_set._values)


def verify_regular_sample_set_dimension(sampler, input_dim,
                                        num_samples_per_dim):

    input_domain = np.repeat([[0, 1]], input_dim, axis=0)
    input_sample_set = sample_set(input_dim)
    input_sample_set.set_domain(input_domain)

    test_sample_set = input_sample_set
    dim = input_dim
    # recreate the samples
    if num_samples_per_dim is None:
        num_samples_per_dim = 5

    if not isinstance(num_samples_per_dim, collections.Iterable):
        num_samples_per_dim = num_samples_per_dim * \
            np.ones((dim,), dtype='int')

    sampler.num_samples = np.product(num_samples_per_dim)

    test_domain = test_sample_set.get_domain()
    if test_domain is None:
        test_domain = np.repeat([[0, 1]], test_sample_set.get_dim(), axis=0)

    test_values = np.zeros((sampler.num_samples, test_sample_set.get_dim()))

    vec_samples_dimension = np.empty((dim), dtype=object)
    for i in np.arange(0, dim):
        bin_width = (test_domain[i, 1] - test_domain[i, 0]) / \
            np.float(num_samples_per_dim[i])
        vec_samples_dimension[i] = list(np.linspace(
            test_domain[i, 0] - 0.5 * bin_width,
            test_domain[i, 1] + 0.5 * bin_width,
            num_samples_per_dim[i] + 2))[1:num_samples_per_dim[i] + 1]

    arrays_samples_dimension = np.meshgrid(
        *[vec_samples_dimension[i] for i in np.arange(0, dim)], indexing='ij')

    for i in np.arange(0, dim):
        test_values[:, i:i +
                    1] = np.vstack(arrays_samples_dimension[i].flat[:])

    test_sample_set.set_values(test_values)

    # create the sample set from sampler
    my_sample_set = sampler.regular_sample_set(input_dim,
                                               num_samples_per_dim=num_samples_per_dim)

    # compare the samples
    nptest.assert_array_equal(test_sample_set._values,
                              my_sample_set._values)


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
            return np.vstack(([x[:, 0] + x[:, 1], x[:, 2]])).transpose()
        # create 10-4 map
        self.input_domain10 = np.column_stack(
            (np.zeros((10,)), np.ones((10,))))

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

        self.input_dim1 = 1
        self.input_dim2 = 3  # can be set to other dimensions.
        self.input_dim3 = 10

        self.input_sample_set1 = sample_set(self.input_dim1)
        self.input_sample_set2 = sample_set(self.input_dim2)
        self.input_sample_set3 = sample_set(self.input_dim3)

        self.input_sample_set4 = sample_set(self.input_domain1.shape[0])
        self.input_sample_set4.set_domain(self.input_domain1)

        self.input_sample_set5 = sample_set(self.input_domain3.shape[0])
        self.input_sample_set5.set_domain(self.input_domain3)

        self.input_sample_set6 = sample_set(self.input_domain10.shape[0])
        self.input_sample_set6.set_domain(self.input_domain10)

    def tearDown(self):
        """
        Clean up extra files
        """
        comm.barrier()
        if comm.rank == 0:
            for f in self.savefiles:
                if os.path.exists(f + ".mat"):
                    os.remove(f + ".mat")
        comm.barrier()

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
        mdict = {"frog": 3, "moose": 2}
        self.samplers[0].update_mdict(mdict)
        assert self.samplers[0].num_samples == mdict["num_samples"]

    def test_compute_QoI_and_create_discretization(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.user_samples`
        for three different QoI maps (1 to 1, 3 to 1, 3 to 2, 10 to 4).
        """
        # create a list of different sets of samples
        list_of_samples = [np.ones((4, )), np.ones((4, 1)), np.ones((4, 3)),
                           np.ones((4, 3)), np.ones((4, 10))]
        list_of_dims = [1, 1, 3, 3, 10]

        list_of_sample_sets = [None] * len(list_of_samples)

        for i, array in enumerate(list_of_samples):
            list_of_sample_sets[i] = sample_set(list_of_dims[i])
            list_of_sample_sets[i].set_values(array)

        test_list = list(zip(self.models, self.samplers, list_of_sample_sets,
                             self.savefiles))

        for model, sampler, input_sample_set, savefile in test_list:
            verify_compute_QoI_and_create_discretization(model, sampler,
                                                         input_sample_set,
                                                         savefile)

    def test_random_sample_set(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.random_sample_set`
        for six different sample sets
        """
        input_sample_set_list = [self.input_sample_set1,
                                 self.input_sample_set2,
                                 self.input_sample_set3,
                                 self.input_sample_set4,
                                 self.input_sample_set5,
                                 self.input_sample_set6]

        test_list = list(zip(self.samplers, input_sample_set_list))

        for sampler, input_sample_set in test_list:
            for sample_type in ["random", "r", "lhs"]:
                for num_samples in [None, 25]:
                    verify_random_sample_set(sampler, sample_type,
                                             input_sample_set, num_samples)

    def test_random_sample_set_domain(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.random_sample_set`
        for five different input domains.
        """
        input_domain_list = [self.input_domain1, self.input_domain1,
                             self.input_domain3, self.input_domain3,
                             self.input_domain10]

        test_list = list(zip(self.samplers, input_domain_list))

        for sampler, input_domain in test_list:
            for sample_type in ["random", "r", "lhs"]:
                for num_samples in [None, 25]:
                    verify_random_sample_set_domain(sampler, sample_type,
                                                    input_domain, num_samples)

    def test_random_sample_set_dim(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.random_sample_set_dim`
        for three different input dimensions.
        """
        input_dim_list = [self.input_dim1, self.input_dim2, self.input_dim3]

        test_list = list(zip(self.samplers, input_dim_list))

        for sampler, input_dim in test_list:
            for sample_type in ["random", "r", "lhs"]:
                for num_samples in [None, 25]:
                    verify_random_sample_set_dimension(sampler, sample_type,
                                                       input_dim, num_samples)

    def test_regular_sample_set(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.regular_sample_set`
        for six different sample sets
        """
        input_sample_set_list = [self.input_sample_set1,
                                 self.input_sample_set2,
                                 self.input_sample_set4,
                                 self.input_sample_set5]

        test_list = list(zip(self.samplers, input_sample_set_list))

        for sampler, input_sample_set in test_list:
            for num_samples_per_dim in [None, 10]:
                verify_regular_sample_set(
                    sampler, input_sample_set, num_samples_per_dim)

    def test_regular_sample_set_domain(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.regular_sample_set_domain`
        for six different sample sets
        """
        input_domain_list = [self.input_domain1,
                             self.input_domain3]

        test_list = list(zip(self.samplers, input_domain_list))

        for sampler, input_domain in test_list:
            for num_samples_per_dim in [None, 10]:
                verify_regular_sample_set_domain(
                    sampler, input_domain, num_samples_per_dim)

    def test_regular_sample_set_dimension(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.regular_sample_set_dimension`
        for six different sample sets
        """
        input_dimension_list = [self.input_dim1,
                                self.input_dim2]

        test_list = list(zip(self.samplers, input_dimension_list))

        for sampler, input_dim in test_list:
            for num_samples_per_dim in [None, 10]:
                verify_regular_sample_set_dimension(
                    sampler, input_dim, num_samples_per_dim)

    def test_create_random_discretization(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.create_random_discretization`
        for three different QoI maps (1 to 1, 3 to 1, 3 to 2, 10 to 4).
        """
        input_domain_list = [self.input_domain1, self.input_domain1,
                             self.input_domain3, self.input_domain3,
                             self.input_domain10]

        test_list = list(zip(self.models, self.samplers, input_domain_list,
                             self.savefiles))

        for model, sampler, input_domain, savefile in test_list:
            for sample_type in ["random", "r", "lhs"]:
                for num_samples in [None, 25]:
                    verify_create_random_discretization(model, sampler,
                                                        sample_type,
                                                        input_domain,
                                                        num_samples,
                                                        savefile)


class Test_basic_sampler_extended(Test_basic_sampler):
    """
    Test model and output appending and reference
    values for :class:`bet.sampling.basicSampling.sampler`.
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
            try:
                return np.vstack(([x[:, 0] + x[:, 1], x[:, 2]])).transpose()
            except IndexError:  # support reference-parameter mapping
                return np.vstack(([x[0] + x[1], x[2]])).transpose()
        # create 10-4 map
        self.input_domain10 = np.column_stack(
            (np.zeros((10,)), np.ones((10,))))

        def map_10t4(x):
            if len(x.shape) == 1:
                x = np.array([x])  # support for reference parameter
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

        self.input_dim1 = 1
        self.input_dim2 = 3
        self.input_dim3 = 10

        self.input_sample_set1 = sample_set(self.input_dim1)
        ref_val1 = np.random.rand(self.input_dim1)
        self.input_sample_set1.set_reference_value(ref_val1)
        self.input_sample_set2 = sample_set(self.input_dim2)
        ref_val2 = np.random.rand(self.input_dim2)
        self.input_sample_set2.set_reference_value(ref_val2)
        self.input_sample_set3 = sample_set(self.input_dim3)
        ref_val3 = np.random.rand(self.input_dim3)
        self.input_sample_set3.set_reference_value(ref_val3)

        self.input_sample_set4 = sample_set(self.input_domain1.shape[0])
        self.input_sample_set4.set_reference_value(ref_val1)
        self.input_sample_set4.set_domain(self.input_domain1)

        self.input_sample_set5 = sample_set(self.input_domain3.shape[0])
        self.input_sample_set5.set_reference_value(ref_val2)
        self.input_sample_set5.set_domain(self.input_domain3)

        self.input_sample_set6 = sample_set(self.input_domain10.shape[0])
        self.input_sample_set6.set_reference_value(ref_val3)
        self.input_sample_set6.set_domain(self.input_domain10)

    def test_add_qoi(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.add_qoi`
        for three different QoI maps (1 to 1, 3 to 1, 3 to 2, 10 to 4).
        """
        # create a list of different sets of samples
        list_of_samples = [np.ones((4, )), np.ones((4, 1)), np.ones((4, 3)),
                           np.ones((4, 3)), np.ones((4, 10))]
        list_of_dims = [1, 1, 3, 3, 10]

        list_of_sample_sets = [None] * len(list_of_samples)

        for i, array in enumerate(list_of_samples):
            dim = list_of_dims[i]
            list_of_sample_sets[i] = sample_set(dim)
            list_of_sample_sets[i].set_values(array)
            list_of_sample_sets[i].set_reference_value(np.random.rand(dim))

        test_list = list(zip(self.models, self.samplers, list_of_sample_sets,
                             self.savefiles))

        for model, sampler, input_sample_set, savefile in test_list:
            verify_add_qoi(model, sampler, input_sample_set, savefile)

    def test_add_qoi_with_data(self):
        """
        Test :meth:`bet.sampling.basicSampling.sampler.add_qoi`
        for three different QoI maps (1 to 1, 3 to 1, 3 to 2, 10 to 4).
        """
        # create a list of different sets of samples
        list_of_samples = [np.ones((4, )), np.ones((4, 1)), np.ones((4, 3)),
                           np.ones((4, 3)), np.ones((4, 10))]
        list_of_dims = [1, 1, 3, 3, 10]

        list_of_sample_sets = [None] * len(list_of_samples)

        for i, array in enumerate(list_of_samples):
            dim = list_of_dims[i]
            list_of_sample_sets[i] = sample_set(dim)
            list_of_sample_sets[i].set_values(array)
            list_of_sample_sets[i].set_reference_value(np.random.rand(dim))

        test_list = list(zip(self.models, self.samplers, list_of_sample_sets,
                             self.savefiles))

        for model, sampler, input_sample_set, savefile in test_list:
            verify_add_qoi_with_data(
                model, sampler, input_sample_set, savefile)
