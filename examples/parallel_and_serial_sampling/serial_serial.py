# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-

# This demonstrates how to use BET in serial to sample a serial external model.
# run by calling "python serial_serial.py"

import os
import subprocess
import scipy.io as sio
import bet.sampling.basicSampling as bsam


def lb_model(input_data):
    io_file_name = "io_file"
    io_mdat = dict()
    io_mdat['input'] = input_data

    # save the input to file
    sio.savemat(io_file_name, io_mdat)

    # run the model
    subprocess.call(['python', 'serial_model.py', io_file_name])

    # read the output from file
    io_mdat = sio.loadmat(io_file_name)
    output_data = io_mdat['output']
    return output_data


my_sampler = bsam.sampler(lb_model)
my_discretization = my_sampler.create_random_discretization(sample_type='r',
                                                            input_obj=4, savefile="serial_serial_example", num_samples=100)
