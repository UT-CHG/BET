# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-

import numpy as np
import sys
import scipy.io as sio

# Parameter space is nD
# Data space is n/2 D


def my_model(io_file_name):
    # read in input from file
    io_mdat = sio.loadmat(io_file_name)
    input_samples = io_mdat['input']
    # model is y = x[:, 0:dim/2 ] + x[:, dim/2:]
    output_samples = sum(np.split(input_samples, 2, 1))
    # save output to file
    io_mdat['output'] = output_samples
    sio.savemat(io_file_name, io_mdat)


def usage():
    print("usage: [io_file]")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        my_model(sys.argv[1])
    else:
        usage()
