# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-
r"""
The user should set the environment variables LAUNCHER_DIR
and LAUNCHER_PPN below.
"""

import os
import scipy.io as sio
import sys
import numpy as np

def lb_model(input_data):
    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]

    # Setup the job file for Launcher.
    f = open('launcher_runs.txt', 'w')
    for i in range(0, num_runs):
        output_str = sys.executable + ' myModel_serial.py '  + `i` + ' '
        for j in range(0, num_runs_dim):
            output_str = output_str + `input_data[i,j]` + ' '
        output_str += '\n'
        f.write(output_str)
    f.close()
    os.environ["LAUNCHER_JOB_FILE"] = "launcher_runs.txt"

    # USER SETS THESE ENVIRONMENT VARIABLES
    os.environ["LAUNCHER_DIR"] = "DIRECTORY_TO_LAUNCHER_REPO"
    os.environ["LAUNCHER_PPN"] = "NUM_OF_PROCS_TO_USE"

    # Execute Launcher to start multiple serial runs of FEniCS
    os.system("bash /home/troy/Packages/launcher/paramrun")

    # Read in data from files and cleanup files.
    QoI_samples = np.zeros((num_runs, 2))
    for i in range(0, num_runs):
        io_file_name = 'QoI_sample' + `i`
        io_mdat = sio.loadmat(io_file_name)
        QoI_samples[i,:] = io_mdat['output']
        io_file_str = io_file_name + '.mat'
        os.remove(io_file_str)

    return QoI_samples
