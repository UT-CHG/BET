# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-

import os
import scipy.io as sio
import bet.sampling.basicSampling as bsam
from bet.Comm import comm
import sys
import subprocess
import numpy as np

def lb_model(input_data):
    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]

    # f = open('launcher_runs.txt', 'w')
    # for i in range(0, num_runs):
    #     output_str = sys.executable + ` i ` + ' myModel_serial.py '
    #     for j in range(0, num_runs_dim):
    #         output_str = output_str + `input_data[i,j]` + ' '
    #     output_str += '\n'
    #     f.write(output_str)
    # f.close()

    os.environ["LAUNCHER_DIR"] = "/home/troy/Packages/launcher"
    os.environ["LAUNCHER_JOB_FILE"] = "launcher_runs.txt"
    os.environ["LAUNCHER_PPN"] = "2"
    #os.environ["LAUNCHER_SCHED"] = "interleaved"

    os.system("bash paramrun")
    
    return np.zeros((400,1))
