# Copyright (C) 2016 The BET Development Team

r"""

Evaluate the model at input values.

"""

import os
import scipy.io as sio
import sys
import numpy as np
import numpy.random as rand

"""
Defines an exact and approximate nonlinear map.

"""

def lb_model_exact(input_data):
    """
    Exact map.
    """
    num_runs = input_data.shape[0]
    num_runs_dim = input_data.shape[1]
    
    values = np.zeros((num_runs, 2))
    jacobians = np.zeros((num_runs, 2, 3))
    for i in range(num_runs):
        x = input_data[i, 0]
        y = input_data[i, 1]
        z = input_data[i, 2]
        values[i,0] = 2.0*x*x + y*y +z*z
        values[i,1] = x*y + y + 2.0*z
        jac = np.array([[2.0*x, y, z],
                        [y, 1.0, 2.0]])
        jacobians[i] = jac
    return (values, jacobians)

def lb_model(input_data):
    """

    Approximate map. Adds error to the exact map.

    """
    (values, jacobians) = lb_model_exact(input_data)
    num_runs = input_data.shape[0]
    error_estimates = 0.2 * rand.randn(num_runs, 2)
    values = values - error_estimates
    return(values, error_estimates, jacobians)
