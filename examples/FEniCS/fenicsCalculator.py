# Example from Fenics Project Tutorial: 
# https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1003.html

import numpy as np
import scipy as sci
from myModel import my_model

# loads parameter samples & saves output quantities of interest
in_samples = np.load("parameter_sample_values.npy")

out_samples = my_model(in_samples)

np.save("QoI_outsample_values",out_samples)
