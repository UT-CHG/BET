#! /usr/bin/env python

# Copyright (C) 2014-2019 The BET Development Team

# import necessary modules
import numpy as np
import bet.sampling.adaptiveSampling as asam
import bet.sampling.basicSampling as bsam
import bet.postProcess.postTools as ptools
import scipy.io as sio
from scipy.interpolate import griddata

sample_save_file = 'sandbox2d'

# Set minima and maxima
lam_domain = np.array([[.07, .15], [.1, .2]])
lam3 = 0.012
ymin = -1050
xmin = 1420
xmax = 1580
ymax = 1500
wall_height = -2.5


# Select only the stations I care about this will lead to better sampling
station_nums = [0, 5]  # 1, 6


# Create Transition Kernel
transition_set = asam.transition_set(.5, .5**5, 1.0)

# Read in Q_ref and Q to create the appropriate rho_D
mdat = sio.loadmat('../matfiles/Q_2D')
Q = mdat['Q']
Q = Q[:, station_nums]
Q_ref = mdat['Q_true']
Q_ref = Q_ref[15, station_nums]  # 16th/20
bin_ratio = 0.15
bin_size = (np.max(Q, 0) - np.min(Q, 0)) * bin_ratio

# Create experiment model
points = mdat['points']


def model(inputs):
    interp_values = np.empty((inputs.shape[0], Q.shape[1]))
    for i in range(Q.shape[1]):
        interp_values[:, i] = griddata(points.transpose(), Q[:, i],
                                       inputs)
    return interp_values


# Create kernel
maximum = 1 / np.product(bin_size)


def rho_D(outputs):
    rho_left = np.repeat([Q_ref - .5 * bin_size], outputs.shape[0], 0)
    rho_right = np.repeat([Q_ref + .5 * bin_size], outputs.shape[0], 0)
    rho_left = np.all(np.greater_equal(outputs, rho_left), axis=1)
    rho_right = np.all(np.less_equal(outputs, rho_right), axis=1)
    inside = np.logical_and(rho_left, rho_right)
    max_values = np.repeat(maximum, outputs.shape[0], 0)
    return inside.astype('float64') * max_values


kernel_mm = asam.maxima_mean_kernel(np.array([Q_ref]), rho_D)
kernel_rD = asam.rhoD_kernel(maximum, rho_D)
kernel_m = asam.maxima_kernel(np.array([Q_ref]), rho_D)
kern_list = [kernel_mm, kernel_rD, kernel_m]

# Create sampler
chain_length = 125
num_chains = 80
num_samples = chain_length * num_chains
sampler = asam.sampler(num_samples, chain_length, model)
inital_sample_type = "lhs"

# Get samples
# Run with varying kernels
gen_results = sampler.run_gen(kern_list, rho_D, maximum, lam_domain,
                              transition_set, sample_save_file)
# run_reseed_results = sampler.run_gen(kern_list, rho_D, maximum, lam_domain,
#       t_kernel, sample_save_file, reseed=3)

# Run with varying transition sets bounds
init_ratio = [0.1, 0.25, 0.5]
min_ratio = [2e-3, 2e-5, 2e-8]
max_ratio = [.5, .75, 1.0]
tk_results = sampler.run_tk(init_ratio, min_ratio, max_ratio, rho_D,
                            maximum, lam_domain, kernel_rD, sample_save_file)

# Run with varying increase/decrease ratios and tolerances for a rhoD_kernel
increase = [1.0, 2.0, 4.0]
decrease = [0.5, 0.5e2, 0.5e3]
tolerance = [1e-4, 1e-6, 1e-8]
incdec_results = sampler.run_inc_dec(increase, decrease, tolerance, rho_D,
                                     maximum, lam_domain, transition_set, sample_save_file)

# Compare the quality of several sets of samples
print("Compare yield of sample sets with various kernels")
ptools.compare_yield(gen_results[3], gen_results[2], gen_results[4])
print("Compare yield of sample sets with various transition sets bounds")
ptools.compare_yield(tk_results[3], tk_results[2], tk_results[4])
print("Compare yield of sample sets with variouos increase/decrease ratios")
ptools.compare_yield(incdec_results[3], incdec_results[2], incdec_results[4])

# Read in points_ref and plot results
p_ref = mdat['points_true']
p_ref = p_ref[5:7, 15]
