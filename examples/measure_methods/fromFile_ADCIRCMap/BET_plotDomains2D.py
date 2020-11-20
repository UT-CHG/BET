#! /usr/bin/env python

# Copyright (C) 2014-2020 The BET Development Team

# import necessary modules
import numpy as np
import bet.postProcess.plotDomains as pDom
import scipy.io as sio
import bet.sample as sample

# Set minima and maxima
lam_domain = np.array([[.07, .15], [.1, .2]])

# Select only the stations I care about this will lead to better sampling
station_nums = [0, 5]  # 1, 6

# Read in Q_ref and Q to create the appropriate rho_D
mdat = sio.loadmat('Q_2D.mat')
Q = mdat['Q']
Q = Q[:, station_nums]
Q_ref = mdat['Q_true']
Q_ref = Q_ref[15, station_nums]  # 16th/20
bin_ratio = 0.15
bin_size = (np.max(Q, 0) - np.min(Q, 0)) * bin_ratio

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


# Read in points_ref and plot results
ref_sample = mdat['points_true']
ref_sample = ref_sample[5:7, 15]

# Create input, output, and discretization from data read from file
points = mdat['points']
input_sample_set = sample.sample_set(points.shape[0])
input_sample_set.set_values(points.transpose())
input_sample_set.set_domain(lam_domain)
output_sample_set = sample.sample_set(Q.shape[1])
output_sample_set.set_values(Q)
my_disc = sample.discretization(input_sample_set, output_sample_set)

# Show the samples in the parameter space
pDom.scatter_rhoD(my_disc, rho_D=rho_D, ref_sample=ref_sample, io_flag='input')
# Show the corresponding samples in the data space
pDom.scatter_rhoD(output_sample_set, rho_D=rho_D, ref_sample=Q_ref,
                  io_flag='output')
# Show the data domain that corresponds with the convex hull of samples in the
# parameter space
pDom.show_data_domain_2D(my_disc, Q_ref=Q_ref)

# Show multiple data domains that correspond with the convex hull of samples in
# the parameter space
pDom.show_data_domain_multi(my_disc, Q_ref=mdat['Q_true'][15],
                            showdim='all')
