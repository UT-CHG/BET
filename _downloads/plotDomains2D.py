#! /usr/bin/env python

# Copyright (C) 2014-2015 Lindley Graham and Steven Mattis

# import necessary modules
import numpy as np
import bet.postProcess.plotDomains as pDom
import scipy.io as sio

# Set minima and maxima
lam_domain = np.array([[.07, .15], [.1, .2]])
param_min = lam_domain[:, 0]
param_max = lam_domain[:, 1]

# Select only the stations I care about this will lead to better sampling
station_nums = [0, 5] # 1, 6

# Read in Q_ref and Q to create the appropriate rho_D 
mdat = sio.loadmat('Q_2D')
Q = mdat['Q']
Q = Q[:, station_nums]
Q_ref = mdat['Q_true']
Q_ref = Q_ref[15, station_nums] # 16th/20
bin_ratio = 0.15
bin_size = (np.max(Q, 0)-np.min(Q, 0))*bin_ratio

# Create kernel
maximum = 1/np.product(bin_size)
def rho_D(outputs):
    rho_left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
    rho_right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
    rho_left = np.all(np.greater_equal(outputs, rho_left), axis=1)
    rho_right = np.all(np.less_equal(outputs, rho_right), axis=1)
    inside = np.logical_and(rho_left, rho_right)
    max_values = np.repeat(maximum, outputs.shape[0], 0)
    return inside.astype('float64')*max_values

# Read in points_ref and plot results
p_ref = mdat['points_true']
p_ref = p_ref[5:7, 15]

# Show the samples in the parameter space
pDom.show_param(samples=points.transpose(), data=Q, rho_D=rho_D, p_ref=p_ref)
# Show the corresponding samples in the data space
pDom.show_data(data=Q, rho_D=rho_D, Q_ref=Q_ref)
# Show the data domain that corresponds with the convex hull of samples in the
# parameter space
pDom.show_data_domain_2D(samples=points.transpose(), data=Q, Q_ref=Q_ref)

# Show multiple data domains that correspond with the convex hull of samples in
# the parameter space
pDom.show_data_domain_multi(samples=points.transpose(), data=mdat['Q'],
        Q_ref=mdat['Q_true'][15], Q_nums=[1,2,5], showdim='all')
