#! /usr/bin/env python
# import necessary modules
import numpy as np
import polysim.pyADCIRC.basic as basic
import bet.sampling.adaptiveSampling as aps
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import math

sample_save_file = 'sandbox2d'

# Set minima and maxima
lam_domain = np.array([[.07, .15], [.1, .2]])
lam3 = 0.012
ymin = -1050
xmin = 1420
xmax = 1580
ymax = 1500
wall_height = -2.5

param_min = lam_domain[:, 0]
param_max = lam_domain[:, 1]

# Create stations
stat_x = np.concatenate((1900*np.ones((7,)), [1200], 1300*np.ones((3,)),
    [1500])) 
stat_y = np.array([1200, 600, 300, 0, -300, -600, -1200, 0, 1200,
        0, -1200, -1400])
all_stations = []
for x, y in zip(stat_x, stat_y):
    all_stations.append(basic.location(x, y))

# Select only the stations I care about this will lead to better sampling
station_nums = [0, 5] # 1, 6
stations = []
for s in station_nums:
    stations.append(all_stations[s])

# Create Transition Kernel
transition_kernel = aps.transition_kernel(.5, .5**5, 1.0)

# Read in Q_true and Q to create the appropriate rho_D 
mdat = sio.loadmat('Q_2D')
Q = mdat['Q']
Q = Q[:, station_nums]
Q_true = mdat['Q_true']
Q_true = Q_true[15, station_nums] # 16th/20
bin_ratio = 0.15
bin_size = (np.max(Q, 0)-np.min(Q, 0))*bin_ratio

# Create experiment model
points = mdat['points']
def model(inputs):
    interp_values = np.empty((inputs.shape[0], Q.shape[1])) 
    for i in xrange(Q.shape[1]):
        interp_values[:, i] = griddata(points.transpose(), Q[:, i],
                inputs)
    return interp_values 

# Create heuristic
maximum = 1/np.product(bin_size)
def rho_D(outputs):
    rho_left = np.repeat([Q_true-.5*bin_size], outputs.shape[0], 0)
    rho_right = np.repeat([Q_true+.5*bin_size], outputs.shape[0], 0)
    rho_left = np.all(np.greater_equal(outputs, rho_left), axis=1)
    rho_right = np.all(np.less_equal(outputs, rho_right), axis=1)
    inside = np.logical_and(rho_left, rho_right)
    max_values = np.repeat(maximum, outputs.shape[0], 0)
    return inside.astype('float64')*max_values

heuristic_rD = aps.rhoD_heuristic(maximum, rho_D)

# Create sampler
num_batches = 125
samples_per_batch = 80
num_samples = num_batches*samples_per_batch
sampler = aps.sampler(num_samples, num_batches, model)

# Get samples
inital_sample_type = "lhs"
(samples, data, all_step_ratios) = sampler.generalized_chains(param_min, param_max,
        transition_kernel, heuristic_rD, sample_save_file, inital_sample_type)

# Read in points_true and plot results
p_true = mdat['points_true']
p_true = p_true[5:7,15]

        
