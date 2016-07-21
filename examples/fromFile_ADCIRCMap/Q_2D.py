# Copyright (C) 2014-2015 The BET Development Team

import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as sfun
import numpy as np
import scipy.io as sio
import bet.sample as sample

# Import "Truth"
mdat = sio.loadmat('../matfiles/Q_2D')
Q = mdat['Q']
Q_ref = mdat['Q_true']

# Import Data
points = mdat['points']
lam_domain = np.array([[0.07, .15], [0.1, 0.2]])

# Create input, output, and discretization from data read from file
input_sample_set = sample.sample_set(points.shape[0])
input_sample_set.set_values(points.transpose())
input_sample_set.set_domain(lam_domain)
print "Finished loading data"

def postprocess(station_nums, ref_num):
    
    filename = 'P_q'+str(station_nums[0]+1)+'_q'+str(station_nums[1]+1)
    if len(station_nums) == 3:
        filename += '_q'+str(station_nums[2]+1)
    filename += '_ref_'+str(ref_num+1)

    data = Q[:, station_nums]
    output_sample_set = sample.sample_set(data.shape[1])
    output_sample_set.set_values(data)
    q_ref = Q_ref[ref_num, station_nums]

    # Create Simple function approximation
    # Save points used to parition D for simple function approximation and the
    # approximation itself (this can be used to make close comparisions...)
    output_probability_set = sfun.regular_partition_uniform_distribution_rectangle_scaled(\
            output_sample_set, q_ref, rect_scale=0.15,
            center_pts_per_edge=np.ones((data.shape[1],)))

    num_l_emulate = 1e4
    set_emulated = bsam.random_sample_set('r', lam_domain, num_l_emulate)
    my_disc = sample.discretization(input_sample_set, output_sample_set,
            output_probability_set, emulated_input_sample_set=set_emulated)

    print "Finished emulating lambda samples"

    # Calculate P on lambda emulate
    print "Calculating prob_on_emulated_samples"
    calcP.prob_on_emulated_samples(my_disc)
    sample.save_discretization(my_disc, filename, "prob_on_emulated_samples_solution")

    # Calclate P on the actual samples with assumption that voronoi cells have
    # equal size
    input_sample_set.estimate_volume_mc()
    print "Calculating prob"
    calcP.prob(my_disc)
    sample.save_discretization(my_disc, filename, "prob_solution")

    # Calculate P on the actual samples estimating voronoi cell volume with MC
    # integration
    calcP.prob_with_emulated_volumes(my_disc)
    print "Calculating prob_with_emulated_volumes"
    sample.save_discretization(my_disc, filename, "prob_with_emulated_volumes_solution")

# Post-process and save P and emulated points
ref_nums = [6, 11, 15] # 7, 12, 16
stations = [1, 4, 5] # 2, 5, 6

ref_nums, stations = np.meshgrid(ref_nums, stations)
ref_nums = ref_nums.ravel()
stations = stations.ravel()

for tnum, stat in zip(ref_nums, stations):
    postprocess([0, stat], tnum)

