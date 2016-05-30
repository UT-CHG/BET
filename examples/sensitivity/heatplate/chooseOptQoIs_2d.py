# Copyright (C) 2014-2016 The BET Development Team

"""
This examples takes in samples, specifically chosen in clusters around 16 random
points in the inout space, and corresponding QoIs (data) from a simulation
modeling the variations in temperature of a thin plate forced by a localized
source.  It then calculates the gradients using an RBF (FFD or CFD) scheme and
uses the gradient information to choose the optimal set of 2 QoIs to use in the
inverse problem.

The optimal set of QoI is defined as the set that minimizes the average skewness
of the inverse image.
"""

import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoIs
import bet.Comm as comm
import scipy.io as sio
import numpy as np
import bet.sample as sample

# Import the data from the FEniCS simulation (RBF or FFD or CFD clusters)
matfile = sio.loadmat('heatplate_2d_16clustersRBF_1000qoi.mat')
#matfile = sio.loadmat('heatplate_2d_16clustersFFD_1000qoi.mat')
#matfile = sio.loadmat('heatplate_2d_16clustersCFD_1000qoi.mat')

# Initialize some sample objects we will need
input_samples = sample.sample_set(2)
output_samples = sample.sample_set(1000)

# Set the samples from the imported file
input_samples._values = matfile['samples']

# Make the MC assumption and compute the volumes of each voronoi cell
input_samples.estimate_volume_mc()

# Set the data fromthe imported file
output_samples._values = matfile['data']

# Calculate the gradient vectors at each of the 16 centers for each of the
# QoI maps
input_samples._jacobians = grad.calculate_gradients_rbf(input_samples,
    output_samples, normalize=False)
#G = grad.calculate_gradients_ffd(input_samples, output_sample)
#G = grad.calculate_gradients_cfd(input_samples, output_sample)

# With a set of QoIs to consider, we check all possible combinations
# of the QoIs and choose the best sets.
indexstart = 0
indexstop = 20
qoiIndices = range(indexstart, indexstop)

# Compute the skewness for each of the possible sets of QoI (20 choose 2 = 190)
skewness_indices_mat = cQoIs.chooseOptQoIs(input_samples, qoiIndices,
    num_optsets_return=190, measure=False)

qoi1 = skewness_indices_mat[0, 1]
qoi2 = skewness_indices_mat[0, 2]

if comm.rank==0:
    print 'The 10 smallest condition numbers are in the first column, the \
corresponding sets of QoIs are in the following columns.'
    print skewness_indices_mat[:10, :]

# Choose a specific set of QoIs to check the skewness of
index1 = 0
index2 = 4
(scpecific_skewness, _) = cQoIs.calculate_avg_skewness(input_samples, qoi_set=[index1, index2])
