# Copyright (C) 2014-2015 The BET Development Team

"""
This examples takes in samples, specifically chosen in clusters around 16 random
points in lam_domain, and corresponding QoIs (data) from a simulation modeling
the variations in temperature of a this plate forced by a localized source.
It then calcualtes the gradients using an RBF (FFD or CFD) scheme and uses the
gradient information to choose the optimal set of 2 QoIs to use in the inverse
problem.
"""

import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoIs
import bet.Comm as comm
import scipy.io as sio
import numpy as np

# Import the data from the FEniCS simulation (RBF or FFD or CFD clusters)
matfile = sio.loadmat('heatplate_2d_16clustersRBF_1000qoi.mat')
#matfile = sio.loadmat('heatplate_2d_16clustersFFD_1000qoi.mat')
#matfile = sio.loadmat('heatplate_2d_16clustersCFD_1000qoi.mat')

samples = matfile['samples']
data = matfile['data']
Lambda_dim = samples.shape[1]

# Calculate the gradient vectors at each of the 16 centers for each of the
# QoI maps
G = grad.calculate_gradients_rbf(samples, data)
#G = grad.calculate_gradients_ffd(samples, data)
#G = grad.calculate_gradients_cfd(samples, data)

# With a set of QoIs to consider, we check all possible combinations
# of the QoIs and choose the best sets.
indexstart = 0
indexstop = 20
qoiIndices = range(indexstart, indexstop)
condnum_indices_mat = cQoIs.chooseOptQoIs(G, qoiIndices)
qoi1 = condnum_indices_mat[0, 1]
qoi2 = condnum_indices_mat[0, 2]

if comm.rank==0:
    print 'The 10 smallest condition numbers are in the first column, the \
corresponding sets of QoIs are in the following columns.'
    print condnum_indices_mat

# Choose a specific set of QoIs to check the condition number of
index1 = 0
index2 = 4
singvals = np.linalg.svd(G[:, [index1, index2], :], compute_uv=False)
spec_condnum = np.sum(singvals[:,0]/singvals[:,-1], axis=0)/16
