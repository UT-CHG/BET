# Copyright (C) 2014-2015  BET Development Team

"""
This examples takes in samples, specifically chosen to be used for a
forward finite difference stencil, and corresponding QoIs (data) from the 4d heatplate model.
It then calcualte the gradients using this FFD scheme and uses the gradient information to 
choose the optimal set of 4 QoIs to use in the inverse problem.
"""

import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoIs
import numpy as np
import bet.Comm as comm
import scipy.io as sio

# Import the data from the FEniCS run
matfile = sio.loadmat('heatplate_2d_16clustersCFD_1000qoi.mat')
samples = matfile['samples']
data = matfile['data']

# We use 16 random points in lam_domain and approximate the gradient
# at each of them.
num_xeval = 16
xeval = samples[:16,:]
Lambda_dim = samples.shape[1]

# The parameter domain is a 4d box with the same bounds in
# each direction.
lam_min = 0.01
lam_max = 0.2

# The radius we chose for the FFD stencil is 1/100 of the
# range of each parameter.
r = (lam_max-lam_min)/100.

# Number of points we gather QoIs at in space, for each time step
num_points = 20

#####################################

# With the samples and data we calcualte the normalized gradient vectors a
# each of the 16 random points in lam_domain.
G = grad.calculate_gradients_cfd(samples=samples, data=data, xeval=xeval, r=r)

# We have 1,000 QoIs to choose from (50 time steps * 20 points).  Here we
# choose which QoI we want to start and end with.
pointstart = 1
timestepstart = 1
pointstop = 20
timestepstop = 10
indexstart = num_points*(timestepstart-1) + pointstart - 1
indexstop = num_points*(timestepstop-1) + pointstop - 1

# With a set of QoIs to consider, we check all possible combinations
# of the QoIs and choose the best set.
[min_condnum, qoiIndices] = cQoIs.chooseOptQoIs(G, indexstart, indexstop, num_qois_returned=Lambda_dim)

if comm.rank==0:
    print 'The minimum condition number found is : ', min_condnum
    print 'This corresponds to the set of QoIs : ', qoiIndices

#################################
# Choose a specific set of QoIs to test
point1 = 17
timestep1 = 1
point2 = 18
timestep2 = 1

index1 = num_points*(timestep1-1) + point1 - 1
index2 = num_points*(timestep2-1) + point2 - 1


singvals = np.linalg.svd(G[:, [index1, index2], :], compute_uv=False)
spec_condnum = np.sum(singvals[:,0]/singvals[:,-1], axis=0)/num_xeval



