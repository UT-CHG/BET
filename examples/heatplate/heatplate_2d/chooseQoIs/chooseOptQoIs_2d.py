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
import bet.calculateP as calculateP
import bet.calculateP.simpleFunP as simpleFunP
import bet.postProcess as postP
import bet.postProcess.plotP as plotP
import bet.postProcess.postTools as postTools
import bet.Comm as comm
import scipy.io as sio
import numpy as np



import bet.calculateP.calculateP as calculateP

import scipy.io

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

''' Need to solve the model over uniform random before doing this
#######################################
# With this information we can solve the inverse problem using these optimal
# sets of QoIs and visulaize the results.  First we set a reference point in
# the data space and then invert some region around that reference point
# into the parameter space.  We choose it to be data from the simulation.
Q_ref = data[0, [qoi1, qoi2]]
data_spec = data[:, [qoi1, qoi2]]

lam_domain= np.array([[0.01, 0.2],
                      [0.01, 0.2]])

(d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle(
    data=data_spec, Q_ref=Q_ref, bin_ratio=0.2, center_pts_per_edge=np.ones(( \
    data_spec.shape[1],)))


# calculate probabilities
(P,  lambda_emulate, io_ptr, emulate_ptr) = calculateP.prob_emulated( \
    samples=samples, data=data_spec, rho_D_M=d_distr_prob,
    d_distr_samples=d_distr_samples, lambda_emulate=samples, d_Tree=d_Tree)


(bins, marginals2D) = plotP.calculate_2D_marginal_probs(P_samples=P,
    samples=samples, lam_domain=lam_domain, nbins = [150, 150])


# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D,bins, sigma=0.05)

#plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, lam_domain,
    filename = 'heatplate', plot_surface=False, interactive=False)


'''


