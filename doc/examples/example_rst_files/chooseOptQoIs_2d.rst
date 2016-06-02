.. _chooseQoIs:


===========================
Example: Heatplate Model with Clustered Sampling
===========================

This example takes in samples, specifically chosen in clusters around 16 random points in Lambda, and corresponding QoIs (data) from a simulation modeling the variations in temperature of a thin plate forced by a localized source. It then calculates the gradients using a Radial Basis Function (or Forward Finite Difference or Centered Finite Difference) scheme and uses the gradient information to choose the optimal set of 2 QoIs to use in the inverse problem.  This optimality is with respect to the skewness of the gradient vectors.

Import the necessary modules::


    import bet.sensitivity.gradients as grad
    import bet.sensitivity.chooseQoIs as cQoIs
    import bet.Comm as comm
    import scipy.io as sio
    import numpy as np

Import the data from the FEniCS simulation (RBF or FFD or CFD clusters).  The parameter space needs to be sampled with respect to the gradient approximation scheme chosen::

  matfile = sio.loadmat('heatplate_2d_16clustersRBF_1000qoi.mat')

Calculate the gradient vectors at each of the 16 centers for each of the QoI
maps::

    G = grad.calculate_gradients_rbf(samples, data)

We have gathered 1,000 QoIs from the model.  To ensure we choose the best possible pair of QoIs to use in the inverse problem we would want to check the conditioning of all 1,000 choose 2 possible pairs.  To illustrate the functionality of the method, we simply consider the first 20 QoIs and choose the best pair from the 20 choose 2 options::

    indexstart = 0
    indexstop = 20
    qoiIndices = range(indexstart, indexstop)
    condnum_indices_mat = cQoIs.chooseOptQoIs(G, qoiIndices)

A smaller condition number corresponds to a better conditioned pair of QoIs.  We print the top 10 pairs of QoIs::

    if comm.rank==0:
        print 'The 10 smallest condition numbers are in the first column, the \
    corresponding sets of QoIs are in the following columns.'
        print condnum_indices_mat

If interested, we can find the condition number of any pair of QoIs::

    index1 = 0
    index2 = 4
    singvals = np.linalg.svd(G[:, [index1, index2], :], compute_uv=False)
    spec_condnum = np.sum(singvals[:,0]/singvals[:,-1], axis=0)/16
