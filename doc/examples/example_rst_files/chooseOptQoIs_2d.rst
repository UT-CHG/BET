.. _chooseQoIs:


===========================
Example: Heatplate Model with Clustered Sampling
===========================

We will go through the example :download:`example
<../../../examples/heatplate/chooseOptQoIs_2d.py>`This example takes in samples, specifically chosen in clusters around 16 random points in lam_domain, and corresponding QoIs (data) from a simulation modeling the variations in temperature of a this plate forced by a localized source. It then calcualtes the gradients using an RBF (FFD or CFD) scheme and uses the gradient information to choose the optimal set of 2 QoIs to use in the inverse problem.

Import the necessary modules::


    import bet.sensitivity.gradients as grad
    import bet.sensitivity.chooseQoIs as cQoIs
    import bet.Comm as comm
    import scipy.io as sio
    import numpy as np

Import the data from the FEniCS simulation (RBF or FFD or CFD clusters)::

  matfile = sio.loadmat('heatplate_2d_16clustersRBF_1000qoi.mat')

Calculate the gradient vectors at each of the 16 centers for each of the QoI
maps::

    G = grad.calculate_gradients_rbf(samples, data)



