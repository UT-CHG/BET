# Copyright (C) 2014-2016 The BET Development Team

"""
Consider a thin 2-dimensional (square) metal plate constructed by welding
together two rectangular metal plates of similar alloy types together.
The alloy types differ due to variations in the manufacturing of the
rectangular plates, so the thermal diffusivity is different on the left
and right sides of the resulting square plate.
We want to quantify uncertainties in these thermal diffusivities using
the heat equation to model experiments where the square plates are subject
to an external, localized, source at the center of the plate.
Assuming we have exactly two contact thermometers with which to record
exactly two temperature measurements during the experiment, the question
is the following: what are the optimal placements of these thermometers
in space-time?

See the manuscript at http://arxiv.org/abs/1601.06702 for details involving
this problem and the simulations used to generate the data in Section 5.1.
Here, we take the simulated data from the model problem saved as *.mat files
that contains parameter samples chosen in clusters around 16 random
points in the input space, and the corresponding QoIs (data) from the points
in space shown in Figure 6 of the manuscript at the 50 time steps of the
simulation (resulting in 1000 total different possible QoIs to choose from
in space-time).
We use the clusters of samples to compute gradients of the QoI using either
radial basis function, forward, or centered finite difference schemes.
These gradients are used to compute the average skewness in the possible 2D maps.
We then choose the optimal set of 2 QoIs to use in the inverse problem by
minimizing average skewness.
"""

import scipy.io as sio
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cqoi
import bet.Comm as comm
import bet.sample as sample

# Select the type of finite difference scheme as either RBF, FFD, or CFD
fd_scheme = 'RBF'

# Import the data from the FEniCS simulation (RBF or FFD or CFD clusters)
if fd_scheme.upper() in ['RBF', 'FFD', 'CFD']:
    file_name = 'heatplate_2d_16clusters' + fd_scheme.upper() + '_1000qoi.mat'
    matfile = sio.loadmat(file_name)
else:
    print('no data files for selected finite difference scheme')
    exit()

# Select a subset of QoI to check for optimality
'''
In Figure 6 of the manuscript at http://arxiv.org/abs/1601.06702, we see
that there are 20 spatial points considered and 50 time steps for a total
of 1000 different QoI.
The QoI are indexed so that the QoI corresponding to indices

    (i-1)*20 to i*20

for i between 1 and 50 corresponds to the 20 labeled QoI from Figure 6
at time step i.

Using this information, we can check QoI either across the entire range
of all space-time locations (``indexstart = 0``, ``indexstop = 1000``), or,
we can check the QoI at a particular time (e.g., setting ``indexstart=0`` and
``indexstop = 20`` considers all the spatial QoI only at the first time step).

In general, ``indexstart`` can be any integer between 0 and 998  and
``indexstop`` must be at least 2 greater than ``indexstart`` (so between
2 and 1000 subject to the additional constraint that ``indexstop``
:math: `\geq` ``indexstart + 2`` to ensure that we check at least a single pair
of QoI.)
'''
indexstart = 0
indexstop = 20
qoiIndices = range(indexstart, indexstop)

# Initialize the necessary sample objects
input_samples = sample.sample_set(2)
output_samples = sample.sample_set(1000)

# Set the input sample values from the imported file
input_samples.set_values(matfile['samples'])

# Set the data fromthe imported file
output_samples.set_values(matfile['data'])

# Create the cluster discretization
cluster_discretization = sample.discretization(input_samples, output_samples)

# Calculate the gradient vectors at each of the 16 centers for each of the
# QoI maps
if fd_scheme.upper() in ['RBF']:
    center_discretization = grad.calculate_gradients_rbf(cluster_discretization,
        normalize=False)
elif fd_scheme.upper() in ['FFD']:
    center_discretization = grad.calculate_gradients_ffd(cluster_discretization)
else:
    center_discretization = grad.calculate_gradients_cfd(cluster_discretization)

input_samples_centers = center_discretization.get_input_sample_set()

# Choose a specific set of QoIs to check the average skewness of
index1 = 0
index2 = 4
(specific_skewness, _) = cqoi.calculate_avg_skewness(input_samples_centers,
        qoi_set=[index1, index2])
if comm.rank == 0:
    print 'The average skewness of the QoI map defined by indices ' + str(index1) + \
        ' and ' + str(index2) + ' is ' + str(specific_skewness)

# Compute the skewness for each of the possible QoI maps determined by choosing
# any two QoI from the set defined by the indices selected by the
# ``indexstart`` and ``indexend`` values
skewness_indices_mat = cqoi.chooseOptQoIs(input_samples_centers, qoiIndices,
    num_optsets_return=10, measure=False)

qoi1 = skewness_indices_mat[0, 1]
qoi2 = skewness_indices_mat[0, 2]

if comm.rank == 0:
    print 'The 10 smallest condition numbers are in the first column, the \
corresponding sets of QoIs are in the following columns.'
    print skewness_indices_mat[:10, :]

