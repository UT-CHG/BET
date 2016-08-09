.. _chooseQoIs:


===========================
Example: Optimizing space-time measurements of temperature on a thin plate
===========================
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

See the manuscript `Experimental Design : Optimizing Quantities of Interest to
Reliably Reduce the Uncertainty in Model Input Parameters
<http://arxiv.org/abs/1601.06702>`_ for details involving
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

We do not solve the stochastic inverse problem in this example.
We simply show the process of choosing the optimal QoI according to the criteria
of minimizing the average skewness property (to see more about skewness, the
interested reader should refer to
`Definition and solution of a stochastic inverse problem for the Manning’s n parameter field in
hydrodynamic models <http://dx.doi.org/10.1016/j.advwatres.2015.01.011>`_
where the concept was initially introduced).

This example takes in samples, specifically chosen in clusters around 16 random points in Lambda, and corresponding QoIs (data) from a simulation modeling the variations in temperature of a thin plate forced by a localized source. It then calculates the gradients using a Radial Basis Function (or Forward Finite Difference or Centered Finite Difference) scheme and uses the gradient information to choose the optimal set of 2 QoIs to use in the inverse problem.  This optimality is with respect to the skewness of the gradient vectors.


Step (0): Setting up the environment
===========================
Import the necessary modules::

    import scipy.io as sio
    import bet.sensitivity.gradients as grad
    import bet.sensitivity.chooseQoIs as cqoi
    import bet.Comm as comm
    import bet.sample as sample


Step (0*): Understanding your data and computing derivatives
============================================================
Computing the skewness (or other criteria such as scaling of measures of
inverse sets described in `Experimental Design : Optimizing Quantities of Interest to
Reliably Reduce the Uncertainty in Model Input Parameters
<http://arxiv.org/abs/1601.06702>`_) requires a sensitivity analysis.
If the code used to generate possible QoI data does not also produce derivatives
with respect to model parameters (e.g., by using adjoints), then we can use
several different types of finite differencing.
Assuming the user wants to work strictly with random sampling (to possibly
re-use samples in the parameter space for solving the resulting stochastic
inverse problem), then we can compute derivatives using a radial basis function
finite difference scheme.
Otherwise, we can use typical finite differencing schemes (forward or centered)
on regular grids of parameter samples.
Understanding the code, derivative capabilities, and/or the types of sampling
in the parameter space is crucial to setting up how the gradients of QoI are
computed/loaded into this code.
We provide data files for different types of sampling in the parameter space
in ``BET/examples/``::

    # Select the type of finite difference scheme as either RBF, FFD, or CFD
    fd_scheme = 'RBF'

    # Import the data from the FEniCS simulation (RBF or FFD or CFD clusters)
    if fd_scheme.upper() in ['RBF', 'FFD', 'CFD']:
        file_name = 'heatplate_2d_16clusters' + fd_scheme.upper() + '_1000qoi.mat'
        matfile = sio.loadmat(file_name)
    else:
        print('no data files for selected finite difference scheme')
        exit()

Step (1): Define the space of possible QoI maps
==============================================
In Figure 6 of the manuscript at
`Experimental Design : Optimizing Quantities of Interest to
Reliably Reduce the Uncertainty in Model Input Parameters
<http://arxiv.org/abs/1601.06702>`_, we see
that there are 20 spatial points considered and 50 time steps for a total
of 1000 different QoI.
Since we assume we can only choose 2 of the possible QoI to define a particular
QoI map, then we can define a space :math:`\mathcal{Q}` of possible QoI maps
by this set of 1000 choose 2 possible combinations of measurements.

However, we can define a :math:`\mathcal{Q}` of smaller cardinality by restricting
the possible maps subject to certain considerations.
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
:math:`\geq` ``indexstart + 2`` to ensure that we check at least a single pair
of QoI.)::

    indexstart = 0
    indexstop = 20
    qoiIndices = range(indexstart, indexstop)

Step (2): Create the discretization object from the input and output samples
============================================================================
Load the sampled parameter and QoI values::

    # Initialize the necessary sample objects
    input_samples = sample.sample_set(2)
    output_samples = sample.sample_set(1000)

    # Set the input sample values from the imported file
    input_samples.set_values(matfile['samples'])

    # Set the data fromthe imported file
    output_samples.set_values(matfile['data'])

    # Create the cluster discretization
    cluster_discretization = sample.discretization(input_samples, output_samples)

Step (3): Compute the gradients of all the maps in :math:`\mathcal{Q}`
======================================================================
Using whichever finite difference scheme we have chosen for our sample set based
on Step (0*) above, we now compute the gradients of each component of the QoI maps
defining :math:`\mathcal{Q}`::

    # Calculate the gradient vectors at each of the 16 centers for each of the
    # QoI maps
    if fd_scheme.upper() in ['RBF']:
        center_discretization = grad.calculate_gradients_rbf(cluster_discretization,
            normalize=False)
    elif fd_scheme.upper() in ['FFD']:
        center_discretization = grad.calculate_gradients_ffd(cluster_discretization)
    else:
        center_discretization = grad.calculate_gradients_cfd(cluster_discretization)

Step (4): Compute skewness properties of the maps and display information
=========================================================================
We now examine the geometric property of average skewness on the possible QoI maps.
The skewness defines an ordering on :math:`\mathcal{Q}` that is useful in selecting
the optimal QoI map. We can also compute skewness properties for any particular QoI map
we want individually. The first step is extracting the subset of samples for which we
actually computed derivatives, which are the centers of the clusters determined above::

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