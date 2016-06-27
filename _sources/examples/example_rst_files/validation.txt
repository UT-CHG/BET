.. _validation:

============================================
Example: Validation
============================================

We will walk through the following `example
<https://github.com/UT-CHG/BET/blob/master/examples/validationExample/linearMap.py>`_. 
This 2D linear example shows that geometrically distinct QoI can
recreate a probability measure on the input parameter space
used to define the output probability measure. The user can
explore various discretization effects. 

1D and 2D marginals are calculated, smoothed, and plotted.
The actual process is quite simple requiring a total of 5 steps
to solve the stochastic inverse problem with BET excluding any
post-processing
the user may want.
The most complicated part of this problem is Step (4)
defining the
user distribution on the data space from propagated samples.
In general the user will probably not write code with various
options as was done here for pedagogical purposes.
We break down the actual example included with BET step-by-step
below, but first, to showcase the overall simplicitly, we show
the "entire" code (omitting setting the environment,
post-processing, and commenting) required
for solving
the stochastic inverse problem using some default options::

    sampler = bsam.sampler(my_model)

    input_samples = samp.sample_set(2)
    input_samples.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))
    input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=[30, 30])
    input_samples.estimate_volume_mc()

    my_discretization = sampler.compute_QoI_and_create_discretization(input_samples)

    num_samples_discretize_D = 10
    num_iid_samples = 1E5
    Partition_set = samp.sample_set(2)
    Monte_Carlo_set = samp.sample_set(2)
    Partition_set.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))
    Monte_Carlo_set.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))
    Partition_discretization = sampler.create_random_discretization('random',
                                                            Partition_set,
                                                            num_samples=num_samples_discretize_D)
    Monte_Carlo_discretization = sampler.create_random_discretization('random',
                                                            Monte_Carlo_set,
                                                            num_samples=num_iid_samples)
    simpleFunP.user_partition_user_distribution(my_discretization,
                                            Partition_discretization,
                                            Monte_Carlo_discretization)

    calculateP.prob(my_discretization)

Step (0): Setting up the environment
===========================
Import the necessary modules::

    import numpy as np
    import bet.calculateP.simpleFunP as simpleFunP
    import bet.calculateP.calculateP as calculateP
    import bet.postProcess.plotP as plotP
    import bet.postProcess.plotDomains as plotD
    import bet.sample as samp
    import bet.sampling.basicSampling as bsam


Step (1): Define interface to the model
===========================
Import the Python script interface to the (simple Python) `model
<https://github.com/UT-CHG/BET/blob/master/examples/validationExample/myModel.py>`_
that takes as input a numpy array of model input parameter samples,
generated from the sampler (see below), evaluates the model to
generate QoI samples, and returns the QoI samples::

    from myModel import my_model

Define the sampler that will be used to create the discretization
object, which is the fundamental object used by BET to compute
solutions to the stochastic inverse problem.
The sampler and my_model is the interface of BET to the model,
and it allows BET to create input/output samples of the model::

    sampler = bsam.sampler(my_model)


Step (2): Describe and sample the input space
===========================
Initialize the (2-dimensional) input parameter sample set object
and set the parameter domain to be a unit-square::

    input_samples = samp.sample_set(2)
    input_samples.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))

Suggested changes for user (1)
------------------------------
Try with and without random sampling.

If using random sampling, try ``num_samples = 1E3`` and
``num_samples = 1E4``.
See what happens when ``num_samples = 1E2``.
Try using ``'lhs'`` instead of ``'random'`` in the random_sample_set.

If using regular sampling, try different numbers of samples
per dimension::

    randomSampling = False
    if randomSampling is True:
        input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E3)
    else:
        input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=[30, 30])

Suggested changes for user (2)
------------------------------
A standard Monte Carlo (MC) assumption is that every Voronoi cell
has the same volume. If a regular grid of samples was used, then
the standard MC assumption is true.

See what happens if the MC assumption is not assumed to be true, and
if different numbers of points are used to estimate the volumes of
the Voronoi cells::

    MC_assumption = True
    if MC_assumption is False:
        input_samples.estimate_volume(n_mc_points=1E5)
    else:
        input_samples.estimate_volume_mc()


Step (3): Generate QoI samples
===========================

Create the discretization object holding all the input (parameter) samples
and output (QoI) samples using the sampler::

    my_discretization = sampler.compute_QoI_and_create_discretization(input_samples,
                                               savefile = 'Validation_discretization.txt.gz')

At this point, all of the model information has been extracted for BET,
so the model is no
longer required for evaluation.
The user could do Steps (0)-(3) in a separate script, and then simply load
the discretization object as part of a separate BET script that does the
remaining steps.
When the model is expensive to evaluate, this is an attractive option
since we can now solve the stochastic inverse problem (with many
different distributions defined on the data space) without ever
having to re-solve the model (so long as we are happy with the resolution
provided by the current discretization of the parameter and data spaces).


Step (4): Describe the data distribution
===========================
This problem is nominally a "parameter distribution estimation"
problem and not a "parameter identification under uncertainty" problem
(e.g., see :ref:`linearMap` or almost any of the other examples).
Thus, unlike most other examples, the distribution on data space is
not coming from uncertain data but rather variable input parameters
that vary according to a fixed distribution. The goal is to determine
this distribution by inverting the observed distribution on the data
space (via discretizing the data space and binning samples).

Suggested changes for user (3)
------------------------------
Compute the output distribution simple function approximation by
propagating a different set of samples to implicitly define a Voronoi
discretization of the data space, corresponding to an implicitly defined
set of contour events defining a discretization of the input parameter
space. The probabilities of the Voronoi cells in the data space (and
thus the probabilities of the corresponding contour events in the
input parameter space) are determined by Monte Carlo sampling using
a set of i.i.d. uniform samples to bin into these cells.

See the effect of using different values for ``num_samples_discretize_D``.
Choosing ``num_samples_discretize_D = 1``
produces exactly the right answer and is equivalent to assigning a
uniform probability to each data sample above (why?).
Try setting this to 2, 5, 10, 50, and 100. Can you explain what you
are seeing? To see an exaggerated effect, try using random sampling
above with ``n_samples`` set to 1E2::

    num_samples_discretize_D = 1
    num_iid_samples = 1E5

    Partition_set = samp.sample_set(2)
    Monte_Carlo_set = samp.sample_set(2)

    Partition_set.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))
    Monte_Carlo_set.set_domain(np.repeat([[0.0, 1.0]], 2, axis=0))

    Partition_discretization = sampler.create_random_discretization('random',
                                                                Partition_set,
                                                                num_samples=num_samples_discretize_D)

    Monte_Carlo_discretization = sampler.create_random_discretization('random',
                                                                Monte_Carlo_set,
                                                                num_samples=num_iid_samples)

    simpleFunP.user_partition_user_distribution(my_discretization,
                                                Partition_discretization,
                                                Monte_Carlo_discretization)



Step (5): Solve the stochastic inverse problem
===========================
Calculate probablities on the parameter space (which are stored within
the discretization object)::

    calculateP.prob(my_discretization)



Step (6) [Optional]: Post-processing
===========================
Show some plots of the different sample sets::

    plotD.scatter_2D(my_discretization._input_sample_set, filename = 'Parameter_Samples.eps')
    plotD.scatter_2D(my_discretization._output_sample_set, filename = 'QoI_Samples.eps')
    plotD.scatter_2D(my_discretization._output_probability_set, filename = 'Data_Space_Discretization.eps')

There are ways to determine "optimal" smoothing parameters (e.g., see CV, GCV,
and other similar methods), but we have not incorporated these into the code
as lower-dimensional marginal plots generally have limited value in understanding
the structure of a high dimensional non-parametric probability measure.

The user may want to change ``nbins`` or ``sigma`` in the ``plotP.*`` inputs
(which influences the
kernel density estimation with smaller values of ``sigma`` implying a density
estimate that
looks more like a histogram and larger values smoothing out the values
more).

In general, the user will have to tune these for any given problem especially
when looking at marginals of higher-dimensional problems with parameter ranges
that have disparate scales (assuming the parameters were not first normalized
as part of a "un-dimensionalization" of the space, which is highly encouraged)::

    # calculate 2d marginal probs
    (bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                            nbins = [30, 30])

    # plot 2d marginals probs
    plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "validation_raw",
                                 file_extension = ".eps", plot_surface=False)

    # smooth 2d marginals probs (optional)
    marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.1)

    # plot 2d marginals probs
    plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "validation_smooth",
                                 file_extension = ".eps", plot_surface=False)

    # calculate 1d marginal probs
    (bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                            nbins = [30, 30])

    # plot 2d marginal probs
    plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "validation_raw",
                                 file_extension = ".eps")

    # smooth 1d marginal probs (optional)
    marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.1)

    # plot 2d marginal probs
    plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "validation_smooth",
                                 file_extension = ".eps")