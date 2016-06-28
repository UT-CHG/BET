.. _fenicsMultipleSerialExample:

===================
Example: Multiple Serial FEniCS
===================

We will walk through the following `example
<https://github.com/UT-CHG/BET/blob/master/examples/FEniCS/BET_multiple_serial_models_script.py>`_.
This example will run a serial version of BET and multiple (serial) runs
of a model using different parameters.
The purpose of this example is to porivde a template on how to make use of
 `Launcher <https://github.com/TACC/launcher>`_
(typically used on a cluster)
to launch multiple
instances of a serial code to more efficiently do parameter sweeps.
If we have k processors and sample the model N=nk times, then the
wall clock time spent solving the model goes from N to n (there is
some additional overhead with the I/O).
On a local machine with 2-8 processors, since this particular model is
relatively cheap to evaluate, the overhead of the I/O makes this
less efficient than the purely serial :ref:`fenicsExample`.
If we used a more refined mesh and/or more computationally expensive
model where the model solve time was significantly more than the
reading/writing of files, then this approach is faster.

This example requires the following external packages not shipped
with BET:

* An installation of `FEniCS <http://fenicsproject.org/>`_
  that can be run using the same python as used for installing BET.

* A copy of `Launcher <https://github.com/TACC/launcher>`_.
  The user needs to set certain environment
  variables inside of `lbModel.py
  <https://github.com/UT-CHG/BET/blob/master/examples/FEniCS/lbModel.py>`_
  for this to run.

This example generates samples for a KL expansion associated with
a covariance defined by ``cov`` in `computeSaveKL.py
<https://github.com/UT-CHG/BET/blob/master/examples/FEniCS/computeSaveKL.py>`_
on an L-shaped mesh
that defines the permeability field for a Poisson equation solved in
`myModel_serial.py
<https://github.com/UT-CHG/BET/blob/master/examples/FEniCS/myModel_serial.py>`_.

The quantities of interest (QoI) are defined as two spatial
averages of the solution to the PDE.

The user defines the dimension of the parameter space (corresponding
to the number of KL terms) and the number of samples in this space.

Even though we are coupling to the state-of-the-art FEniCS code for
solving a PDE, we again see that the actual process for solving
the stochastic inverse problem is quite simple requiring a total of
5 steps with BET excluding any
post-processing
the user may want.
In general the user will probably not write code with various
options as was done here for pedagogical purposes.
We break down the actual example included with BET step-by-step
below, but first, to showcase the overall simplicitly, we show
the "entire" code (omitting setting the environment,
post-processing, and commenting) required
for solving
the stochastic inverse problem using some default options::

    sampler = bsam.sampler(lb_model)

    num_KL_terms = 2
    computeSaveKL(num_KL_terms)
    input_samples = samp.sample_set(num_KL_terms)
    KL_term_min = -3.0
    KL_term_max = 3.0
    input_samples.set_domain(np.repeat([[KL_term_min, KL_term_max]],
                                       num_KL_terms,
                                       axis=0))
    input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=[10, 10])
    input_samples.estimate_volume_mc()

    my_discretization = sampler.compute_QoI_and_create_discretization(input_samples)

    param_ref = np.ones((1,num_KL_terms))
    Q_ref = my_model(param_ref)
    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref[0,:], rect_scale=0.1,
        center_pts_per_edge=3)

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
Import the Python script interface to the `load balancing model
<https://github.com/UT-CHG/BET/blob/master/examples/FEniCS/lbModel.py>`_
that takes as input a numpy array of model input parameter samples,
generated from the sampler (see below), creates the Launcher job file
for running and evaluating multiple serial instances of the model to
generate batches of QoI samples, and returns the QoI samples::

    from lbModel import lb_model

Define the sampler that will be used to create the discretization
object, which is the fundamental object used by BET to compute
solutions to the stochastic inverse problem.
The sampler and my_model is the interface of BET to the model,
and it allows BET to create input/output samples of the model::

    sampler = bsam.sampler(lb_model)


Step (2): Describe and sample the input space
===========================
We compute and save the KL expansion once so that this part, which
can be computationally expensive, can be done just once and then
commented out for future runs of the code using the same set of KL
coefficients defining the parameter space::

    from Compute_Save_KL import computeSaveKL
    num_KL_terms = 2
    computeSaveKL(num_KL_terms)

We then initialize the parameter space and assume that any KL
coefficient belongs to the interval [-3.0,3.0]::

    input_samples = samp.sample_set(num_KL_terms)
    KL_term_min = -3.0
    KL_term_max = 3.0
    input_samples.set_domain(np.repeat([[KL_term_min, KL_term_max]],
                                   num_KL_terms,
                                   axis=0))


Suggested changes for user (1)
------------------------------
Try with and without random sampling.

If using regular sampling, try different numbers of samples
per dimension (note that if ``num_KL_terms`` is not equal to 2, then
the user needs to be careful using regular sampling)::

    randomSampling = False
    if randomSampling is True:
        input_samples = sampler.random_sample_set('random', input_samples, num_samples=1E2)
    else:
        input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=[10, 10])

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

    my_discretization = sampler.compute_QoI_and_create_discretization(
                                input_samples, savefile='FEniCS_Example.txt.gz')

At this point, all of the model information has been extracted for BET
(with the possibly exception of evaluating the model to generate a
reference QoI datum or a distribution of the QoI), so the model is no
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
This problem is nominally a "parameter identification under uncertainty"
problem.
Thus, we take a reference QoI datum (from one more model solve), and
define a distribution "around" this datum.

Suggested changes for user (3)
------------------------------
Try different reference parameters that produce different
reference QoI data.::

    param_ref = np.ones((1,num_KL_terms))
    Q_ref = my_model(param_ref)

Use the reference samples and discretization to generate plots (this
is completely optional)::

    plotD.scatter_2D(input_samples, ref_sample=param_ref[0,:],
                     filename='FEniCS_ParameterSamples.eps')
    if Q_ref.size == 2:
        plotD.show_data_domain_2D(my_discretization, Q_ref=Q_ref[0,:],
                file_extension="eps")

Suggested changes for user (4)
------------------------------
Try different ways of discretizing the probability measure on D defined
as a uniform probability measure on a rectangle or interval depending
on choice of QoI_num in `myModel.py
<https://github.com/UT-CHG/BET/blob/master/examples/FEniCS/myModel.py>`_::

    randomDataDiscretization = False
    if randomDataDiscretization is False:
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
            data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.25,
            center_pts_per_edge = 3)
    else:
        simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled(
            data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.25,
            M=50, num_d_emulate=1E5)



Step (5): Solve the stochastic inverse problem
===========================
Calculate probablities on the parameter space (which are stored within
the discretization object)::

    calculateP.prob(my_discretization)



Step (6) [Optional]: Post-processing
===========================
The user may want to play around with ``nbins`` and ``sigma`` if different
input domains or different discretizations other than the defaults above
are used::

    (bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                            nbins=20)
    marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.5)
    plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename="FEniCS",
                                 lam_ref=param_ref[0,:], file_extension=".eps",
                                 plot_surface=False)

    (bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                            nbins=20)
    marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.5)
    plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename="FEniCS",
                                 lam_ref=param_ref[0,:], file_extension=".eps")






