.. _fromFile3DExample:

=======================================================================
Example: Batch Adaptive Sampling (3-to-3 example)
=======================================================================

.. note::

    This example shows how to generate adaptive samples in a specific
    way by implicitly defining an input event of interest. It does NOT
    show how to solve the stochastic inverse problem using these samples,
    which can be found by reading other examples. Thus, we only present
    the first few steps involved in discretizing the parameter and data
    spaces using a specific type of adaptive sampling. The user is
    referred to some other examples for filling in the remaining steps
    for solving the stochastic inverse problem following the construction
    of the adaptive samples.

We will walk through the following `example
<https://github.com/UT-CHG/BET/blob/master/examples/fromFile_ADCIRCMap/fromFile3D.py>`_
that uses a linear interpolant of a 3-dimensional QoI map used
to define a 3-dimensional data space.
The parameter space is also 3-dimensional.

This example specifically demonstrates the adaptive generation of samples
using  a
goal-oriented adaptive sampling algorithm.
This example is based upon the results shown in Section 8.6 of the
manuscript `Definition and solution
of a stochastic inverse problem for the Manning’s n parameter field in
hydrodynamic models <http://dx.doi.org/10.1016/j.advwatres.2015.01.011>`_
where the QoI map is given by
:math:`Q(\lambda) = (q_1(\lambda), q_5(\lambda), q_2(\lambda))`.
We refer the reader to that example for more information about the
physical interpretation of the parameter and data space, as well as
the physical locations of the observation stations defining the QoI map.

.. note::

    In this example, we have used ADCIRC to generate data files
    based on a regular discretization of the parameter space whose
    sole purpose is to create an (accurate) surrogate QoI map defined as a
    piecewise linear interpolant. This is quite different from many of the
    other examples, but the use of the surrogate QoI map is immaterial. The
    user could also interface the sampler directly to ADCIRC, but this would
    require a copy of ADCIRC, the finite element mesh, and significant
    training on the use of this state-of-the-art shallow water equation code.
    The primary focus of this example is the generation of adaptive samples.
    If the user knows how to use the ADCIRC model, then the user may instead
    opt to significantly change Step (1) below to interface to ADCIRC instead
    of to our "model" defined in terms of the surrogate QoI map.
    Interfacing to ADCIRC directly would likely require the use of `PolyADCIRC
    <https://github.com/UT-CHG/PolyADCIRC>`_.

.. note::

    This example is very similar to :ref:`fromFile2DExample` which involved
    a 2-to-2 map. The user may want to modify either example to involve fewer
    QoI's in the map (e.g., defining a 2-to-1 or 3-to-2 or 3-to-1 map). The
    example discussed in Section 8.6 of the paper referenced above discusses
    that the results for solving the stochastic inverse problem using a 3-to-3
    map are almost identical to those using a 3-to-2 map.

Generating a single set of adaptive samples
===========================================

Step (0): Setting up the environment
------------------------------------

Import the necessary modules::::

    import numpy as np
    import bet.sampling.adaptiveSampling as asam
    import bet.postProcess.plotDomains as pDom
    import scipy.io as sio
    from scipy.interpolate import griddata


Step (1): Define the interface to the model and goal-oriented adaptive sampler
------------------------------------------------------------------------------
This is where we interface the adaptive sampler imported above
to the model.
In other examples, we have imported a Python interface to a
computational model.
In this example,  we instead define the model as
a (piecewise-defined) linear interpolant to the QoI map :math:`Q(\lambda) =
(q_1(\lambda), q_5(\lambda), q_2(\lambda))` using data read from a ``.mat``
`file <https://github.com/UT-CHG/BET/blob/master/examples/matfiles/Q_3D.mat>`_::

    station_nums = [0, 4, 1] # 1, 5, 2
    mdat = sio.loadmat('Q_3D')
    Q = mdat['Q']
    Q = Q[:, station_nums]
    # Create experiment model
    points = mdat['points']
    def model(inputs):
        interp_values = np.empty((inputs.shape[0], Q.shape[1]))
        for i in xrange(Q.shape[1]):
            interp_values[:, i] = griddata(points.transpose(), Q[:, i],
                inputs)
        return interp_values


In this example, we use the adaptive sampler defined by
:class:`~bet.sampling.adaptiveSampling.rhoD_kernel`, which requires
an identification of a data distribution used to modify the transition
kernel for input samples. The idea is to place more samples in the
parameter space that correspond to a contour event of higher probability
as specified by the data distribution ``rho_D`` shown below.

First, we create the :mod:`~bet.sampling.adaptiveSampling.transition_set`
with an
initial step size ratio of 0.5 and a minimum, maximum step size ratio of
``.5**5`` and 1.0 respectively. Note that this algorithm only generates
samples inside the parameter domain, ``lam_domain`` (see Step (2) below)::

    # Create Transition Kernel
    transition_set = asam.transition_set(.5, .5**5, 1.0)

Here, we implicty designate a region of interest :math:`\Lambda_k =
Q^{-1}(D_k)` in :math:`\Lambda` for some :math:`D_k \subset \mathcal{D}`
through the use of the data distribution kernel.
In this instance we choose our kernel
:math:`p_k(Q) = \rho_\mathcal{D}(Q)`, see
:class:`~bet.sampling.adaptiveSampling.rhoD_kernel`.

We choose some :math:`\lambda_{ref}` and
let :math:`Q_{ref} = Q(\lambda_{ref})`::

    Q_ref = mdat['Q_true']
    Q_ref = Q_ref[14, station_nums] # 15th/20

We define a 3-D box, :math:`R_{ref} \subset \mathcal{D}` centered at
:math:`Q(\lambda_{ref})` with sides 15% the length of :math:`q_1`,
:math:`q_5`, and :math:`q_2`.
Set :math:`\rho_\mathcal{D}(q) = \frac{\mathbf{1}_{R_{ref}}(q)}{||\mathbf{1}_{R_{ref}}||}`::

    bin_ratio = 0.15
    bin_size = (np.max(Q, 0)-np.min(Q, 0))*bin_ratio
    # Create kernel
    maximum = 1/np.product(bin_size)
    def rho_D(outputs):
        rho_left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
        rho_right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
        rho_left = np.all(np.greater_equal(outputs, rho_left), axis=1)
        rho_right = np.all(np.less_equal(outputs, rho_right),axis=1)
        inside = np.logical_and(rho_left, rho_right)
        max_values = np.repeat(maximum, outputs.shape[0], 0)
        return inside.astype('float64')*max_values

    kernel_rD = asam.rhoD_kernel(maximum, rho_D)

The basic idea is that when the region of interest has been "found" by
some sample in a chain, the transition set is modified by the
adaptive sampler (it is made smaller) so that more samples are placed
within this event of interest.

Given a (M, mdim) data vector
:class:`~bet.sampling.adaptiveSampling.rhoD_kernel` expects that ``rho_D``
will return a :class:`~numpy.ndarray` of shape (M,).

Next, we create the :mod:`~bet.sampling.adaptiveSampling.sampler`. This
:mod:`~bet.sampling.adaptiveSampling.sampler` will create 80 independent
sampling chains that are each 125 samples long::

    # Create sampler
    chain_length = 125
    num_chains = 80
    num_samples = chain_length*num_chains
    sampler = asam.sampler(num_samples, chain_length, model)

.. note::

    * In the lines 54, 54 change ``chain_length`` and ``num_chains`` to
      reduce the total number of forward solves.
    * If ``num_chains = 1`` above, then this is no longer a "batch"
      sampling process where multiple chains are run simultaneously to
      "search for" the region of interest.
    * Saves to ``sandbox2d.mat``.

Step (2) [and Step (3)]: Describe and (adaptively) sample the input (and output) space
---------------------------------------------------------------------------------------

The adaptive sampling of the input space requires feedback from the
corresponding output samples, so the sets of samples are, in a sense,
created simultaneously in order to define the discretization of the
spaces used to solve the stochastic inverse problem.
While this can always be the case, in other examples, we often sampled the
input space completely in one step, and then propagated the samples
through the model to generate the QoI samples in another step, and
these two samples sets together were used to define the
discretization object used to solve the stochastic inverse problem.

The compact (bounded, finite-dimensional) paramter space for this
example is::

    lam_domain = np.array([[-900, 1500], [.07, .15], [.1, .2]])

We choose an initial sample type to seed the sampling chains, which
in this case comes from using Latin-Hypercube sampling::

    inital_sample_type = "lhs"

Finally, we adaptively generate the samples using
:meth:`~bet.sampling.adaptiveSampling.sampler.generalized_chains`::

    (my_disc, all_step_ratios) = sampler.generalized_chains(lam_domain,
        transition_set, kernel_rD, sample_save_file, inital_sample_type)

[OPTIONAL] We may choose to visualize the results by executing the
following code::

    # Read in points_ref and plot results
    ref_sample = mdat['points_true']
    ref_sample = ref_sample[:, 14]

    # Show the samples in the parameter space
    pDom.scatter_rhoD(my_disc, rho_D=rho_D, ref_sample=ref_sample, io_flag='input')
    # Show the corresponding samples in the data space
    pDom.scatter_rhoD(my_disc, rho_D=rho_D, ref_sample=Q_ref, io_flag='output')
    # Show the data domain that corresponds with the convex hull of samples in the
    # parameter space
    pDom.show_data_domain_2D(my_disc, Q_ref=Q_ref)

    # Show multiple data domains that correspond with the convex hull of samples in
    # the parameter space
    pDom.show_data_domain_multi(my_disc, Q_ref=Q_ref, showdim='all')

.. note::

    The user could simply run the example `plotDomains3D.py
    <https://github.com/UT-CHG/BET/tree/master/examples/fromFile_ADCIRCMap/plotDomains3D.py>`_
    to see the results for a previously generated set of adaptive
    samples.

Steps (4)-(5) [user]: Defining and solving a stochastic inverse problem
-----------------------------------------------------------------------

In the call to ``sampler.generalized_chains`` above, a discretization
object is created and saved. The user may wish to follow some of the other
examples (e.g., :ref:`linearMap` or :ref:`nonlinearMap`)
along with the paper referenced above to describe a data
distribution around a reference datum (Step (4)) and solve the stochastic
inverse problem (Step (5)) using the adaptively generated discretization
object by loading it from file. This can be done in a separate script
(but do not forget to do Step (0) which sets up the environment before
coding Steps (4) and (5)).


Generating and comparing several sets of adaptive samples
==========================================================
In some instances the user may want to generate and compare several sets of
adaptive samples using a surrogate model to determine what the best kernel,
transition set, number of generalized chains, and chain length are before
adaptively sampling a more computationally expensive model. See
`sandbox_test_3D.py <https://github.com/UT-CHG/BET/tree/master/examples/fromFile_ADCIRCMap/sandbox_test_3D.py>`_.
The set up in
`sandbox_test_3D.py <https://github.com/UT-CHG/BET/tree/master/examples/fromFile_ADCIRCMap/sandbox_test_3D.py>`_
is very similar to the
set up in `fromFile3D <https://github.com/UT-CHG/BET/tree/master/examples/fromFile_ADCIRCMap/fromFile3D.py>`_
and is
omitted for brevity.

We can explore several types of kernels::

    kernel_mm = asam.maxima_mean_kernel(np.array([Q_ref]), rho_D)
    kernel_m = asam.maxima_kernel(np.array([Q_ref]), rho_D)
    kernel_rD = asam.rhoD_kernel(maximum, rho_D)
    kern_list = [kernel_mm, kernel_rD, kernel_m]
    # Get samples
    # Run with varying kernels
    gen_results = sampler.run_gen(kern_list, rho_D, maximum, param_min,
            param_max, transition_set, sample_save_file)

We can explore :class:`~bet.sampling.adaptiveSampling.transition_set` with
various inital, minimum, and maximum step size ratios::

    # Run with varying transition sets bounds
    init_ratio = [0.1, 0.25, 0.5]
    min_ratio = [2e-3, 2e-5, 2e-8]
    max_ratio = [.5, .75, 1.0]
    tk_results = sampler.run_tk(init_ratio, min_ratio, max_ratio, rho_D,
            maximum, param_min, param_max, kernel_rD, sample_save_file)

We can explore a single kernel with varying values of ratios for increasing
and decreasing the step size (i.e. the size of the hyperrectangle to draw a new
step from using a transition set)::

    increase = [1.0, 2.0, 4.0]
    decrease = [0.5, 0.5e2, 0.5e3]
    tolerance = [1e-4, 1e-6, 1e-8]
    incdec_results = sampler.run_inc_dec(increase, decrease, tolerance, rho_D,
        maximum, param_min, param_max, transition_set, sample_save_file)

.. note::

    The above examples just use a ``zip`` combination of the lists uses to
    define varying parameters for the kernels and transition sets. To explore
    the product of these lists you need to use ``numpy.meshgrid`` and
    ``numpy.ravel`` or a similar process.

To compare the results in terms of yield or the total number of samples
generated in the region of interest we can use
:class:`~bet.sampling.basicSampling.compare_yield` to display the results to screen::

    # Compare the quality of several sets of samples
    print "Compare yield of sample sets with various kernels"
    bsam.compare_yield(gen_results[3], gen_results[2], gen_results[4])
    print "Compare yield of sample sets with various transition sets bounds"
    bsam.compare_yield(tk_results[3], tk_results[2], tk_results[4])
    print "Compare yield of sample sets with variouos increase/decrease ratios"
    bsam.compare_yield(incdec_results[3], incdec_results[2],incdec_results[4])

Here :meth:`~bet.sampling.basicSampling.compare_yield` simply displays to screen the
``sample_quality`` and ``run_param`` sorted by ``sample_quality`` and indexed
by ``sort_ind``.