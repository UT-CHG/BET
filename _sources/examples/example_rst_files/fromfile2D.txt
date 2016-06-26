.. _fromFile2DExample:

=======================================================================
Example: Generalized Chains with a 2,2-dimensional data,parameter space
=======================================================================

This example demonstrates the adaptive generation of samples using  a
goal-oriented adaptive sampling algorithm.

Generating a single set of adaptive samples
-------------------------------------------

We will walk through the following `example
<https://github.com/UT-CHG/BET/blob/master/examples/fromFile_ADCIRCMap/fromFile2D.py>`_ that uses a linear interpolant of
the QoI map :math:`Q(\lambda) = (q_1(\lambda), q_6(\lambda))` for a
2-dimensional data space. The parameter space in this example is also
2-dimensional.

.. note::

    * In the lines 56, 57 change chain length and num chains to
      reduce the total number of forward solves.
    * Saves to ``sandbox2d.mat``.

The modules required by this example are::

    import numpy as np
    import bet.sampling.adaptiveSampling as asam
    import scipy.io as sio
    from scipy.interpolate import griddata

The compact (bounded, finite-dimensional) paramter space is::

    # [[min \lambda_1, max \lambda_1], [min \lambda_2, max \lambda_2]]
    lam_domain = np.array([[.07, .15], [.1, .2]])
    param_min = lam_domain[:, 0]
    param_max = lam_domain[:, 1]

In this example we form a linear interpolant to the QoI map :math:`Q(\lambda) =
(q_1(\lambda), q_6(\lambda))` using data read from a ``.mat`` :download:`file
<../../../examples/fromFile_ADCIRCMap/Q_2D.mat>`::

    station_nums = [0, 5] # 1, 6
    mdat = sio.loadmat('Q_2D')
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

Next, we implicty designate the region of interest :math:`\Lambda_k =
Q^{-1}(D_k)` in :math:`\Lambda` for some :math:`D_k \subset \mathcal{D}`
through the use of some kernel. In this instance we choose our kernel
:math:`p_k(Q) = \rho_\mathcal{D}(Q)`, see
:class:`~bet.sampling.adaptiveSampling.rhoD_kernel`.

We choose some :math:`\lambda_{ref}` and let :math:`Q_{ref} = Q(\lambda_{ref})`::

    Q_ref = mdat['Q_true']
    Q_ref = Q_ref[15, station_nums] # 16th/20

We define a rectangle, :math:`R_{ref} \subset \mathcal{D}` centered at
:math:`Q(\lambda_{ref})` with sides 15% the length of :math:`q_1` and
:math:`q_6`. Set :math:`\rho_\mathcal{D}(q) = \frac{\mathbf{1}_{R_{ref}}(q)}{||\mathbf{1}_{R_{ref}}||}`::

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

We create the :mod:`~bet.sampling.adaptiveSampling.transition_set` with an
initial step size ratio of 0.5 and a minimum, maximum step size ratio of
``.5**5`` and 1.0 respectively. Note that this algorithm will not generate
samples out side of the bounded parameter domain, ``lambda_domain`` ::

    # Create Transition Kernel
    transition_set = asam.transition_set(.5, .5**5, 1.0)

We choose an initial sample type to seed the sampling chains::

    inital_sample_type = "lhs"

Finally, we adaptively generate the samples using
:meth:`~bet.sampling.adaptiveSampling.sampler.generalized_chains`::

    (samples, data, all_step_ratios) = sampler.generalized_chains(param_min,
        param_max, transition_set, kernel_rD, sample_save_file,
        inital_sample_type)

Generating and comparing several sets of adaptive samples
---------------------------------------------------------
In some instances the user may want to generate and compare several sets of
adaptive samples using a surrogate model to determine what the best kernel,
transition set, number of generalized chains, and chain length are before
adaptively sampling a more computationally expensive model. See
`sandbox_test_2D.py <https://github.com/UT-CHG/BET/tree/master/examples/fromFile_ADCIRCMap/sandbox_test_2D.py>`_. The set up in
sandbox_test_2D.py <https://github.com/UT-CHG/BET/tree/master/examples/fromFile_ADCIRCMap/sandbox_test_2D.py>`_ is very similar to the
set up in `fromFile2D <https://github.com/UT-CHG/BET/tree/master/examples/fromFile_ADCIRCMap/fromFile2D.py>`_ and is
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

..note:: The above examples just use a ``zip`` combination of the lists uses to
define varying parameters for the kernels and transition sets. To explore
the product of these lists you need to use ``numpy.meshgrid`` and
``numpy.ravel`` or a similar process.

To compare the results in terms of yield or the total number of samples
generated in the region of interest we can use
`~bet.sampling.basicSampling.compare_yield` to display the results to screen::

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

