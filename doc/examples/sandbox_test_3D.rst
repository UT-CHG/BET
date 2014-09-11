=======================================================================
Example: Generalized Chains with a 3,3-dimensional data,parameter space
=======================================================================

This example demonstrates the adaptive generation of samples using the
algorithm described in :ref:`adaptive-sampling`.

Generating a single set of adaptive samples
-------------------------------------------

We will walk through the following :download:`example
<../examples/fromFileMap/sandbox_test_3D.py>` that uses a linear interpolant of the QoI map
:math:`Q(\lambda) = (q_1(\lambda), q_5(\lambda), q_2(\lambda))` for a 3-dimensional data
space, see :ref:`inlet-data`. The parameter space in this example is also
3-dimensional, see :ref:`inlet-3d-parameter`. 

The modules required by this example are::

    import numpy as np
    import polysim.pyADCIRC.basic as basic
    import bet.sampling.adaptiveSampling as asam
    import bet.sampling.basicSampling as bsam
    import scipy.io as sio
    from scipy.interpolate import griddata

The compact (bounded, finite-dimensional) paramter space is::

    # [[min \lambda_1, max \lambda_1],..., [min \lambda_3, max \lambda_3]]
    param_domain = np.array([[-900, 1500], [.07, .15], [.1, .2]])
    param_min = param_domain[:, 0]
    param_max = param_domain[:, 1]

In this example we form a linear interpolant to the QoI map :math:`Q(\lambda) =
(q_1(\lambda), q_5(\lambda), q_2(\lambda))` using data read from a ``.mat`` :download:`file
<../examples/Q_3D.mat>`::

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

Next, we implicty designate the region of interest :math:`\Lambda_k =
Q^{-1}(D_k)` in :math:`\Lambda` for some :math:`D_k \subset \mathcal{D}`
through the use of some heuristic. In this instance we choose our heuristic
:math:`p_k(Q) = \rho_\mathcal{D}(Q)`, see
:class:`~bet.sampling.adaptiveSampling.rhoD_heuristic`.

We choose some :math:`\lambda_{true}` and let :math:`Q_{true} = Q(\lambda_{true})`::

    Q_true = mdat['Q_true']
    Q_true = Q_true[14, station_nums] # 15th/20

We define a rectangle, :math:`R_{true} \subset \mathcal{D}` centered at
:math:`Q(\lambda_{true})` with sides 15% the length of :math:`q_1` and
:math:`q_6`. Set :math:`\rho_\mathcal{D}(q) = \frac{\mathbf{1}_{R_{true}}(q)}{||\mathbf{1}_{R_{true}}||}`::

    bin_ratio = 0.15
    bin_size = (np.max(Q, 0)-np.min(Q, 0))*bin_ratio
    # Create heuristic
    maximum = 1/np.product(bin_size)
    def rho_D(outputs):
        rho_left = np.repeat([Q_true-.5*bin_size], outputs.shape[0], 0)
        rho_right = np.repeat([Q_true+.5*bin_size], outputs.shape[0], 0)
        rho_left = np.all(np.greater_equal(outputs, rho_left), axis=1)
        rho_right = np.all(np.less_equal(outputs, rho_right),axis=1)
        inside = np.logical_and(rho_left, rho_right)
        max_values = np.repeat(maximum, outputs.shape[0], 0)
        return inside.astype('float64')*max_values

    heuristic_rD = aps.rhoD_heuristic(maximum, rho_D)

Given a (M, mdim) data vector
:class:`~bet.sampling.adaptiveSampling.rhoD_heuristic` expects that ``rho_D``
will return a :class:`~numpy.ndarray` of shape (M,). 

Next, we create the :mod:`~bet.sampling.adaptiveSampling.sampler`. This
:mod:`~bet.sampling.adaptiveSampling.sampler` will create 80 independent
sampling chains that are each 125 samples long::

    # Create sampler
    chain_length = 125
    num_chains = 80
    num_samples = chain_length*num_chains
    sampler = aps.sampler(num_samples, chain_length, model)

We create the :mod:`~bet.sampling.adaptiveSampling.transition_kernel` with an
initial step size ratio of 0.5 and a minimum, maximum step size ratio of
``.5**5`` and 1.0 respectively. Note that this algorithm will not generate
samples out side of the bounded parameter domain, ``lambda_domain`` ::

    # Create Transition Kernel
    transition_kernel = aps.transition_kernel(.5, .5**5, 1.0)

We choose an initial sample type to seed the sampling chains::

    inital_sample_type = "lhs"

Finally, we adaptively generate the samples using
:meth:`~bet.sampling.adaptiveSampling.sampler.generalized_chains`::

    (samples, data, all_step_ratios) = sampler.generalized_chains(param_min,
        param_max, transition_kernel, heuristic_rD, sample_save_file,
        inital_sample_type)

Generating and comparing several sets of adaptive samples
---------------------------------------------------------
In some instances the user may want to generate and compare several sets of
adaptive samples using a surrogate model to determine what the best heuristic,
transition kernel, number of generalized chains, and chain length are before
adaptively sampling a more computationally expensive model. See
:download:`sandbox_test_2D.py <../examples/fromFileMap/sandbox_test_2D.py>`. The set up in
:download:`sandbox_test_2D.py <../examples/fromFileMap/sandbox_test_2D.py>` is very similar to the
set up in :download:`fromFile2D <../examples/fromFileMap/fromFile2D.py>` and is
omitted for brevity.

We can explore several types of heuristics::

    heuristic_mm = asam.maxima_mean_heuristic(np.array([Q_true]), rho_D)
    heuristic_rD = asam.rhoD_heuristic(maximum, rho_D)
    heuristic_m = asam.maxima_heuristic(np.array([Q_true]), rho_D)
    heuristic_md = asam.multi_dist_heuristic()
    heur_list = [heuristic_mm, heuristic_rD, heuristic_m, heuristic_md]
    # Get samples
    # Run with varying heuristics
    gen_results = sampler.run_gen(heur_list, rho_D, maximum, param_min,
            param_max, transition_kernel, sample_save_file)

We can explore :class:`~bet.sampling.adaptiveSampling.transition_kernel` with
various inital, minimum, and maximum step size ratios::

    # Run with varying transition kernels bounds
    init_ratio = [0.1, 0.25, 0.5]
    min_ratio = [2e-3, 2e-5, 2e-8]
    max_ratio = [.5, .75, 1.0]
    tk_results = sampler.run_tk(init_ratio, min_ratio, max_ratio, rho_D,
            maximum, param_min, param_max, heuristic_rD, sample_save_file)

We can explore a single heuristic with varying values of ratios for increasing
and decreasing the step size (i.e. the size of the hyperrectangle to draw a new
step from using a transition kernel)::

    increase = [1.0, 2.0, 4.0]
    decrease = [0.5, 0.5e2, 0.5e3]
    tolerance = [1e-4, 1e-6, 1e-8]
    incdec_results = sampler.run_inc_dec(increase, decrease, tolerance, rho_D,
        maximum, param_min, param_max, transition_kernel, sample_save_file)

..note:: The above examples just use a ``zip`` combination of the lists uses to
define varying parameters for the heuristics and transition kernels. To explore
the product of these lists you need to use ``numpy.meshgrid`` and
``numpy.ravel`` or a similar process.

To compare the results in terms of yield or the total number of samples
generated in the region of interest we can use
`~bet.sampling.basicSampling.compare_yield` to display the results to screen::

    # Compare the quality of several sets of samples
    print "Compare yield of sample sets with various heuristics"
    bsam.compare_yield(gen_results[3], gen_results[2], gen_results[4])
    print "Compare yield of sample sets with various transition kernels bounds"
    bsam.compare_yield(tk_results[3], tk_results[2], tk_results[4])
    print "Compare yield of sample sets with variouos increase/decrease ratios"
    bsam.compare_yield(incdec_results[3], incdec_results[2],incdec_results[4])

Here :meth:`~bet.sampling.basicSampling.compare_yield` simply displays to screen the
``sample_quality`` and ``run_param`` sorted by ``sample_quality`` and indexed
by ``sort_ind``. 

