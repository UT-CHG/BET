.. _adaptive2D:

=======================================================================
Example: Generalized Chains with a 2,2-dimensional data,parameter space
=======================================================================

This example demonstrates the adaptive generation of samples using a
goal-oriented adaptive sampleing algorithm.

Generating a single set of adaptive samples
-------------------------------------------

We will walk through the following :download:`example
<../../../examples/fromFileMap/fromFile2D.py>` that uses a linear interpolant of
the QoI map :math:`Q(\lambda) = (q_1(\lambda), q_6(\lambda))` for a
2-dimensional data space. The parameter space in this example is also
2-dimensional. 

The modules required by this example are::

    import polyadcirc.run_framework.domain as dom
    import polyadcirc.run_framework.random_wall_Q as rmw
    import numpy as np
    import polyadcirc.pyADCIRC.basic as basic
    import bet.sampling.adaptiveSampling as asam
    import bet.sampling.basicSampling as bsam
    import scipy.io as sio

The next big section of code uses
:class:`~polyadcirc.run_framework.randoma_wall_Q` to create the "model" that
:class:`~bet.sampling.adaptiveSampling.sampler`
interrogates for data::

    adcirc_dir = '/work/01837/lcgraham/v50_subdomain/work'
    grid_dir = adcirc_dir + '/ADCIRC_landuse/Inlet_b2/inputs/poly_walls'
    save_dir = adcirc_dir + '/ADCIRC_landuse/Inlet_b2/runs/adaptive_random_2D'
    basis_dir = adcirc_dir +'/ADCIRC_landuse/Inlet_b2/gap/beach_walls_2lands'
    # assume that in.prep* files are one directory up from basis_dir
    script = "adaptive_random_2D.sh"
    # set up saving
    model_save_file = 'py_save_file'
    sample_save_file = 'full_run'

    # Select file s to record/save
    timeseries_files = []#["fort.63"]
    nontimeseries_files = ["maxele.63"]#, "timemax63"]

    # NoNx12/TpN where NoN is number of nodes and TpN is tasks per node, 12 is
    the
    # number of cores per node See -pe line in submission_script <TpN>way<NoN x
    # 12>
    nprocs = 4 # number of processors per PADCIRC run
    ppnode = 16
    NoN = 20
    TpN = 16 # must be 16 unless using N option
    num_of_parallel_runs = (TpN*NoN)/nprocs

    domain = dom.domain(grid_dir)
    domain.update()
    main_run = rmw.runSet(grid_dir, save_dir, basis_dir, num_of_parallel_runs,
            base_dir=adcirc_dir, script_name=script)
    main_run.initialize_random_field_directories(num_procs=nprocs)

The compact (bounded, finite-dimensional) paramter space is::

    # Set minima and maxima
    lam_domain = np.array([[.07, .15], [.1, .2]])
    lam3 = 0.012
    ymin = -1050
    xmin = 1420
    xmax = 1580
    ymax = 1500
    wall_height = -2.5

    param_min = lam_domain[:, 0]
    param_max = lam_domain[:, 1]

Specify the observation stations to use to record data to be used as QoI::

    # Create stations
    stat_x = np.concatenate((1900*np.ones((7,)), [1200], 1300*np.ones((3,)),
	[1500])) 
    stat_y = np.array([1200, 600, 300, 0, -300, -600, -1200, 0, 1200,
	    0, -1200, -1400])
    all_stations = []
    for x, y in zip(stat_x, stat_y):
	all_stations.append(basic.location(x, y))

    # Select only the stations I care about this will lead to better sampling
    station_nums = [0, 5] # 1, 6
    stations = []
    for s in station_nums:
	stations.append(all_stations[s])

    # Read in Q_ref and Q to create the appropriate rho_D 
    mdat = sio.loadmat('Q_2D')
    Q = mdat['Q']
    Q = Q[:, station_nums]

In this example we use :class:`~polyadcirc.run_framework.random_wall_Q` for the
QoI map :math:`Q(\lambda) = (q_1(\lambda), q_6(\lambda))` ::

    # Create experiment model
    def model(sample):
	# box_limits [xmin, xmax, ymin, ymax, wall_height]
	wall_points = np.outer([xmin, xmax, ymin, ymax, wall_height],
		np.ones(sample.shape[1]))
	# [lam1, lam2, lam3]
	mann_pts = np.vstack((sample, lam3*np.ones(sample.shape[1])))
	return main_run.run_nobatch_q(domain, wall_points, mann_pts,
		model_save_file, num_procs=nprocs, procs_pnode=ppnode,
		stations=stations, TpN=TpN)

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
:download:`sandbox_test_2D.py <../../../examples/fromFileMap/sandbox_test_2D.py>`. The set up in
:download:`sandbox_test_2D.py <../../../examples/fromFileMap/sandbox_test_2D.py>` is very similar to the
set up in :download:`fromFile2D <../../../examples/fromFileMap/fromFile2D.py>` and is
omitted for brevity.

We can explore several types of kernels::

    kernel_mm = asam.maxima_mean_kernel(np.array([Q_ref]), rho_D)
    kernel_rD = asam.rhoD_kernel(maximum, rho_D)
    kernel_m = asam.maxima_kernel(np.array([Q_ref]), rho_D)
    kernel_md = asam.multi_dist_kernel()
    kern_list = [kernel_mm, kernel_rD, kernel_m, kernel_md]
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

