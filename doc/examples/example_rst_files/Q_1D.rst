.. _q1D:

==============================================================================
Example: Estimate the parameter space probabiliy density with  a 1D data space
==============================================================================

In this example the parameter space :math:`\Lambda \subset \mathbb{R}^2` is 2
dimensional.
This example demostrates three different methods to estimate
:math:`\hat{\rho}_{\Lambda, j}` where 

.. math::

    P_\Lambda \approx \sum_{\mathcal{V}_j \subset A} \hat{\rho}_{\Lambda, j}.

These methods are distinguished primarily by the way :math:`\mathcal{V}_j` are
defined and the approximation of the volume of :math:`\mathcal{V}_j`. See `Q_1D.py
<https://github.com/UT-CHG/BET/blob/master/examples/fromFile_ADCIRCMap/Q_1D.py>`_ for the example source code.

First, import the necessary packages and modules::

    import bet.sampling.basicSampling as bsam
    import bet.calculateP.calculateP as calcP
    import bet.calculateP.simpleFunP as sfun
    import numpy as np
    import scipy.io as sio
    import bet.sample as sample

Load the data where our parameter space is 2-dimensional::

    points = mdat['points']
    
Load a reference solution::

    mdat = sio.loadmat('../matfiles/Q_2D')
    Q = mdat['Q']
    Q_ref = mdat['Q_true']

We will use the to the ``points``, :math:`\lambda_{samples} = \{ \lambda^{(j)
} \}, j = 1, \ldots, N`, to create an input :class:`~bet.sample.sample_set`
object. These ``points`` are the points in parameter space where we solve the
forward model to generate the data ``Q`` where :math:`Q_j =
Q(\lambda^{(j)})`.

Define the parameter domain :math:`\Lambda`::

    lam_domain = np.array([[0.07, .15], [0.1, 0.2]])

Create input sample set objects::

    input_sample_set = sample.sample_set(points.shape[0])
    input_sample_set.set_values(points.transpose())
    input_sample_set.set_domain(lam_domain)

    
For ease of use we have created a function, ``postprocess(station_nums,
ref_num)`` so that we can loop through different QoI (maximum water surface
height at various measurement stations) and reference solutions (point in data
space around which we center a uniform probability solution. The function is
defined as follows::

    def postprocess(station_nums, ref_num):

Define the filename to save :math:`\hat{\rho}_{\Lambda, j}` to::

        filename = 'P_q'+str(station_nums[0]+1)+'_q'
        if len(station_nums) == 3:
            filename += '_q'+str(station_nums[2]+1)
        filename += '_truth_'+str(ref_num+1)

Define the data space :math:`\mathcal{D} \subset \mathbb{R}^d` where :math:`d`
is the dimension of the data space::

        data = Q[:, station_nums]
        output_sample_set = sample.sample_set(data.shape[1])
        output_sample_set.set_values(data)
    
Define the refernce solution. We define a region of interest, :math:`R_{ref}
\subset \mathcal{D}` centered at :math:`Q_{ref}` that is 15% the length of
:math:`q_n` (the QoI for station :math:`n`). We set :math:`\rho_\mathcal{D}(q)
= \frac{\mathbf{1}_{R_{ref}}(q)}{||\mathbf{1}_{R_{ref}}||}` and then create a
simple function approximation to this density::

        q_ref = Q_ref[ref_num, station_nums]
        output_probability_set = sfun.regular_partition_uniform_distribution_rectangle_scaled(\
                output_sample_set, q_ref, rect_scale=0.15,
                center_pts_per_edge=np.ones((data.shape[1],)))

We generate 1e6 uniformly distributed points in :math:`\Lambda`. We call these points :math:`\lambda_{emulate} = \{ \lambda_j \}_{j=1}^{10^6}`::

        num_l_emulate = 1e4
        set_emulated = bsam.random_sample_set('r', lam_domain, num_l_emulate)
        my_disc = sample.discretization(input_sample_set, output_sample_set,
                output_probability_set, emulated_input_sample_set=set_emulated)

Calculate :math:`\hat{\rho}_{\Lambda, j}` where :math:`\mathcal{V}_j` are the
voronoi cells defined by :math:`\lambda_{emulate}`::

        calcP.prob_on_emulated_samples(my_disc)
        sample.save_discretization(my_disc, filename, "prob_on_emulated_samples_solution")

Calculate :math:`\hat{\rho}_{\Lambda, j}` where :math:`\mathcal{V}_j` are the
voronoi cells defined by :math:`\lambda_{samples}` assume that :math:`\lambda_{samples}`
are uniformly distributed and therefore have approximately the same volume::

        input_sample_set.estimate_volume_mc()
        calcP.prob(my_disc)
        sample.save_discretization(my_disc, filename, "prob_solution")

Calculate :math:`\hat{\rho}_{\Lambda, j}` where :math:`\mathcal{V}_j` are the
voronoi cells defined by :math:`\lambda_{samples}` and we approximate the volume of
:math:`\mathcal{V}_j` using Monte Carlo integration. We use
:math:`\lambda_{emulate}` to estimate the volume of :math:`\mathcal{V}_j` ::

        calcP.prob_with_emulated_volumes(my_disc)
        sample.save_discretization(my_disc, filename, "prob_with_emulated_volumes_solution")

Finally, we calculate :math:`\hat{\rho}_{\Lambda, j}` for three reference solutions and 3 QoI::

    ref_nums = [6, 11, 15] # 7, 12, 16
    stations = [1, 4, 5] # 2, 5, 6

    ref_nums, stations = np.meshgrid(ref_nums, stations)
    ref_nums = ref_nums.ravel()
    stations = stations.ravel()

    for tnum, stat in zip(ref_nums, stations):
        postprocess([0], tnum)
