.. _q2D:

==============================================================================
Example: Estimate the parameter space probabiliy density with  a 2D data space
==============================================================================


In this example the parameter space :math:`\Lambda \subset \mathbb{R}^2` is 2
dimensional.
This example demostrates three different methods to estimate
:math:`\hat{\rho}_{\Lambda, j}` where 

.. math::

    P_\Lambda \approx \sum_{\mathcal{V}_j \subset A} \hat{\rho}_{\Lambda, j}.

These methods are distinguished primarily by the way :math:`\mathcal{V}_j` are
defined and the approximation of the volume of :math:`\mathcal{V}_j`. See
`Q_2D.py
<https://github.com/UT-CHG/BET/blob/master/examples/fromFile_ADCIRCMap/Q_2D.py>`_ for the example source code. Since
this example is essentially the same as :ref:`q1D` we will only highlight the
differences between the two.


.. note::

    * reduce number of ``ref_nums`` to a single number so that we only
        examine a single set of QoI
    * add plotting of marginals using
        :mod:`bet.postProcess.plotP.plot_2D_marginal_probs`

Define the filename to save :math:`\hat{\rho}_{\Lambda, j}` to::

        filename = 'P_q'+str(station_nums[0]+1)+'_q'+str(station_nums[1]+1)
        if len(station_nums) == 3:
            filename += '_q'+str(station_nums[2]+1)
        filename += '_truth_'+str(ref_num+1)

Define the input sample set, here it is 2D rather than 1D::

    lam_domain = np.array([[0.07, .15], [0.1, 0.2]])
    input_sample_set = sample.sample_set(points.shape[0])
    input_sample_set.set_values(points.transpose())
    input_sample_set.set_domain(lam_domain)


Define the data space :math:`\mathcal{D} \subset \mathbb{R}^d` where :math:`d` is the dimension of the data space::

        data = Q[:, station_nums]
        output_sample_set = sample.sample_set(data.shape[1])
        output_sample_set.set_values(data)
    
Define the refernce solution. We define a region of interest, :math:`R_{ref} \subset \mathcal{D}` centered at
:math:`Q_{ref}`  with sides 15% the length of :math:`q_{station\_num[0]}` and
:math:`q_{station\_num[1]}` (the QoI for stations :math:`n`). We set :math:`\rho_\mathcal{D}(q) = \frac{\mathbf{1}_{R_{ref}}(q)}{||\mathbf{1}_{R_{ref}}||}` and then create a simple function approximation to this density::

        q_ref = Q_ref[ref_num, station_nums]
        output_probability_set = sfun.regular_partition_uniform_distribution_rectangle_scaled(\
                output_sample_set, q_ref, rect_scale=0.15,
                cells_per_dimension=np.ones((data.shape[1],)))


Finally, we calculate :math:`\hat{\rho}_{\Lambda, j}` for three reference solutions and the QoI :math:`( (q_1,q_2), (q_1, q_5)`, and :math:`(q_1, q_6))` ::

    ref_nums = [6, 11, 15] # 7, 12, 16
    stations = [1, 4, 5] # 2, 5, 6

    ref_nums, stations = np.meshgrid(ref_nums, stations)
    ref_nums = ref_nums.ravel()
    stations = stations.ravel()

    for tnum, stat in zip(ref_nums, stations):
        postprocess([0, stat], tnum)

