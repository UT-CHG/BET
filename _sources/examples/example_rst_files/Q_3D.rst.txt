.. _q3D:

==============================================================================
Example: Estimate the parameter space probabiliy density with  a 3D data space
==============================================================================

In these examples the parameter space :math:`\Lambda \subset \mathbb{R}^3` is 3
dimensional.


This example demostrates how to estimate :math:`\hat{\rho}_{\Lambda, j}` using
:meth:`~bet.calculateP.calculateP.prob` where 

.. math::

    P_\Lambda \approx \sum_{\mathcal{V}_j \subset A} \hat{\rho}_{\Lambda, j}.

See `Q_3D.py <https://github.com/UT-CHG/BET/blob/master/examples/fromFile_ADCIRCMap/Q_3D.py>`_
for the example source code. Since example is essentially the same as
:ref:`q2D` we will only highlight the differences between the two.

Instead of loading data for a 2-dimensional parameter space we load data for a
3-dimensional data space::

    mdat = sio.loadmat('Q_3D')

We define the parameter domain :math:`\Lambda`::

    lam_domain = np.array([[-900, 1200], [0.07, .15], [0.1, 0.2]])


Define the input sample set, here it is 3D rather than 2D::

    input_sample_set = sample.sample_set(points.shape[0])
    input_sample_set.set_values(points.transpose())
    input_sample_set.set_domain(lam_domain)


Define the filename to save :math:`\hat{\rho}_{\Lambda, j}` to::

        filename = 'P_q'+str(station_nums[0]+1)+'_q'+str(station_nums[1]+1)
        if len(station_nums) == 3:
            filename += '_q'+str(station_nums[2]+1)
        filename += '_ref_'+str(ref_num+1)



Example solutions
~~~~~~~~~~~~~~~~~~
Finally, we calculate :math:`\hat{\rho}_{\Lambda, j}` for the 15th reference
solution at :math:`Q = (q_1, q_5, q_2), (q_1, q_5), (q_1, q_5, q_{12}), (q_1,
q_9, q_7),` and :math:`(q_1, q_9, q_{12})`::

    ref_num = 14

    station_nums = [0, 4, 1] # 1, 5, 2
    postprocess(station_nums, ref_num)

    station_nums = [0, 4] # 1, 5
    postprocess(station_nums, ref_num)

    station_nums = [0, 4, 11] # 1, 5, 12
    postprocess(station_nums, ref_num)

    station_nums = [0, 8, 6] # 1, 9, 7
    postprocess(station_nums, ref_num)

    station_nums = [0, 8, 11] # 1, 9, 12
    postprocess(station_nums, ref_num)


