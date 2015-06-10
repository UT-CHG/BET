.. _q3D:

==============================================================================
Example: Estimate the parameter space probabiliy density with  a 3D data space
==============================================================================

In these examples the parameter space :math:`\Lambda \subset \mathbb{R}^3` is 3
dimensional.

Serial Example
~~~~~~~~~~~~~~
This example demostrates how to estimate :math:`\hat{\rho}_{\Lambda, j}` using
:meth:`~bet.calculateP.calculateP.prob` where 

.. math::

    P_\Lambda \approx \sum_{\mathcal{V}_j \subset A} \hat{\rho}_{\Lambda, j}.

See :download:`Q_3D_serial.py <../../../examples/fromFile_ADCIRCMap/Q_3D_serial.py>`
for the example source code. Since example is essentially the same as
:ref:`q2D` we will only highlight the differences between the two.

Instead of loading data for a 2-dimensional parameter space we load data for a
3-dimensional data space::

    mdat = sio.loadmat('Q_3D')

We define the parameter domain :math:`\Lambda`::

    lam_domain = np.array([[-900, 1200], [0.07, .15], [0.1, 0.2]])

Also the ``postprocess(station_nums, ref_num)`` function in this case only uses 
:meth:`~bet.calculateP.calculateP.prob`. 

Parallel Example
~~~~~~~~~~~~~~~~

.. note:: The parallel version of this example has been moved to the development branch. 

This example demostrates how to estimate :math:`\hat{\rho}_{\Lambda, j}` using
:meth:`~bet.calculateP.calculateP.prob_mc` where 

.. math::

    P_\Lambda \approx \sum_{\mathcal{V}_j \subset A} \hat{\rho}_{\Lambda, j}.

See :download:`Q_3D_parallel.py <../../../examples/fromFile_ADCIRCMap/Q_3D_serial.py>`
for the example source code. Since example is essentially the same as
:ref:`q2D` we will only highlight the differences between the two.

This is example takes advantage of the embarrisingly parallel nature of the
algorithm(s) used to estimate :math:`\hat{\rho}_{\Lambda, j}` and the volumes
of :math:`\mathcal{V}_j`. To do this we need to import two additional modules::

    from bet import util
    import scipy.io as sio
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

Instead of loading data for a 2-dimensional parameter space we load data for a
3-dimensional data space::

    mdat = sio.loadmat('Q_3D')

We define the parameter domain :math:`\Lambda`::

    lam_domain = np.array([[-900, 1200], [0.07, .15], [0.1, 0.2]])

Within the ``postprocess(station_nums, ref_num)`` function in this case we only use 
:meth:`~bet.calculateP.calculateP.prob_mc`. Since this script is parallel we
need to use :meth:`~bet.postProcess.plotP.get_global_values` to concatenate the arrays
spread out across the processors into a single array::

    mdict['num_l_emulate'] = mdict['lambda_emulate'].shape[1]
    mdict['P3'] = util.get_global_values(P3)
    mdict['lam_vol3'] = lam_vol3
    mdict['io_ptr3'] = io_ptr3
    
Furthermore, we only want to write out the solution using a single processor::

    if rank == 0:
        sio.savemat(filename, mdict, do_compression=True)

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


