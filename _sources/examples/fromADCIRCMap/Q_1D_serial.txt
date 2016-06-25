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
defined and the approximation of the volume of :math:`\mathcal{V}_j`. See
:download:`Q_1D_serial.py
<../../../examples/fromADCIRCMap/Q_1D_serial.py>` for the example source code.

First, import the necessary packages and modules::

    import bet.calculateP.calculateP as calcP
    import bet.calculateP.simpleFunP as sfun
    import numpy as np
    import scipy.io as sio

Load the data where our parameter space is 2-dimensional::

    mdat = sio.loadmat('Q_2D')
    Q = mdat['Q']
    
Load a reference solution::

    Q_ref = mdat['Q_true']
    samples = mdat['points'].transpose()

We will to the ``samples`` above as :math:`\lambda_{samples} = \{ \lambda^{(j)
} \}, j = 1, \ldots, N`. These ``samples`` are the points in parameter space
where we solve the forward model to generate the data ``Q`` where :math:`Q_j =
Q(\lambda^{(j)})`.

Define the parameter domain :math:`\Lambda`::

    lam_domain = np.array([[0.07, .15], [0.1, 0.2]])
    
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

Define the data space :math:`\mathcal{D} \subset \mathbb{R}^d` where :math:`d` is the dimension of the data space::

        data = Q[:, station_nums]
    
Define the refernce solution. We define a region of interest, :math:`R_{ref} \subset \mathcal{D}` centered at
:math:`Q_{ref}` that is 15% the length of :math:`q_n` (the QoI for station :math:`n`). We set :math:`\rho_\mathcal{D}(q) = \frac{\mathbf{1}_{R_{ref}}(q)}{||\mathbf{1}_{R_{ref}}||}` and then create a simple function approximation to this density::

        q_ref = Q_ref[ref_num, station_nums]
        # Create Simple function approximation
        # Save points used to parition D for simple function approximation and the
        # approximation itself (this can be used to make close comparisions...)
        (rho_D_M, d_distr_samples, d_Tree) = sfun.uniform_hyperrectangle(data,
                q_ref, bin_ratio=0.15,
                center_pts_per_edge=np.ones((data.shape[1],)))

We generate 1e6 uniformly distributed points in :math:`\Lambda`. We call these points :math:`\lambda_{emulate} = \{ \lambda_j \}_{j=1}^{10^6}`::

        num_l_emulate = 1e6
        lambda_emulate = calcP.emulate_iid_lebesgue(lam_domain, num_l_emulate)

Set up a dictonary and store various values to be used in calculating :math:`\hat{\rho}_{\Lambda, j}`::

        mdict = dict()
        mdict['rho_D_M'] = rho_D_M
        mdict['d_distr_samples'] = d_distr_samples 
        mdict['num_l_emulate'] = num_l_emulate
        mdict['lambda_emulate'] = lambda_emulate

Calculate :math:`\hat{\rho}_{\Lambda, j}` where :math:`\mathcal{V}_j` are the
voronoi cells defined by :math:`\lambda_{emulate}`::

        (P0, lem0, io_ptr0, emulate_ptr0) = calcP.prob_emulated(samples, data,
                rho_D_M, d_distr_samples, lambda_emulate, d_Tree)
        mdict['P0'] = P0
        mdict['lem0'] = lem0
        mdict['io_ptr0'] = io_ptr0
        mdict['emulate_ptr0'] = emulate_ptr0

Calculate :math:`\hat{\rho}_{\Lambda, j}` where :math:`\mathcal{V}_j` are the
voronoi cells defined by :math:`\lambda_{samples}` assume that :math:`\lambda_{samples}`
are uniformly distributed and therefore have approximately the same volume::

        (P1, lam_vol1, lem1, io_ptr1) = calcP.prob(samples, data,
                rho_D_M, d_distr_samples, d_Tree)
        mdict['P1'] = P1
        mdict['lam_vol1'] = lam_vol1
        mdict['lem1'] = lem1
        mdict['io_ptr1'] = io_ptr1

Calculate :math:`\hat{\rho}_{\Lambda, j}` where :math:`\mathcal{V}_j` are the
voronoi cells defined by :math:`\lambda_{samples}` and we approximate the volume of
:math:`\mathcal{V}_j` using Monte Carlo integration. We use
:math:`\lambda_{emulate}` to estimate the volume of :math:`\mathcal{V}_j` ::

        (P3, lam_vol3, lambda_emulate3, io_ptr3, emulate_ptr3) = calcP.prob_mc(samples,
                data, rho_D_M, d_distr_samples, lambda_emulate, d_Tree)
        mdict['P3'] = P3
        mdict['lam_vol3'] = lam_vol3
        mdict['io_ptr3'] = io_ptr3
        mdict['emulate_ptr3'] = emulate_ptr3

Save the various estimates to a MATLAB formatted file which can be later used
to visualize estimates of :math:`\rho_\Lambda`::

        sio.savemat(filename, mdict, do_compression=True)

Finally, we calculate :math:`\hat{\rho}_{\Lambda, j}` for three reference solutions and 3 QoI::

    ref_nums = [6, 11, 15] # 7, 12, 16
    stations = [1, 4, 5] # 2, 5, 6

    ref_nums, stations = np.meshgrid(ref_nums, stations)
    ref_nums = ref_nums.ravel()
    stations = stations.ravel()

    for tnum, stat in zip(ref_nums, stations):
        postprocess([0], tnum)
