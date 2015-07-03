.. _contaminantTransport:


============================================================================
Example: Concentration of Contaminant in Wells Based on Transport Parameters
============================================================================

We will walk through the following `example
<https://github.com/UT-CHG/BET/tree/master/examples/contaminantTransport>`. 
This example takes uniformly distributed samples of parameters and
output data from a simple groundwater contaminant transport model,
and calculates solutions to the stochastic inverse problem.
The parameter domain is 5D, where the uncertain parameters are the x and y 
locations of the source, the horizontal dispersivity, the flow angle,
and the contaminant flux. There are 11 measured QoIs in the data space 
available. By changing the choice of QoIs that we use to solve the stochastic
inverse problem, we see the effect of geometric distinctness. 
Probabilities in the parameter space are 
calculated using the MC assumption.  1D and 2D marginals are calculated,
smoothed, and plotted.

Import the necessary modules::

    import numpy as np
    import bet.calculateP as calculateP
    import bet.postProcess as postProcess
    import bet.calculateP.simpleFunP as simpleFunP
    import bet.calculateP.calculateP as calculateP
    import bet.postProcess.plotP as plotP
    import bet.postProcess.plotDomains as plotD


Import the samples, data, reference data, and reference parameters::

  lam_domain = np.loadtxt("files/lam_domain.txt.gz") #parameter domain
  ref_lam = np.loadtxt("files/lam_ref.txt.gz") #reference parameter set
  Q_ref = np.loadtxt("files/Q_ref.txt.gz") #reference QoI set
  samples = np.loadtxt("files/samples.txt.gz") # uniform samples in parameter domain
  data = np.loadtxt("files/data.txt.gz") # data from model

Choose which subset of the 11 QoIs to use for inversion::

  QoI_indices=[7,8,9,10] # Indices for output data with which you want to invert

Choose the bin ratio for the uniform output probability::

  bin_ratio = 0.25 #ratio of length of data region to invert

Slice data and form approximate PDF for output for plotting::
  data = data[:,QoI_indices]
  Q_ref=Q_ref[QoI_indices]

  dmax = data.max(axis=0)
  dmin = data.min(axis=0)
  dscale = bin_ratio*(dmax-dmin)
  Qmax = Q_ref + 0.5*dscale
  Qmin = Q_ref -0.5*dscale
  def rho_D(x):
     return np.all(np.logical_and(np.greater(x,Qmin), np.less(x,Qmax)),axis=1)

Plot 2D projections of the data domain::

  plotD.show_data(data, Q_ref = Q_ref, rho_D=rho_D, showdim=2)

Suggested changes for user (1)
------------------------------

Change the ``QoI_indices`` and note how it changes the plots of the data
domain. The white points are ones with zero probability and the dark points
are those with nonzero probability. 


Suggested changes for user (2)
------------------------------

Try different ways of discretizing the probability measure on
:math:`\mathcal{D}` defined as a uniform probability measure on a rectangle
(since :math:`\mathcal{D}` is 2-dimensional).
    
*   unif_unif creates a uniform measure on a hyperbox with dimensions relative   to the size of the circumscribed hyperbox of the set :math:`\mathcal{D}`  using the bin_ratio. A total of M samples are drawn within a slightly larger  scaled hyperbox to discretize this measure defining M total generalized  contour events in Lambda.  The reason a slightly larger scaled hyperbox is  used to draw the samples to discretize :math:`\mathcal{D}` is because  otherwise every generalized contour event will have non-zero probability  which obviously defeats the purpose of "localizing" the probability within a  subset of :math:`\mathcal{D}`.
    
*   uniform_hyperrectangle uses the same measure defined in the same way as  unif_unif, but the difference is in the discretization which is on a regular  grid defined by ``center_pts_per_edge``.  If ``center_pts_per_edge = 1``,  then the contour event corresponding to the entire support of  :math:`\rho_\mathcal{D}` is approximated as a single event. This is done by  carefully placing a regular 3x3 grid (since :math:`dim(\mathcal{D})=2` in this  case) of points in :math:`\mathcal{D}` with the center point of the grid in  the center of the support of the measure and the other points placed outside  of the rectangle defining the support to define a total of 9 contour events  with 8 of them having exactly zero probability.

Create a simple function approximation of the probablity measure on
:math:`\mathcal{D}`::

    deterministic_discretize_D = True

    if deterministic_discretize_D == True:
      (d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle(data=data,
                                                  Q_ref=Q_ref, bin_ratio=0.2, center_pts_per_edge = 1)
    else:
      (d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.unif_unif(data=data,
                                                  Q_ref=Q_ref, M=50, bin_ratio=0.2, num_d_emulate=1E5)

  
Calculate probablities using the MC assumption::

  (P,  lam_vol, io_ptr) = calculateP.prob(samples=samples,
                                        data=data,
                                        rho_D_M=d_distr_prob,
                                        d_distr_samples=d_distr_samples)

                                                                                                                                                  
Calculate 2D marginal probs  - Suggested changes for user (3)
-------------------------------------------------------------
    
At this point, the only thing that should change in the plotP.* inputs
should be either the nbins values or sigma (which influences the kernel
density estimation with smaller values implying a density estimate that
looks more like a histogram and larger values smoothing out the values
more).
    
There are ways to determine "optimal" smoothing parameters (e.g., see CV, GCV,
and other similar methods), but we have not incorporated these into the code
as lower-dimensional marginal plots have limited value in understanding the
structure of a high dimensional non-parametric probability measure.

Plot the marginal probabilities::

    (bins, marginals2D) = plotP.calculate_2D_marginal_probs(P_samples = P, samples = samples, lam_domain = lam_domain, nbins = [10, 10, 10])

Smooth 2d marginals probs (optional)::

    marginals2D = plotP.smooth_marginals_2D(marginals2D,bins, sigma=1.0)

Plot 2d marginals probs::

    plotP.plot_2D_marginal_probs(marginals2D, bins, lam_domain, filename = "contaminant_map", interactive=False, lam_ref=ref_lam, lambda_labels=labels)

Calculate 1d marginal probs::

    (bins, marginals1D) = plotP.calculate_1D_marginal_probs(P_samples = P, samples = samples, lam_domain = lam_domain, nbins = [10, 10, 10])

Smooth 1d marginal probs (optional)::

    marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=1.0)

Plot 1d marginal probs::

    plotP.plot_1D_marginal_probs(marginals1D, bins, lam_domain, filename = "contaminant_map", interactive=False, lam_ref=ref_lam, lambda_labels=labels)

Sort samples by highest probability density and take highest x percent::

  (num_samples, P_high, samples_high, lam_vol_high, data_high)= postTools.sample_highest_prob(top_percentile=percentile, P_samples=P, samples=samples, lam_vol=lam_vol,data = data,sort=True)

Print the number of these samples  and the ratio of the volume they take up::

  print (numsamples, np.sum(lam_vol_high)


Suggested changes for user (4):
-------------------------------
Notice how the marginal probabilites change with different choices of  ``QoI_indices``.
Try choosing only 2 or 3, instead of 4, indices and notice the higher-dimensionality of the structure in the 2d marginals. Notice how some QoI concentrate the probability into smaller regions. These QoI are more geometrically distinct. 

Notice that the volume that the high-probability samples take up is smaller with more geometrically distinct QoIs.

Suggested changes for user (5):
-------------------------------
Change ``percentile`` to values between 1.0 and 0.0. Notice that while the region of nonzero probabibilty may have a significant volume, much of this volume contains relatively low probability. Change the value to 0.95, 0.9, 0.75, and 0.5 and notice the volume decrease significantly. 

