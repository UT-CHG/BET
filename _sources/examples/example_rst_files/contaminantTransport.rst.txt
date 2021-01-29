.. _contaminantTransport:


============================================================================
Example: Concentration of Contaminant in Wells Based on Transport Parameters
============================================================================

We will walk through the following `example
<https://github.com/UT-CHG/BET/tree/master/examples/contaminantTransport>`_. 
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
smoothed, and plotted. The samples are then ranked by probability density
and the volume of high-probability regions are calculated. Probabilistic predictions of other QoI are made.

Import the necessary modules::

  import numpy as np
  import bet.calculateP as calculateP
  import bet.postProcess as postProcess
  import bet.calculateP.simpleFunP as simpleFunP
  import bet.calculateP.calculateP as calculateP
  import bet.postProcess.plotP as plotP
  import bet.postProcess.plotDomains as plotD
  import bet.postProcess.postTools as postTools
  import bet.sample as samp


Import the samples, data, reference data, and reference parameters::

 parameter_domain = np.loadtxt("files/lam_domain.txt.gz") #parameter domain
 parameter_dim = parameter_domain.shape[0]
 # Create input sample set
 input_samples = samp.sample_set(parameter_dim)
 input_samples.set_domain(parameter_domain)
 input_samples.set_values(np.loadtxt("files/samples.txt.gz"))
 input_samples.estimate_volume_mc() # Use standard MC estimate of volumes

Choose which subset of the 11 QoIs to use for inversion::

 QoI_indices_observe = np.array([0,1,2,3])
 output_samples = samp.sample_set(QoI_indices_observe.size)
 output_samples.set_values(np.loadtxt("files/data.txt.gz")[:,QoI_indices_observe])

 Create discretization object::
   my_discretization = samp.discretization(input_sample_set=input_samples,
                                           output_sample_set=output_samples)

Choose the bin ratio for the uniform output probability::

  bin_ratio = 0.25 #ratio of length of data region to invert

Load the reference parameter and QoI values::

  param_ref = np.loadtxt("files/lam_ref.txt.gz") #reference parameter set
  Q_ref = np.loadtxt("files/Q_ref.txt.gz")[QoI_indices_observe] #reference QoI set

Plot 2D projections of the data domain::

  plotD.scatter_rhoD(my_discretization, ref_sample=Q_ref, io_flag='output', showdim=2)
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
    
*   regular_partition_uniform_distribution_rectangle_scaled creates a uniform measure on a hyperbox with dimensions relative   to the size of the circumscribed hyperbox of the set :math:`\mathcal{D}`  using the bin_ratio. A total of M samples are drawn within a slightly larger  scaled hyperbox to discretize this measure defining M total generalized  contour events in Lambda.  The reason a slightly larger scaled hyperbox is  used to draw the samples to discretize :math:`\mathcal{D}` is because  otherwise every generalized contour event will have non-zero probability  which obviously defeats the purpose of "localizing" the probability within a  subset of :math:`\mathcal{D}`.
    
*   uniform_partition_uniform_distribution_rectangle_scaled uses the same measure defined in the same way as  unif_unif, but the difference is in the discretization which is on a regular  grid defined by ``cells_per_dimension``.  If ``cells_per_dimension = 1``,  then the contour event corresponding to the entire support of  :math:`\rho_\mathcal{D}` is approximated as a single event. This is done by  carefully placing a regular 3x3 grid (since :math:`dim(\mathcal{D})=2` in this  case) of points in :math:`\mathcal{D}` with the center point of the grid in  the center of the support of the measure and the other points placed outside  of the rectangle defining the support to define a total of 9 contour events  with 8 of them having exactly zero probability.

Create a simple function approximation of the probablity measure on
:math:`\mathcal{D}`::

	deterministic_discretize_D = True
	if deterministic_discretize_D == True:
	    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(data_set=my_discretization,
                                                                     Q_ref=Q_ref,
                                                                     rect_scale=0.25,
                                                                     cells_per_dimension = 1)
	else:
	    simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled(data_set=my_discretization,
                                                                     Q_ref=Q_ref,
                                                                     rect_scale=0.25,
                                                                     M=50,
                                                                     num_d_emulate=1E5)

  
Calculate probablities using the MC assumption::

  calculateP.prob(my_discretization)


                                                                                                                                                  
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

  (bins, marginals2D) = plotP.calculate_2D_marginal_probs(my_discretization, nbins = 10)

Smooth 2d marginals probs (optional)::

    marginals2D = plotP.smooth_marginals_2D(marginals2D,bins, sigma=1.0)

Plot 2d marginals probs::

    plotP.plot_2D_marginal_probs(marginals2D, bins, my_discretization, filename = "contaminant_map",
                             plot_surface=False,
                             lam_ref = param_ref,
                             lambda_label=labels,
                             interactive=False)

Calculate 1d marginal probs::

    (bins, marginals1D) = plotP.calculate_1D_marginal_probs(my_discretization, nbins = 20)

Smooth 1d marginal probs (optional)::

  marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=1.0)

Plot 1d marginal probs::

    plotP.plot_1D_marginal_probs(marginals1D, bins, my_discretization,
                             filename = "contaminant_map",
                             interactive=False,
                             lam_ref=param_ref,
                             lambda_label=labels)

Sort samples by highest probability density and take highest x percent::

  (num_samples, my_discretization_highP, indices)= postTools.sample_highest_prob(
    percentile, my_discretization, sort=True)

Print the number of these samples  and the ratio of the volume they take up::

  print (num_samples, np.sum(my_discretization_highP._input_sample_set.get_volumes()))


Suggested changes for user (4):
-------------------------------
Notice how the marginal probabilites change with different choices of  ``QoI_indices``.
Try choosing only 2 or 3, instead of 4, indices and notice the higher-dimensionality of the structure in the 2d marginals. Notice how some QoI concentrate the probability into smaller regions. These QoI are more geometrically distinct. 

Notice that the volume that the high-probability samples take up is smaller with more geometrically distinct QoIs.

Suggested changes for user (5):
-------------------------------
Change ``percentile`` to values between 1.0 and 0.0. Notice that while the region of nonzero probabibilty may have a significant volume, much of this volume contains relatively low probability. Change the value to 0.95, 0.9, 0.75, and 0.5 and notice the volume decrease significantly. 



Propogate highest probability part of the probability measure through a different QoI map::

  QoI_indices_predict = np.array([7])
  output_samples_predict = samp.sample_set(QoI_indices_predict.size)
  output_samples_predict.set_values(np.loadtxt("files/data.txt.gz")[:,QoI_indices_predict])
  output_samples_predict.set_probabilities(input_samples.get_probabilities())
  
Calculate and plot PDF of predicted QoI::

  (bins_pred, marginals1D_pred) = plotP.calculate_1D_marginal_probs(output_samples_predict,
                                                                  nbins = 20)
   plotP.plot_1D_marginal_probs(marginals1D_pred, bins_pred, output_samples_predict,
                             filename = "contaminant_prediction", interactive=False)

Suggested changes for user (6):
-------------------------------
Change the prediction QoI map. Compare to the reference values.
