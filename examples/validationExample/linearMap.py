#! /usr/bin/env python

# Copyright (C) 2014-2015 Lindley Graham and Steven Mattis

"""
This 2D linear example verifies that geometrically distinct QoI can
recreate a probability measure on the input parameter space
used to define the output probability measure. 
"""

import numpy as np
import bet.calculateP as calculateP
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import scipy.spatial as spatial

# parameter domain
lam_domain= np.array([[0.0, 1.0],
                      [0.0, 1.0]])

'''
Suggested changes for user:
    
Try setting n0 and n1 all to 10 and compare the results.
    
Also, we can do uniform random sampling by setting 

  random_sample = True
  
If random_sample = True, consider defining
   
  n_samples = 2.5E3
        
Then also try n_samples = 1E4. What happens when n_samples = 1E2?
'''
random_sample = False

if random_sample == False:
  n0 = 50 # number of samples in lam0 direction
  n1 = 50 # number of samples in lam1 direction
  n_samples = n0*n1
else:
  n_samples = 2.5E3  

# QoI map
Q_map = np.array([[0.506, 0.463],[0.253, 0.918]])

# reference QoI
Q_ref =  np.array([0.3795, 0.6905])

# reference parameters
ref_lam = [0.5, 0.5]

#set up samples
if random_sample == False:
  vec0=list(np.linspace(lam_domain[0][0], lam_domain[0][1], n0))
  vec1 = list(np.linspace(lam_domain[1][0], lam_domain[1][1], n1))
  vecv0, vecv1 = np.meshgrid(vec0, vec1, indexing='ij')
  samples=np.vstack((vecv0.flat[:], vecv1.flat[:])).transpose()
else:
  samples = calculateP.emulate_iid_lebesgue(lam_domain=lam_domain, 
					    num_l_emulate = n_samples)

# calc data
data = np.dot(samples,Q_map)

'''
Compute the output distribution simple function approximation by
propagating a different set of samples to implicitly define a Voronoi
discretization of the data space, and then propagating i.i.d. uniform
samples to bin into these cells.

Suggested changes for user:

See the effect of using different values for d_distr_samples_num.
Choosing 

  d_distr_samples_num = 1
  
produces exactly the right answer and is equivalent to assigning a
uniform probability to each data sample above (why?). 

Try setting this to 2, 5, 10, 50, and 100. Can you explain what you 
are seeing? To see an exaggerated effect, try using random sampling
above with n_samples set to 1E2. 
'''
d_distr_samples_num = 1

samples_discretize = calculateP.emulate_iid_lebesgue(lam_domain=lam_domain, 
					    num_l_emulate = d_distr_samples_num)

d_distr_samples = np.dot(samples_discretize, Q_map)

d_Tree = spatial.KDTree(d_distr_samples)

samples_distr_prob_num = d_distr_samples_num*1E3

samples_distr_prob = calculateP.emulate_iid_lebesgue(lam_domain=lam_domain, 
					    num_l_emulate = samples_distr_prob_num)

data_prob = np.dot(samples_distr_prob, Q_map)

# Determine which data samples go to which d_distr_samples_num bins using the QoI
(_, oo_ptr) = d_Tree.query(data_prob)

# Calculate Probabilities of the d_distr_samples defined Voronoi cells
d_distr_prob = np.zeros((d_distr_samples_num,))
for i in range(d_distr_samples_num):
  Itemp = np.equal(oo_ptr, i)
  Itemp_sum = float(np.sum(Itemp)) 
  d_distr_prob[i] = Itemp_sum / samples_distr_prob_num

'''
Suggested changes for user:
    
If using a regular grid of sampling (if random_sample = False), we set
    
  lambda_emulate = samples
  
Otherwise, play around with num_l_emulate. A value of 1E2 will probably
give poor results while results become fairly consistent with values 
that are approximately 10x the number of samples.
   
Note that you can always use
    
  lambda_emulate = samples
        
and this simply will imply that a standard Monte Carlo assumption is
being used, which in a measure-theoretic context implies that each 
Voronoi cell is assumed to have the same measure. This type of 
approximation is more reasonable for large n_samples due to the slow 
convergence rate of Monte Carlo (it converges like 1/sqrt(n_samples)).
'''
if random_sample == False:
  lambda_emulate = samples
else:
  lambda_emulate = calculateP.emulate_iid_lebesgue(lam_domain=lam_domain, num_l_emulate = 1E5)


# calculate probablities
(P,  lambda_emulate, io_ptr, emulate_ptr) = calculateP.prob_emulated(samples=samples,
                                                                     data=data,
                                                                     rho_D_M=d_distr_prob,
                                                                     d_distr_samples=d_distr_samples,
                                                                     lambda_emulate=lambda_emulate,
                                                                     d_Tree=d_Tree)
# calculate 2d marginal probs
'''
Suggested changes for user:
    
At this point, the only thing that should change in the plotP.* inputs
should be either the nbins values or sigma (which influences the kernel
density estimation with smaller values implying a density estimate that
looks more like a histogram and larger values smoothing out the values
more).
    
There are ways to determine "optimal" smoothing parameters (e.g., see CV, GCV,
and other similar methods), but we have not incorporated these into the code
as lower-dimensional marginal plots have limited value in understanding the
structure of a high dimensional non-parametric probability measure.
'''
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(P_samples = P, samples = lambda_emulate, lam_domain = lam_domain, nbins = [10, 10])
# smooth 2d marginals probs (optional)
#marginals2D = plotP.smooth_marginals_2D(marginals2D,bins, sigma=0.01)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, lam_domain, filename = "linearMapValidation",
                             plot_surface=False)

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(P_samples = P, samples = lambda_emulate, lam_domain = lam_domain, nbins = [10, 10])
# smooth 1d marginal probs (optional)
#marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.01)
# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, lam_domain, filename = "linearMapValidation")





