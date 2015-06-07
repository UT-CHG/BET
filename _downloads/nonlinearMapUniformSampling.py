#! /usr/bin/env python

# Copyright (C) 2014-2015 Lindley Graham and Steven Mattis

r"""
This example generates samples on a 2D grid and evaluates
a nonlinear map to a 1d or 2d space. The maps are defined
as quantities of interest (QoI) defined as spatial 
observations of the solution to the elliptic PDE .. math::
  :nowrap:
  
  \begin{cases}
    -\nabla \cdot (A(\lambda)\cdot\nabla u) &= f(x,y;\lambda), \ (x,y)\in\Omega, \\
    u|_{\partial \Omega} &= 0,
  \end{cases}

where :math:`A(\lambda)=\text{diag}(1/\lambda_1,1/\lambda_2)`,
:math: `f(x,y;\lambda) = \pi^2 \sin(\pi x\lambda_1)\sin(\pi y \lambda_2)`,
and :math:`\Omega=[0,1]\times[0,1]`.

Probabilities
in the paramter space are calculated using emulated points.
1D and 2D marginals are calculated, smoothed, and plotted.
"""

import numpy as np
import math as m
import bet.calculateP as calculateP
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP

# parameter domain
lam_domain= np.array([[3.0, 6.0],
                      [1.0, 5.0]])

'''
Suggested changes for user:
    
Try setting n0 and n1 both to 10 and compare the results.
    
Also, we can do uniform random sampling by setting 

  random_sample = True
  
If random_sample = True, consider defining
   
  n_samples = 1E3
        
Then also try n_samples = 1E4. What happens when n_samples = 1E2?
'''
random_sample = False

if random_sample == False:
  n0 = 50 # number of samples in lam0 direction
  n1 = 50 # number of samples in lam1 direction
  n_samples = n0*n1 
else:
  n_samples = 1E3  

# QoI function
def QoI(x,y,lam1,lam2):
  z = np.sin(m.pi*x*lam1)*np.sin(m.pi*y*lam2)
  return z #np.vstack(z.flat[:]).transpose()

  
# reference parameters
ref_lam = [5.5, 4.5]

'''
Suggested changes for user:

Try setting QoI_num = 2.  

Play around with the x1, y1, and/or, x2, y2 values to try and
"optimize" the QoI to give the highest probability region 
on the reference parameter above. 

Hint: Try using QoI_num == 1 and systematically varying the
x1 and y1 values to find QoI with contour structures (as inferred
through the 2D marginal plots) that are nearly orthogonal.

Some interesting pairs of QoI to compare are:
(x1,y1)=(0.5,0.5) and (x2,y2)=(0.25,0.25)
(x1,y1)=(0.5,0.5) and (x2,y2)=(0.15,0.15)
(x1,y1)=(0.5,0.5) and (x2,y2)=(0.25,0.15)
'''
# Choose the QoI and define Q_ref
QoI_num = 1

if QoI_num == 1:
  x1 = 0.5
  y1 = 0.5
  x = np.array([x1])
  y = np.array([y1])
  Q_ref = np.array([QoI(x[0],y[0],ref_lam[0],ref_lam[1])])
else:
  x1 = 0.5
  y1 = 0.5
  x2 = 0.25
  y2 = 0.25
  x = np.array([x1,x2])
  y = np.array([y1,y2])
  Q_ref = np.array([QoI(x[0],y[0],ref_lam[0],ref_lam[1]),
		      QoI(x[1],y[1],ref_lam[0],ref_lam[1])])	  

if QoI_num == 1:		      
  def QoI_map(x,y,lam1,lam2):
    Q1 = QoI(x[0],y[0],lam1,lam2)
    z = np.array([Q1]).transpose()
    return z
else:
  def QoI_map(x,y,lam1,lam2):
    Q1 = QoI(x[0],y[0],lam1,lam2)
    Q2 = QoI(x[1],y[1],lam1,lam2)
    z = np.array([Q1,Q2]).transpose()
    return z
  

#set up samples
if random_sample == False:
  vec0 = list(np.linspace(lam_domain[0][0], lam_domain[0][1], n0))
  vec1 = list(np.linspace(lam_domain[1][0], lam_domain[1][1], n1))
  vecv0, vecv1 = np.meshgrid(vec0, vec1, indexing='ij')
  samples = np.vstack((vecv0.flat[:], vecv1.flat[:])).transpose()
else:
  samples = calculateP.emulate_iid_lebesgue(lam_domain=lam_domain, 
					    num_l_emulate = n_samples)

# calc data
data = QoI_map(x,y,samples[:,0],samples[:,1])

'''
Suggested changes for user:
    
Try different ways of discretizing the probability measure on D defined
as a uniform probability measure on a rectangle (if QoI_num = 2) or on
an interval (if QoI_num = 1).
    
unif_unif creates a uniform measure on a hyperbox with dimensions 
relative to the size of the circumscribed hyperbox of the set D using 
the bin_ratio. A total of M samples are drawn within a slightly larger 
scaled hyperbox to discretize this measure defining M total generalized 
contour events in Lambda. The reason a slightly larger scaled hyperbox
is used to draw the samples to discretize D is because otherwise every 
generalized contour event will have non-zero probability which obviously 
defeats the purpose of "localizing" the probability within a subset of D.
    
uniform_hyperrectangle uses the same measure defined in the same way as 
unif_unif, but the difference is in the discretization which is on a 
regular grid defined by center_pts_per_edge. If center_pts_per_edge = 1, 
then the contour event corresponding to the entire support of rho_D is
approximated as a single event. This is done by carefully placing a 
regular 3x3 grid (for the D=2 case) of points in D with the center
point of the grid in the center of the support of the measure and the 
other points placed outside of the rectangle defining the support to 
define a total of 9 contour events with 8 of them with zero probability.
'''
deterministic_discretize_D = True

if deterministic_discretize_D == True:
  (d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle(data=data,
                                              Q_ref=Q_ref, bin_ratio=0.2, center_pts_per_edge = 1)
else:
  (d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.unif_unif(data=data,
                                              Q_ref=Q_ref, M=50, bin_ratio=0.2, num_d_emulate=1E5)


# create emulated points
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
                                                                     data=data, rho_D_M = d_distr_prob, d_distr_samples = d_distr_samples,
                                                                     lambda_emulate=lambda_emulate, d_Tree=d_Tree)
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
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(P_samples = P, samples = lambda_emulate, lam_domain = lam_domain, nbins = [20, 20])
# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D,bins, sigma=0.5)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, lam_domain, filename = "nonlinearMap",
                             plot_surface=False)

                             
# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(P_samples = P, samples = lambda_emulate, lam_domain = lam_domain, nbins = [20, 20])
# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.5)
# plot 1d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, lam_domain, filename = "nonlinearMap")





