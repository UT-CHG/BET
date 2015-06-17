"""
This module provides methods for creating simple funciton approximations to be
used by :mod:`~bet.calculateP.calculateP`.
"""
from bet.Comm import *
import numpy as np
import scipy.spatial as spatial
import bet.calculateP.voronoiHistogram as vHist
import collections
from bet.simpleFunP import *

# TODO since mpi4py DOES NOT PRESERVE ORDERING double check that we are not
# presuming it preserse order anywhere

#TODO Empty function hist_regular 
def hist_regular(data, distr_samples, nbins):
    """
    create nbins regulary spaced bins
    check to make sure  each bin has about 1 data sample per bin, if not
    recompute bins
    (hist, edges) = histdd(distr_samples, bins)
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogramdd.html#numpy.histogramdd
    determine d_distr_samples from edges
    """
    pass

# TODO Empty function hist_gaussian
def hist_gaussian(data, distr_samples, nbins):
    """
    determine mean, standard deviation of distr_samples
    partition D into nbins of equal probability for N(mean, sigma)
    check to make sure  each bin has about 1 data sample per bin, if not
    recompute bins
    (hist, edges) = histdd(distr_samples, bins)
    determine d_distr_samples from edges
    """
    pass

# TODO Empty Function hist_unif
def hist_unif(data, distr_samples, nbins):
    """
    same as hist_regular bit with uniformly spaced bins
    unif_unif can and should call this function
    """
    pass

# TODO Empty function gaussian_regular
def gaussian_regular(data, Q_ref, std, nbins, num_d_emulate=1E6):
    pass
    #return (d_distr_prob, d_distr_samples, d_Tree)

# TODO Comment or remove multivariate_gaussian
def multivariate_gaussian(x, mean, std):
    dim = len(mean)
    detDiagCovMatrix = np.sqrt(np.prod(np.diag(std(std))))
    frac = (2.0*np.pi)**(-dim/2.0)  * (1.0/detDiagCovMatrix)
    fprime = x-mean
    return frac*np.exp(-0.5*np.dot(fprime, 1.0/np.diag(std*std)))

# TODO gaussian_unif is a blank function
def gaussian_unif(data, Q_ref, std, nbins, num_d_emulate=1E6):
    pass
    #return (d_distr_prob, d_distr_samples, d_Tree)

