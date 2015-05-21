# Lindley Graham 04/23/15
"""
This module contains the class :class:`bet.sampling.dev_multi_dist_kernel`
which does not work and may be removed.
"""

import numpy as np
import scipy.io as sio
import bet.sampling.basicSampling as bsam
import math, os
from bet.sampling.adaptiveSampling import kernel

class multi_dist_kernel(kernel):
    """
    The goal is to make a sampling that is robust to different types of
    distributions on QoI, i.e., we do not know a priori where the regions of
    high probability are in D. This class provides a method for determining the
    proposed step size as follows. We keep track of the change of the QoI
    values from one sample to the next compared to the total range of QoI
    values explored so far. If a big relative change is detected, then you know
    that you have come across a region with larger derivatives and you should
    place more samples around there to resolve the induced regions of
    generalized contours, i.e., reduce the step size. If the change in QoI
    values is relatively small, you are in a region where there is little
    sensitivity, so take larger step sizes.

    radius
        current estimate of the radius of D (1/2 the diameter of D)
    mean
        current estimate of the mean QoI
    current_clength
        current batch number
    TOL 
        a tolerance used to determine if two different values are close
    increase
        the multiple to increase the step size by
    decrease
        the multiple to decrease the step size by

    """

    def __init__(self, tolerance=1E-08, increase=2.0, 
            decrease=0.5):
        """
        Initialization
        """
        self.radius = None
        self.mean = None
        self.current_clength = 0
        super(multi_dist_kernel, self).__init__(tolerance, increase,
                decrease)

    def reset(self):
        """
        Resets the the batch number and the estimates of the mean and maximum
        distance from the mean.
        """
        self.radius = None
        self.mean = None
        self.current_clength = 0

    def delta_step(self, data_new, kern_old=None):
        """
        This method determines the proposed change in step size. 
        
        :param data_new: QoI for a given batch of samples 
        :type data_new: :class:`np.array` of shape (num_chains, mdim)
        :param kern_old: QoI evaluated at previous step
        :rtype: tuple
        :returns: (kern_new, proposal)

        """
        # Evaluate kernel for new data.
        kern_new = data_new
        self.current_clength = self.current_clength + 1

        if type(kern_old) == type(None):
            proposal = None
            # calculate the mean
            self.mean = np.mean(data_new, 0)
            # calculate the distance from the mean
            vec_from_mean = kern_new - np.repeat([self.mean],
                    kern_new.shape[0], 0)
            # estimate the radius of D
            self.radius = np.max(np.linalg.norm(vec_from_mean, 2, 1)) 
        else:
            # update the estimate of the mean
            self.mean = (self.current_clength-1)*self.mean + np.mean(data_new, 0)
            self.mean = self.mean / self.current_clength
            # calculate the distance from the mean
            vec_from_mean = kern_new - np.repeat([self.mean],
                    kern_new.shape[0], 0)
            # esitmate the radius of D
            self.radius = max(np.max(np.linalg.norm(vec_from_mean, 2, 1)),
                    self.radius)
            # calculate the relative change in QoI
            kern_diff = (kern_new-kern_old)
            # normalize by the radius of D
            kern_diff = np.linalg.norm(vec_from_mean, 2, 1)#/self.radius
            # Compare to kernel for old data.
            # Is the kernel NOT close?
            kern_close = np.logical_not(np.isclose(kern_diff, 0,
                atol=self.TOL))
            # Is the kernel greater/lesser?
            kern_greater = np.logical_and(kern_diff > 0, kern_close)
            kern_lesser = np.logical_and(kern_diff < 0, kern_close)
            # Determine step size
            proposal = np.ones(kern_diff.shape)
            proposal[kern_greater] = self.decrease
            proposal[kern_lesser] = self.increase
        return (kern_new, proposal)




