# Copyright (C) 2014-2015 Lindley Graham and Steven Mattis

""" Setup up some model parameters"""

from dolfin import *
import numpy as np
import sys
from bet.sensitivity.gradients import *
from bet.Comm import *

#time stepping info
dt = 1./10
t_stop = 5.
total_timesteps = t_stop/dt

#Some fixed parameter values
amp = 50.0  #amplitude of the heat source
px = 0  #location of the heat source
py = 0
width = 0.05 #width of the heat source
T_R = 0 #initial temp of the plate
cap = 1.5 #heat capacity
rho = 1.5 #density

#define the mesh properties
degree = 1
nx = 40
ny = 40
mesh = RectangleMesh(-0.5, -0.5, 0.5, 0.5, nx, ny)
parameters['allow_extrapolation'] = True


#set points on the plate to approximate the temperature at
radii = [1./4, 3./8]
angles = [0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 6*pi/4, 7*pi/4]
corner_radii = [sqrt(2*(1./2)**2-1E-1)]
corner_angles = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]
num_points = len(radii)*len(angles)+len(corner_radii)*len(corner_angles)

#the thermal conductivity will be uncertain in 9 regions of the plate
kappa_min = 0.01
kappa_max = 0.2

#random sampling
Lambda_dim = 2
r = (kappa_max-kappa_min)/100.

# Cluster points around each xeval
num_xeval = 16
xeval = (kappa_max-r-kappa_min-r)*np.random.random((num_xeval,Lambda_dim)) + kappa_min+r
#samples = sample_l1_ball(xeval, Lambda_dim+1, r)
samples = pick_ffd_points(xeval, r)
#samples = xeval
num_samples = samples.shape[0]





