# Copyright (C) 2014-2015  BET Development Team

"""Run the heatplate simulation over a set of samples of the parameter space"""
''' python simulation_2kappas_run test '''

from dolfin import *
import sys, numpy, time
import scipy.io as sio
import random
from bet.Comm import *

from simulation_4kappas_setup import *
from heatplate import *

from datetime import datetime

startTime = datetime.now()

functional_list = []

#loop over each sample and call heatplate
print 'num_samples : ', num_samples
for i in range(len(samples)):
    print 'Sample : ', i
    print 'kappavec = ', samples[i,:]
    kappavec = samples[i,:]

    heatplatecenter(amp, px, py, width, degree, T_R, kappavec, rho, cap, nx, ny, mesh, functional_list, radii, angles, corner_radii, corner_angles, dt, t_stop)

# Find the minimum of the minimums
if rank==0:
    #oraganize data
    data = numpy.reshape(functional_list, (num_samples, num_points*total_timesteps))
    #sio.savemat('heatplate_9d_100clusters_10000qoi', {'data':F, 'samples':x})
    #data = F[:,0:40]

    #output parameter and functional data to matlab file
    #filename = sys.argv[1]
    #sio.savemat(filename, {'data':data, 'samples':x})

if rank==0:
    print ' '
    print datetime.now() - startTime


