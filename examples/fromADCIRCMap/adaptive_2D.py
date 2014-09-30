#! /usr/bin/env python
# import necessary modules
import polyadcirc.run_framework.domain as dom
import polyadcirc.run_framework.random_wall_Q as rmw
import numpy as np
import polyadcirc.pyADCIRC.basic as basic
import bet.sampling.adaptiveSampling as asam
import bet.sampling.basicSampling as bsam
import scipy.io as sio

adcirc_dir = '/work/01837/lcgraham/v50_subdomain/work'
grid_dir = adcirc_dir + '/ADCIRC_landuse/Inlet_b2/inputs/poly_walls'
save_dir = adcirc_dir + '/ADCIRC_landuse/Inlet_b2/runs/adaptive_random_2D'
basis_dir = adcirc_dir +'/ADCIRC_landuse/Inlet_b2/gap/beach_walls_2lands'
# assume that in.prep* files are one directory up from basis_dir
script = "adaptive_random_2D.sh"
# set up saving
model_save_file = 'py_save_file'
sample_save_file = 'full_run'

# Select file s to record/save
timeseries_files = []#["fort.63"]
nontimeseries_files = ["maxele.63"]#, "timemax63"]

# NoNx12/TpN where NoN is number of nodes and TpN is tasks per node, 12 is the
# number of cores per node See -pe line in submission_script <TpN>way<NoN x
# 12>
nprocs = 4 # number of processors per PADCIRC run
ppnode = 16
NoN = 20
TpN = 16 # must be 16 unless using N option
num_of_parallel_runs = (TpN*NoN)/nprocs

domain = dom.domain(grid_dir)
domain.update()
main_run = rmw.runSet(grid_dir, save_dir, basis_dir, num_of_parallel_runs,
        base_dir=adcirc_dir, script_name=script)
main_run.initialize_random_field_directories(num_procs=nprocs)

# Set minima and maxima
lam_domain = np.array([[.07, .15], [.1, .2]])
lam3 = 0.012
ymin = -1050
xmin = 1420
xmax = 1580
ymax = 1500
wall_height = -2.5

param_min = lam_domain[:, 0]
param_max = lam_domain[:, 1]

# Create stations
stat_x = np.concatenate((1900*np.ones((7,)), [1200], 1300*np.ones((3,)),
    [1500])) 
stat_y = np.array([1200, 600, 300, 0, -300, -600, -1200, 0, 1200,
        0, -1200, -1400])
all_stations = []
for x, y in zip(stat_x, stat_y):
    all_stations.append(basic.location(x, y))

# Select only the stations I care about this will lead to better sampling
station_nums = [0, 5] # 1, 6
stations = []
for s in station_nums:
    stations.append(all_stations[s])

# Read in Q_ref and Q to create the appropriate rho_D 
mdat = sio.loadmat('Q_2D')
Q = mdat['Q']
Q = Q[:, station_nums]
Q_ref = mdat['Q_true']
Q_ref = Q_ref[15, station_nums] # 16th/20
bin_ratio = .15
bin_size = (np.max(Q, 0)-np.min(Q, 0))*bin_ratio

# Create experiment model
def model(sample):
    # box_limits [xmin, xmax, ymin, ymax, wall_height]
    wall_points = np.outer([xmin, xmax, ymin, ymax, wall_height],
            np.ones(sample.shape[1]))
    # [lam1, lam2, lam3]
    mann_pts = np.vstack((sample, lam3*np.ones(sample.shape[1])))
    return main_run.run_nobatch_q(domain, wall_points, mann_pts,
            model_save_file, num_procs=nprocs, procs_pnode=ppnode,
            stations=stations, TpN=TpN)

# Create rho_D
maximum = 1/np.product(bin_size)
def rho_D(outputs):
    rho_left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
    rho_right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
    rho_left = np.all(np.greater_equal(outputs, rho_left), axis=1)
    rho_right = np.all(np.less_equal(outputs, rho_right), axis=1)
    inside = np.logical_and(rho_left, rho_right)
    max_values = np.repeat(maximum, outputs.shape[0], 0)
    return inside.astype('float64')*max_values

# Create sampler
chain_length = 125
num_chains = 80
num_samples = chain_length*num_chains
sampler = asam.sampler(num_samples, chain_length, model)

print sampler.num_samples
print sampler.chain_length
print sampler.num_chains

# Get samples
initial_sample_type = "lhs"
# Create Transition Kernel
transition_kernel = asam.transition_kernel(0.5, .5**5, 0.50)
heuristic = asam.rhoD_heuristic(maximum, rho_D, 1e-4, 1.5, .1) 
(samples, data, all_step_ratios) = sampler.generalized_chains(param_min,
        param_max, transition_kernel, heuristic, sample_save_file,
        initial_sample_type)

bsam.in_high_prob(data, rho_D, maximum)

print np.mean(all_step_ratios)

