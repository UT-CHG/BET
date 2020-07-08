# Copyright (C) 2014-2020 The BET Development Team

import bet.sampling.basicSampling as bsam
import bet.calculateP.calculateR as calculateR
import bet.sampling.useLUQ as useLUQ
import bet.postProcess.plotP as plotP
import bet.postProcess.compareP as compP
import numpy as np

"""
Use LUQ to solve the Sel'kov model for glycolysis and learn quantities of interest.
This also illustrates several different options available within `calculateR`  to approximate the updated density.

The LUQ package must be installed to run this example.
"""

# sample for prediction set
p_set = bsam.random_sample_set(rv=[['uniform', {'loc': .01, 'scale': 0.114}],
                              ['uniform', {'loc': .05, 'scale': 1.45}]],
                               input_obj=2, num_samples=500)

# sample for observation set
o_set = bsam.random_sample_set(rv=[['beta', {'a': 2, 'b': 2, 'loc': .01, 'scale': 0.114}],
                                   ['beta', {'a': 2, 'b': 2, 'loc': .05, 'scale': 1.45}]],
                               input_obj=2, num_samples=500)

# Construct the predicted time series data
time_start = 2.0
time_end = 6.5
num_time_preds = int((time_end-time_start)*100)  # number of predictions (uniformly space) between [time_start,time_end]
times = np.linspace(time_start, time_end, num_time_preds)

# Initialize and setup LUQ
luq = useLUQ.useLUQ(predict_set=p_set, obs_set=o_set, lb_model=useLUQ.myModel, times=times)
luq.setup()

# Clean the data
time_start_idx = 0
time_end_idx = len(luq.times) - 1
luq.clean_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
               num_clean_obs=20, tol=5.0e-2, min_knots=3, max_knots=12)

# Cluster and classify the dynamics
luq.dynamics(cluster_method='kmeans', kwargs={'n_clusters': 3, 'n_init': 10})

# Learn quantities of interest and transform the data
luq.learn_qois_and_transform(num_qoi=2)

# Convert LUQ output to discretization objects
disc1, disc2 = luq.make_disc()

# Set labels
param_labels = [r'$a$', r'$b$']

# Calculate initial total variation
comp_init = compP.compare(disc1, disc2, set1_init=True, set2_init=True)
print("Initial TV")
for i in range(2):
    print(comp_init.distance_marginal_quad(i=i, compare_factor=0.2, rtol=1.0e-3, maxiter=100))
# Invert to multivariate Gaussian
print("------------------------------------------------------")
print("Multivariate Gaussian")
calculateR.invert_to_multivariate_gaussian(disc1)

# Plot marginal probabilities and calculate total variations between probability measures
for i in range(2):
    plotP.plot_1d_marginal_densities(sets=(disc1.get_input_sample_set(), disc2.get_input_sample_set()), i=i,
                        sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                        title="Multivariate Gaussian", label=param_labels[i])

# Calculate updated total variation
comp_init = compP.compare(disc1, disc2, set1_init=False, set2_init=True)
print("Updated TV")
for i in range(2):
    print(comp_init.distance_marginal_quad(i=i, compare_factor=0.2, rtol=1.0e-3, maxiter=100))

# Invert to Gaussian Mixture Model
print("------------------------------------------------------")
print("Gaussian Mixture Model")
calculateR.invert_to_gmm(disc1)
for i in range(2):
    plotP.plot_1d_marginal_densities(sets=(disc1.get_input_sample_set(), disc2.get_input_sample_set()), i=i,
                        sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                        title="Gaussian Mixture Model", label=param_labels[i])
# Calculate updated total variation
comp_init = compP.compare(disc1, disc2, set1_init=False, set2_init=True)
print("Updated TV")
for i in range(2):
    print(comp_init.distance_marginal_quad(i=i, compare_factor=0.2, rtol=1.0e-3, maxiter=100))

print("------------------------------------------------------")
print("Weighted Kernel Density Estimate")
calculateR.invert_to_kde(disc1)
for i in range(2):
    plotP.plot_1d_marginal_densities(sets=(disc1.get_input_sample_set(), disc2.get_input_sample_set()), i=i,
                        sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                        title="Weighted KDEs", label=param_labels[i]
                        )
# Calculate updated total variation
comp_init = compP.compare(disc1, disc2, set1_init=False, set2_init=True)
print("Updated TV")
for i in range(2):
    print(comp_init.distance_marginal_quad(i=i, compare_factor=0.2, rtol=1.0e-3, maxiter=100))

print("------------------------------------------------------")
print("Beta distribution")

calculateR.invert_to_random_variable(disc1, rv='beta')
for i in range(2):
    plotP.plot_1d_marginal_densities(sets=(disc1.get_input_sample_set(), disc2.get_input_sample_set()), i=i,
                        sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                        title="Fitted Beta Distribution", label=param_labels[i]
                        )
# Calculate updated total variation
comp_init = compP.compare(disc1, disc2, set1_init=False, set2_init=True)
print("Updated TV")
for i in range(2):
    print(comp_init.distance_marginal_quad(i=i, compare_factor=0.2, rtol=1.0e-3, maxiter=100))
