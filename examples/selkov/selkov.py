import bet.sampling.basicSampling as bsam
import bet.calculateP.dataConsistent as dc
import bet.sampling.useLUQ as useLUQ
import bet.postProcess.plotP as plotP
import numpy as np


p_set = bsam.random_sample_set(rv=[['uniform', {'loc': .01, 'scale': 0.114}],
                              ['uniform', {'loc': .05, 'scale': 1.45}]],
                               input_obj=2, num_samples=300)

o_set = bsam.random_sample_set(rv=[['beta', {'a': 2, 'b': 2, 'loc': .01, 'scale': 0.114}],
                                   ['beta', {'a': 2, 'b': 2, 'loc': .05, 'scale': 1.45}]],
                               input_obj=2, num_samples=300)

# Construct the predicted time series data
time_start = 2.0
time_end = 6.5 
num_time_preds = int((time_end-time_start)*100)  # number of predictions (uniformly space) between [time_start,time_end]
times = np.linspace(time_start, time_end, num_time_preds)


luq = useLUQ.useLUQ(predict_set=p_set, obs_set=o_set, lb_model=useLUQ.myModel, times=times)
luq.setup()

time_start_idx = 0
time_end_idx = len(luq.times) - 1
luq.clean_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
               num_clean_obs=20, tol=5.0e-2, min_knots=3, max_knots=12)

luq.dynamics(cluster_method='kmeans', kwargs={'n_clusters': 3, 'n_init': 10})
luq.learn_qois_and_transform(num_qoi=2)
disc1, disc2 = luq.make_disc()

param_labels = [r'$a$', r'$b$']
dc.invert_to_multivariate_gaussian(disc1)
for i in range(2):
    plotP.plot_marginal(sets=(disc1.get_input_sample_set(), disc2.get_input_sample_set()), i=i,
                        sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                        title="Multivariate Gaussian", label=param_labels[i])

dc.invert_to_gmm(disc1)
for i in range(2):
    plotP.plot_marginal(sets=(disc1.get_input_sample_set(), disc2.get_input_sample_set()), i=i,
                        sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                        title="Gaussian Mixture Model", label=param_labels[i])

dc.invert_to_kde(disc1)
for i in range(2):
    plotP.plot_marginal(sets=(disc1.get_input_sample_set(), disc2.get_input_sample_set()), i=i,
                        sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                        title="Weighted KDEs", label=param_labels[i]
                        )

dc.invert_to_random_variable(disc1, rv='beta')
for i in range(2):
    plotP.plot_marginal(sets=(disc1.get_input_sample_set(), disc2.get_input_sample_set()), i=i,
                        sets_label_initial=['Initial', 'Data-Generating'], sets_label=['Updated', ''],
                        title="Fitted Beta Distribution", label=param_labels[i]
                        )
