import numpy as np
highest_prob = np.load('prob_reduction_results.npy')

print highest_prob[:,[0,2]]
