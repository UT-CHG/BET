import numpy as np
import time
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoI
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.postTools as postTools
import bet.Comm as comm
import itertools
from scipy.special import comb
import scipy.spatial as spatial
from itertools import combinations
from pylab import *

# import plot_piecewise

Lambda_dim = 2
Data_dim = 3
num_samples = 1E5
num_anchors = 10
bin_ratio = 0.25
time_0 = time.clock()
# num_grad_centers = 100 # at how many points  do we compute gradient information?

# define samples in parameter space, random anchor points
np.random.seed(0)
samples = np.random.random([num_samples, Lambda_dim])
anchors  = np.random.random([num_anchors, Lambda_dim])
# anchors = np.array([[0.5, 0.5]])
np.random.seed(0)

# define QoI maps and map samples to data space
rand_int = int(np.round(np.random.random(1) * 1000))
coeffs = [[0.19, -0.25, 0.09, -0.86, -0.07, 0.29], \
    [0.80, 0.47, 0.08, -0.64,  -0.98,  -1.0], \
    [-0.49, -0.44, -0.21, 0.67,  -0.10, 1.2]]
def Q(x): # example case for Lambda_dim = 2, Data_dim = 3
    q = np.zeros([x.shape[0], Data_dim])
    for i in range(Data_dim):
        q[:, i] = coeffs[i][0] * x[:, 0]**5 + coeffs[i][1] * x[:, 1]**3 + \
            coeffs[i][2] * x[:, 0] **3 * x[:, 1] + coeffs[i][3] * x[:, 0] + \
            coeffs[i][4] * x[:, 1] + coeffs[i][5]
    np.random.seed(None)
    return q

def randQ(x): # QoI map using quintic functions for Lambda_dim = 2, Data_dim arbitrary
    np.random.seed(rand_int)
    q = np.zeros([x.shape[0], Data_dim])
    for i in range(Data_dim):
        rand_vec = 2 * np.random.random(6) - 1
        q[:, i] = rand_vec[0] * x[:, 0]**5 + rand_vec[1] * x[:, 1]**3 + \
        rand_vec[2] * x[:, 0] **3 * x[:, 1] + rand_vec[3] * x[:, 0] + \
        rand_vec[4] * x[:, 1] + rand_vec[5]
    np.random.seed(None)
    return q


data = Q(samples)
# print data[0:10]

# perform nearest neighbor searches to set of K anchor points
tree = spatial.KDTree(anchors)
[_, near_anchor] = tree.query(samples)
part_inds = [np.where(near_anchor == i) for i in range(len(anchors))] # index into samples for each partition
if sum( [ len( samples[part_inds[i]] ) for i in range(num_anchors) ] ) != num_samples:
    sys.exit("Something went wrong with nearest neighbor search. Some samples missed.")

# compute possible sets of quantities of interest
combs = int(comb(Data_dim, Lambda_dim))
combs_array = np.array(list(combinations(range(Data_dim),2)))

# feed each list of indices into samples and data, perform chooseQoIs
best_sets = []

for k in range(num_anchors):
    samples_k = np.array(anchors[k],ndmin=2)
    data_k = Q(samples_k)

    # Calculate the gradient vectors at some anchor points.
    # Here the *normalize* argument is set to *True* because we are using bin_ratio to
    # determine the uncertainty in our data.
    G = grad.calculate_gradients_rbf(samples = samples, \
                                        data = data, \
                                        centers = samples_k, \
                                        normalize=True)
    print '\n Partition %d  - Anchor =\n\t'%(k+1), anchors[k,:], '\n'
    best_set_for_anchor = cQoI.chooseOptQoIs_large(grad_tensor = G, \
                                        num_optsets_return = 1, \
                                        volume = False )[Lambda_dim-2][0][1:]
    best_sets.append( [int(best_set_for_anchor[i]) for i in range(Lambda_dim) ] )
    # for each anchor point, record best_sets (accessing [0] for the best one).
print  '\n'
print best_sets

# have a dictionary object or something comparable track all nonempty choices of
# sets of QoI maps, list of indices into samples.

# feed each along with the set of QoIs into the inverse problem.
# solve inverse problem.

P = np.zeros(num_samples)
lam_vol = np.zeros(num_samples)
total = []
ref_lambda  = [0.25, 0.25]

anchors_for_best_set = [np.where((best_sets == combs_array[i]).all(axis=1))[0] for i in range(combs)]
unique_part_inds = []
for idx_array in anchors_for_best_set: # indices of anchors associated with each best set (some may be empty)
    temp_index_list = np.array([], dtype=int8)
    for idx in idx_array:
        temp_index_list = np.concatenate([temp_index_list, part_inds[idx][0]])
    # if temp_index_list.shape[0] > 0: unique_part_inds.append(temp_index_list)
    unique_part_inds.append(temp_index_list)

# figure out what happens if we have empty entries in best sets, make sure ordering is correct.


for k in range(len(unique_part_inds)): # run through unique
    if len(unique_part_inds[k])>0:
        QoI_indices = combs_array[k]
        temp_samples = samples[ unique_part_inds[k] ]
        temp_data = data[:, QoI_indices]
        Q_ref = Q(np.array([ref_lambda]))[0][QoI_indices]

        # Find the simple function approximation to data space density
        (d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle(\
                                                data = temp_data, \
                                                Q_ref = Q_ref, \
                                                bin_ratio = bin_ratio, \
                                                center_pts_per_edge = 1)

        # Calculate probablities making the Monte Carlo assumption
        (temp_P,  temp_lam_vol, io_ptr) = calculateP.prob(samples = temp_samples, \
                                                data = temp_data, \
                                                rho_D_M = d_distr_prob, \
                                                d_distr_samples = d_distr_samples)

        P[ unique_part_inds[k] ] = temp_P*len(temp_P[temp_P>0])
        lam_vol[ unique_part_inds[k] ] = temp_lam_vol*len(temp_P[temp_P>0])
        total.append( len(temp_P[temp_P>0]) )
P = P/sum(total)
lam_vol = lam_vol/sum(total)
ptol = 1E-4
if abs(1-sum(P))>ptol:
    sys.exit('Probability measure deviates from 1 by more than %f. %f'%(sum(P), ptol) )

percentile = 1.0
# Sort samples by highest probability density and find how many samples lie in
# the support of the inverse solution.  With the Monte Carlo assumption, this
# also tells us the approximate volume of this support.
(num_high_samples, P_high, samples_high, lam_vol_high, data_high, sort) =\
    postTools.sample_highest_prob(top_percentile = percentile, \
                                    P_samples = P, \
                                    samples = samples, \
                                    data = data, \
                                    lam_vol = lam_vol, \
                                    sort = True)

    # Print the number of samples that make up the highest percentile percent
    # samples and ratio of the volume of the parameter domain they take up
print '\n'
if comm.rank == 0:
    print (num_high_samples, np.sum(lam_vol_high), sum(P))
    print time.clock() - time_0


# weights distributed as proportion of total weight in cell multiplied by the
# proportion it represents of all samples across partitions that fell in one
# of the data space  densities. really your only contribution is this latter proportioning.
