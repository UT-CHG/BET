import numpy as np
import bet.sensitivity.gradients as grad
import bet.sensitivity.chooseQoIs as cQoI
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.postTools as postTools
import bet.Comm as comm
import itertools
from scipy.special import comb
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from itertools import combinations
from pylab import *

Lambda_dim = 2
Data_dim = 3
num_samples = 1E5
num_anchors = 2
num_grad_centers = 100 # at how many points  do we compute gradient information?

# define samples in parameter space, random anchor points
np.random.seed(0)
samples = np.random.random([num_samples, Lambda_dim])
anchors  = np.random.random([num_anchors, Lambda_dim])
np.random.seed(0)

# define QoI maps and map samples to data space
rand_int = np.int(np.round(np.random.random(1) * 1000))
r = [[0.19, -0.25, 0.09, -0.86, -0.07, 0.29], \
    [0.80, 0.47, 0.08, -0.64,  -0.98,  -1.0], \
    [-0.49, -0.44, -0.21, 0.67,  -0.10, 1.2]]
def Q(x): # example case for Lambda_dim = 2, Data_dim = 3
    q = np.zeros([x.shape[0], Data_dim])
    for i in range(Data_dim):
        q[:, i] = r[i][0] * x[:, 0]**5 + r[i][1] * x[:, 1]**3 + \
            r[i][2] * x[:, 0] **3 * x[:, 1] + r[i][3] * x[:, 0] + \
            r[i][4] * x[:, 1] + r[i][5]
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

# perform nearest neighbor searches to set of K anchor points
tree = spatial.KDTree(anchors)
[r, near_anchor] = tree.query(samples)
part_inds = [np.where(near_anchor == i) for i in range(len(anchors))] # index into samples for each partition

# compute possible sets of quantities of interest
# combs = int(comb(Data_dim, Lambda_dim))
# combs_array = np.array(list(combinations(range(Data_dim),2)))

# feed each list of indices into samples and data, perform chooseQoIs
best_sets = []

for k in range(num_anchors):
    samples_k =  anchors[k:k+1]
    data_k = Q(anchors[k:k+1]) # CANNOT GET THIS LINE WORKING
    # Calculate the gradient vectors at some anchor points.
    # Here the *normalize* argument is set to *True* because we are using bin_ratio to
    # determine the uncertainty in our data.
    G = grad.calculate_gradients_rbf(samples_k, data_k, centers=samples[:num_centers, :],
        normalize=True)
    best_sets.append( cQoI.chooseOptQoIs_large(G, num_optsets_return=1, volume=False) )
    # for each anchor point, record best_sets (accessing [0] for the best one).
print  best_sets


##### NOT WHAT WE'LL DO
# define a quantity of interest map of Lambda_dim where we run nearest neighbor
# on the point against anchor points, then access the best QoI for that anchor point,
# use it to access the appropriate quantity of interest component maps.

# map parameter space under this new map.
# solve inverse problem with this piecwise-defined quantity of interest map.
####

# have a dictionary object or something comparable track all nonempty choices of
# sets of QoI maps, list of indices into samples.

# feed each along with the set of QoIs into the inverse problem.
# solve inverse problem.

# now we have a density that sums to num_unique_parts.


# weights distributed as proportion of total weight in cell multiplied by the
# proportion it represents of all samples across partitions that fell in one
# of the data space  densities. really your only contribution is this latter proportioning.
