import numpy as np
import itertools
import bet.sensitivity.chooseQoIs as cQoI
import bet.sensitivity.gradients as grad
import scipy
import matplotlib.pyplot as plt
from itertools import combinations
from pylab import *

###############################################################################

Lambda_min = 0.0
Lambda_max = 1.0

Lambda_dim = 2
data_dim = 3
num_samples = 1000
num_centers = 100
poly_degree = 3 # found no mention of this  elsewhere in the script
combs = int(np.round(scipy.misc.comb(data_dim, Lambda_dim)))
combs_array = np.array(list(combinations(range(10),2)))

#seed 0 (2)(5)(7), data_dim=50

np.random.seed(0)

rand_int = np.int(np.round(np.round(np.random.random(1) * 1000)))
#rand_int = 754(100)
# rand_int = 357 #(give inds 38 11 14)  these are the figures.  7 is worst
r = [[0.19, -0.25, 0.09, -0.86, -0.07, 0.29], \
    [0.80, 0.47, 0.08, -0.64,  -0.98,  -1.0], \
    [-0.49, -0.44, -0.21, 0.67,  -0.10, 1.2]]
def Q(x):
    np.random.seed(rand_int)
    q = np.zeros([x.shape[0], data_dim])
    for i in range(data_dim):
        # rand_vec = 2 * np.random.random(6) - 1
        # q[:, i] = rand_vec[0] * x[:, 0]**5 + rand_vec[1] * x[:, 1]**3 + \
        #     rand_vec[2] * x[:, 0] **3 * x[:, 1] + rand_vec[3] * x[:, 0] + \
        #     rand_vec[4] * x[:, 1] + rand_vec[5]
        q[:, i] = r[i][0] * x[:, 0]**5 + r[i][1] * x[:, 1]**3 + \
            r[i][2] * x[:, 0] **3 * x[:, 1] + r[i][3] * x[:, 0] + \
            r[i][4] * x[:, 1] + r[i][5]
    np.random.seed(None)
    return q

np.random.seed(0)
samples = np.random.random([num_samples, Lambda_dim])
data = Q(samples)
G = grad.calculate_gradients_rbf(samples, data, centers=samples[:num_centers, :], num_neighbors=20, normalize=False)

# Set bin_size (or radius) and choose Q_ref to invert to middle of Lambda
bin_size = 0.2
bin_radius = 0.1
Q_ref = Q(0.5 * np.ones([1, Lambda_dim]))

best_sets_support = cQoI.chooseOptQoIs_large(G, max_qois_return=Lambda_dim, num_optsets_return=combs, inner_prod_tol=1.0, cond_tol=np.inf, volume=True)#, bin_measure=bin_size**2)

best_sets_skewness = cQoI.chooseOptQoIs_large(G, max_qois_return=Lambda_dim, num_optsets_return=combs, inner_prod_tol=1.0, cond_tol=np.inf, volume=False)


skewness_vec = best_sets_skewness[0][:,0]
support_vec = best_sets_support[0][:,0]

dist_p_sz = support_vec / (1 + support_vec) + (skewness_vec - 1) / skewness_vec

min_suppind = np.argmin(support_vec)
min_skewind = np.argmin(skewness_vec)

skewness_vec_dist = (skewness_vec - 1) / (1 + skewness_vec - 1)
support_vec_dist = support_vec / (1 + support_vec)

sum_suppskew = skewness_vec_dist + support_vec_dist

min_sumind = np.argmin(sum_suppskew)

# worst is index 7

print min_suppind
print min_skewind
print min_sumind

fsize=30
xmax=4.5
ymax=0.15
# highlight minsupp, minskew and minsum

highlight = min_suppind
plt.scatter(skewness_vec, support_vec, s=50)
plt.scatter(skewness_vec[highlight], support_vec[highlight], color='r', s=100)
# plt.xlim(0,xmax)
# plt.ylim(0,ymax)
#plt.xlim(0.9,2)
#plt.ylim(0,.1)
plt.xlabel('$\overline{S_{Q^{(z)},N}}$', fontsize=fsize)
plt.ylabel('$\overline{M_{Q^{(z)},N}}$', fontsize=fsize)
plt.title('Possible sets of QoI', fontsize=fsize)
plt.show()

highlight = min_skewind
plt.scatter(skewness_vec, support_vec, s=50)
plt.scatter(skewness_vec[highlight], support_vec[highlight], color='r', s=100)
# plt.xlim(0,xmax)
# plt.ylim(0,ymax)
#plt.xlim(0.9,2)
#plt.ylim(0,.1)
plt.xlabel('$\overline{S_{Q^{(z)},N}}$', fontsize=fsize)
plt.ylabel('$\overline{M_{Q^{(z)},N}}$', fontsize=fsize)
plt.title('Possible sets of QoI', fontsize=fsize)
plt.show()

highlight = min_sumind
plt.scatter(skewness_vec, support_vec, s=50)
plt.scatter(skewness_vec[highlight], support_vec[highlight], color='g', s=100)
# plt.xlim(0,xmax)
# plt.ylim(0,ymax)
#plt.xlim(0.9,2)
#plt.ylim(0,.1)
plt.xlabel('$\overline{S_{Q^{(z)},N}}$', fontsize=fsize)
plt.ylabel('$\overline{M_{Q^{(z)},N}}$', fontsize=fsize)
plt.title('Possible sets of QoI', fontsize=fsize)
plt.show()

highlight = 1
plt.scatter(skewness_vec, support_vec, s=50)
plt.scatter(skewness_vec[highlight], support_vec[highlight], color='r', s=100)
# plt.xlim(0,xmax)
# plt.ylim(0,ymax)
#plt.xlim(0.9,2)
#plt.ylim(0,.1)
plt.xlabel('$\overline{S_{Q^{(z)},N}}$', fontsize=fsize)
plt.ylabel('$\overline{M_{Q^{(z)},N}}$', fontsize=fsize)
plt.title('Possible sets of QoI', fontsize=fsize)
plt.show()

qoi_combs = np.array(list(combinations(range(data_dim), 2)))

inds = [min_suppind, min_skewind, min_sumind]

delta = 0.025
x = np.arange(Lambda_min, Lambda_max+.1, delta)
y = np.arange(Lambda_min, Lambda_max+.1, delta)
X, Y = np.meshgrid(x, y)
Xv = X.ravel()
Yv = Y.ravel()
Z = Q(np.array([Xv, Yv]).transpose())

fsize=20
for i in inds:
    plt.figure()
    qoi_set = qoi_combs[i]
    qi = qoi_set[0]
    qj = qoi_set[1]

    skew = cQoI.calculate_avg_skewness(G, qoi_set)[0]
    supp = cQoI.calculate_avg_volume(G, qoi_set, bin_volume=bin_size**2)[0]

    Z1 = Z[:, qi].reshape(x.shape[0], y.shape[0])
    Z2 = Z[:, qj].reshape(x.shape[0], y.shape[0])

    levels1 = np.arange(Q_ref[0, qi] - bin_radius, Q_ref[0, qi] + bin_radius, 0.19999)
    levels2 = np.arange(Q_ref[0, qj] - bin_radius, Q_ref[0, qj] + bin_radius, 0.19999)

    cset1 = contourf(X, Y, Z1, levels1, colors='r', alpha=.4)
    cset2 = contourf(X, Y, Z2, levels2, colors='b', alpha=.4)

    plt.title('$Q_{%i,%i}$' % (qi, qj), fontsize=fsize)
    plt.xlabel('$\lambda_1$', fontsize=fsize)
    plt.ylabel('$\lambda_2$', fontsize=fsize)

plt.show()
