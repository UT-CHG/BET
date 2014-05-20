import numpy as np
import bet.calculateP as calculateP
import bet.visualize as visualize
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.visualize.plotP as plotP
lam_domain= np.array([[0.0, 1.0],
                      [0.0, 1.0],
                      [0.0, 1.0]])
n0 = 20
n1 = 20
n2 = 20
Q_map = np.array([[0.506, 0.463],[0.253, 0.918], [0.085, 0.496]])
true_Q =  np.array([0.422, 0.9385])
true_lam = [0.5, 0.5, 0.5]


vec0=list(np.linspace(lam_domain[0][0], lam_domain[0][1], n0))
vec1 = list(np.linspace(lam_domain[1][0], lam_domain[1][1], n1))
vec2 = list(np.linspace(lam_domain[2][0], lam_domain[2][1], n1))
samples = []
for x in vec0:
    for y in vec1:
        for z in vec2:
            samples.append([x,y,z])
samples=np.array(samples)

data= np.dot(samples,Q_map)
import pdb

(d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.unif_unif(data=data,
                                                                         true_Q = true_Q,
                                                                         M = 10,
                                                                         bin_ratio = 0.2,
                                                                         num_d_emulate = 1E4)

#lambda_emulate = calculateP.calculateP.emulate_iid_lebesgue(lambda_domain=lambda_domain,
#                                                            num_l_emulate = 1E4)
(P, lam_vol, lambda_emulate, io_ptr, emulate_ptr) = calculateP.prob(samples=samples,
                                                                               data=data,
                                                                               rho_D_M = d_distr_prob,
                                                                               d_distr_samples = d_distr_samples,
                                                                               lam_domain = lam_domain,
                                                                               d_Tree = d_Tree)
plotP.plot_marginal_probs(P_samples = P,
                    samples = samples,
                    lam_domain = lam_domain,
                    nbins = [10, 10, 10],
                    filename = "linearMap",
                    plot_surface=True)





            

