
P = np.zeros(num_samples)
lam_vol = np.zeros(num_samples)
total = []
for k in range(num_anchors):
    QoI_indices = best_sets[k]
    temp_samples = samples[ part_inds[k] ]
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
    # P[ part_inds[k] ] = temp_P*len( samples[ part_inds[k] ] )
    # lam_vol[ part_inds[k] ] = temp_lam_vol*len( samples[ part_inds[k] ] )
    P[ part_inds[k] ] = temp_P*len(temp_P[temp_P>0])
    lam_vol[ part_inds[k] ] = temp_lam_vol*len(temp_P[temp_P>0])
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
