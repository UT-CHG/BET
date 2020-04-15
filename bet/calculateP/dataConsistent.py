# Copyright (C) 2014-2020 The BET Development Team
import bet.sample
import numpy as np
import logging


def generate_output_kdes(discretization):
    """

    :param discretization: Discretization on which to perform inversion.
    :type discretization: :class:`bet.sample.discretization`
    :return:
    """
    from scipy.stats import gaussian_kde
    discretization.local_to_global()

    predict_set = discretization.get_output_sample_set()
    obs_set = discretization.get_output_probability_set()
    if predict_set.get_region() is None or obs_set.get_region() is None:
        predict_set.set_region(np.array([0] * predict_set.check_num()))
        obs_set.set_region(np.array([0] * obs_set.check_num()))

    if predict_set.get_cluster_maps() is None:
        num_clusters = int(max(np.max(predict_set.get_region()), np.max(obs_set.get_region())) + 1)
    else:
        num_clusters = len(predict_set.get_cluster_maps())

    predict_kdes = []
    obs_kdes = []
    for i in range(num_clusters):
        if predict_set.get_cluster_maps() is not None:
            if len(predict_set.get_cluster_maps()) > 1:
                predict_kdes.append(gaussian_kde(predict_set.get_cluster_maps()[i].T))
            else:
                predict_kdes.append(None)
        else:
            predict_pointer = np.where(predict_set.get_region() == i)[0]
            if len(predict_pointer) > 1:
                predict_kdes.append(gaussian_kde(predict_set.get_values()[predict_pointer].T))
            else:
                predict_kdes.append(None)

        if obs_set.get_cluster_maps() is not None:
            if len(obs_set.get_cluster_maps()) > 1:
                obs_kdes.append(gaussian_kde(obs_set.get_cluster_maps()[i].T))
            else:
                obs_kdes.append(None)
        else:
            obs_pointer = np.where(obs_set.get_region() == i)[0]
            if len(obs_pointer) > 1:
                obs_kdes.append(gaussian_kde(obs_set.get_values()[obs_pointer].T))
            else:
                obs_kdes.append(None)

        # obs_pointer = np.where(obs_set.get_region() == i)[0]
        # if len(predict_pointer) > 1:
        #     predict_kdes.append(gaussian_kde(predict_set.get_values()[predict_pointer].T))
        # else:
        #     predict_kdes.append(None)
        #
        # if len(obs_pointer) > 1:
        #     obs_kdes.append(gaussian_kde(obs_set.get_values()[obs_pointer].T))
        # else:
        #     obs_kdes.append(None)
    predict_set.set_kdes(predict_kdes)
    obs_set.set_kdes(obs_kdes)
    return predict_set, obs_set, num_clusters


def dc_inversion_gkde(discretization):
    """

    :param discretization: Discretization on which to perform inversion.
    :type discretization: :class:`bet.sample.discretization`
    :return:
    """
    from scipy.stats import gaussian_kde

    predict_set, obs_set, num_clusters = generate_output_kdes(discretization)
    predict_kdes = predict_set.get_kdes()
    obs_kdes = obs_set.get_kdes()

    rs = []
    r = []
    lam_ptr = []
    for i in range(num_clusters):
        predict_pointer = np.where(predict_set.get_region() == i)[0]
        # First compute the rejection ratio
        if predict_set.get_cluster_maps() is None:
            vals = predict_set.get_values()[predict_pointer]
        else:
            vals = predict_set.get_cluster_maps()[i]
        if len(predict_pointer) > 0:
            r.append(np.divide(obs_kdes[i](vals.T), predict_kdes[i](vals.T)))
            rs.append((r[i].mean()))
        else:
            r.append(None)
            rs.append(None)
        lam_ptr.append(predict_pointer)

    # Compute marginal probabilities for each parameter and initial condition.
    param_marginals = []
    cluster_weights = []
    num_obs = obs_set.check_num()

    input_dim = discretization.get_input_sample_set().get_dim()
    params = discretization.get_input_sample_set().get_values()

    for i in range(num_clusters):
        cluster_weights.append(len(np.where(obs_set.get_region() == i)[0]) / num_obs)
    for i in range(input_dim):
        param_marginals.append([])
        for j in range(num_clusters):
            if r[j] is not None:
                param_marginals[i].append(gaussian_kde(params[lam_ptr[j], i], weights=r[j]))
            else:
                param_marginals[i].append(None)
    discretization.get_input_sample_set().set_prob_type("kde")
    discretization.get_input_sample_set().set_prob_parameters((param_marginals, cluster_weights))
    print('Diagnostic for clusters [sample average of ratios in each cluster]: ', rs)
    return param_marginals, cluster_weights
