# Copyright (C) 2014-2020 The BET Development Team

r"""
This module contains functions for the density-based approach that utilizes a ratio of observed to predicted densities
to update an initial density on the parameter space.

* :meth:`~bet.calculateP.calculateR.generate_output_kdes` generates KDEs on output sets.
* :meth:`~bet.calculateP.calculateR.invert_to_kde` solves SIP for weighted KDEs.
* :meth:`~bet.calculateP.calculateR.invert_to_gmm` solves SIP for a Gaussian Mixture Model.
* :meth:`~bet.calculateP.calculateR.invert_to_multivariate_gaussian` solves SIP for a multivariate Gaussian.
* :meth:`~bet.calculateP.calculateR.invert_to_random_variable` solves SIP for random variables.
* :meth:`~bet.calculateP.calculateR.invert_rejection_sampling` solves SIP with rejection sampling.

"""
import bet.sample
import numpy as np
import logging

def generate_densities(discretization, which_samples, density_method, method_args=None):
    """
    Generate density estimates on sample set(s) in the discretization indicated by "which_sample".

    :param discretization: Discretization used to calculate density estimates
    :type discretization: :class:`bet.sample.discretization`
    :param which_samples: indicates which samples to compute the density estimate. Options are 
        'predict' (for predicted sample set), 'observe' (for observed sample set), and 'output' (for both)
    :type which_samples: str
    :param density_method: name of the method used to implement density estimation. Options are
        'kde' (`scipy.stats.gaussian_kde` implementation), 
        'clustered_kde' (computes gkdes on clusters defined by cluster_maps in the discretization),
        'gmm' (fits a Gaussian Mixture model using `sklearn.mixture.GaussianMixture`)
    :type density_method: str
    :param method_args: dictionary of arguments to be passed to the density estimation method 
        (e.g., bandwidth params, etc.). See specific density estimation implementation for details.
    :type method_args: dict

    :modifies: discretization object. Specifically, this function saves density estimates to the 
        appropriate `sample_set_base` objects associated with the discretization object.
        
    :returns: list of density estimate objects from density estimation method.
    :rtype: list of :class: depends on the density estimation method chosen.
    """
    # allows modification of discretization
    discretization.local_to_global()
    
    # determine which sample sets to compute the density estimation
    if which_samples == "predict":
        this_sample_list = [discretization.get_output_sample_set()]
    elif which_samples == "observe":
        this_sample_list = [discretization.get_output_observed_set()]
    elif which_samples == "output":
        this_sample_list = [discretization.get_output_sample_set(), discretization.get_output_observed_set()]
    else:
        raise ValueError("Must specify the specific sample set to compute the density estimation.")
    
    # evaluates density estimation method on the samples
    output_densities = []
    output_density_type = None # this must be correct keyword for `sample.evaluate_pdf`
    
    if density_method == "kde":
        from scipy.stats import gaussian_kde
        for this_sample_set in this_sample_list:
            output_densities.append(gaussian_kde(this_sample_set.get_values().T, **method_args))
        output_density_type = "kde"
        
        
    elif density_method == "clustered_kde":
        for this_sample_set in this_sample_list:
            output_densities.append(generate_clustered_kdes(this_sample_set, **method_args))
        output_density_type = "clustered_kde"
        
    # save output densities to sample set
    for this_sample_set, dens in zip(this_sample_list,output_densities):
        this_sample_set.set_prob_type(output_density_type)
        this_sample_set.set_prob_parameters(dens)
    
    
    return output_density_type, output_densities

def generate_clustered_kdes(sample_set_base, kde_args=None):
    """
    Generates clustered kernel density estimate (similar to function `generate_output_kde`).
    *Note that initial weights must now be passed as weights through the dictionary kde_args.
    
    :param sample_set_base: Sample set object used to calculate density estimates
    :type sample_set_base: :class:`bet.sample.sample_set_base`
    :param kde_args: Dictionary of arguments passed to `scipy.stats.kde.gaussian_kde`. Note that
            this includes bandwidth parameters as well as weights for the kde estimate.
    :type kde_args: dict
    
    :returns: list of `gaussian_kde` and corresponding list of cluster weights
    :rtype: list, list
    """
    from scipy.stats import gaussian_kde
    
    if sample_set_base.get_region() is None:
        sample_set_base.set_region(np.array([0] * sample_set_base.check_num()))
    
    if sample_set_base.get_cluster_maps() is None:
        num_clusters = int(np.max(sample_set_base.get_region()) + 1)
    else:
        num_clusters = len(sample_set_base.get_cluster_maps())
    
    clustered_kdes = []
    cluster_weights = []
    num_obs = sample_set_base.check_num()
    for i in range(num_clusters):
        if sample_set_base.get_cluster_maps() is not None:
            if len(sample_set_base.get_cluster_maps()) > 1:
                clustered_kdes.append(gaussian_kde(sample_set_base.get_cluster_maps()[i].T, **kde_args))                
                cluster_weights.append(len(np.where(sample_set_base.get_region() == i)[0]) / num_obs)
            else:
                clustered_kdes.append(None)
                cluster_weights.append(None)
        else:
            values_pointer = np.where(sample_set_base.get_region() == i)[0]
            if len(values_pointer) > 1:
                clustered_kdes.append(gaussian_kde(sample_set_base.get_values()[values_pointer].T, **kde_args))
                cluster_weights.append(len(values_pointer)/num_obs)
            else:
                clustered_kdes.append(None)
                cluster_weights.append(None)
    
    return clustered_kdes, cluster_weights

def generate_output_kdes(discretization, bw_method=None):
    """
    Generate Kernel Density Estimates on predicted and observed output sample sets.

    :param discretization: Discretization used to calculate KDes
    :type discretization: :class:`bet.sample.discretization`
    :param bw_method: bandwidth method for `scipy.stats.gaussian_kde`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    :type bw_method: str

    :returns: prediction set, prediction kdes, observation set, observation kdes, number of clusters
    :rtype: :class:`bet.discretization.sample_set`, list, :class:`bet.discretization.sample_set`, list, int
    """
    from scipy.stats import gaussian_kde
    discretization.local_to_global()

    predict_set = discretization.get_output_sample_set()
    obs_set = discretization.get_output_observed_set()
    if predict_set.get_region() is None or obs_set.get_region() is None:
        predict_set.set_region(np.array([0] * predict_set.check_num()))
        obs_set.set_region(np.array([0] * obs_set.check_num()))

    if predict_set.get_cluster_maps() is None:
        num_clusters = int(np.max(predict_set.get_region()) + 1)
    else:
        num_clusters = len(predict_set.get_cluster_maps())

    predict_kdes = []
    obs_kdes = []
    for i in range(num_clusters):
        if predict_set.get_cluster_maps() is not None:
            if len(predict_set.get_cluster_maps()) > 1:
                if predict_set.get_weights_init() is None:
                    predict_kdes.append(gaussian_kde(predict_set.get_cluster_maps()[i].T, bw_method=bw_method))
                else:
                    predict_pointer = np.where(predict_set.get_region() == i)[0]
                    weights = predict_set.get_weights_init()[predict_pointer]
                    predict_kdes.append(gaussian_kde(predict_set.get_cluster_maps()[i].T, bw_method=bw_method,
                                                     weights=weights))
            else:
                predict_kdes.append(None)
        else:
            predict_pointer = np.where(predict_set.get_region() == i)[0]
            if len(predict_pointer) > 1:
                if predict_set.get_weights_init() is None:
                    predict_kdes.append(gaussian_kde(predict_set.get_values()[predict_pointer].T, bw_method=bw_method))
                else:
                    weights = predict_set.get_weights_init()[predict_pointer]
                    predict_kdes.append(gaussian_kde(predict_set.get_values()[predict_pointer].T, bw_method=bw_method,
                                                     weights=weights))
            else:
                predict_kdes.append(None)

        if obs_set.get_cluster_maps() is not None:
            if len(obs_set.get_cluster_maps()) > 1:
                obs_kdes.append(gaussian_kde(obs_set.get_cluster_maps()[i].T, bw_method=bw_method))
            else:
                obs_kdes.append(None)
        else:
            obs_pointer = np.where(obs_set.get_region() == i)[0]
            if len(obs_pointer) > 1:
                obs_kdes.append(gaussian_kde(obs_set.get_values()[obs_pointer].T, bw_method=bw_method))
            else:
                obs_kdes.append(None)
    return predict_kdes, obs_kdes, num_clusters


def invert(discretization, bw_method=None):
    """
    Solve the data consistent stochastic inverse problem, solving for input sample weights.

    :param discretization: Discretization on which to perform inversion.
    :type discretization: :class:`bet.sample.discretization`
    :param bw_method: bandwidth method for `scipy.stats.gaussian_kde`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    :type bw_method: str

    :return: acceptance rate, mean acceptance rate, pointers for samples to clusters
    :rtype: list, `np.ndarray`, list
    """
    predict_kdes, obs_kdes, num_clusters = generate_output_kdes(discretization, bw_method)
    predict_set = discretization.get_output_sample_set()

    rs = []
    r = []
    lam_ptr = []
    weights = np.zeros((discretization.get_output_sample_set().check_num(), ))
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
        weights[predict_pointer] = r[i]
        lam_ptr.append(predict_pointer)
    discretization.get_input_sample_set().set_weights(weights)
    return rs, r, lam_ptr


def invert_to_kde(discretization, bw_method=None):
    """
    Solve the data consistent stochastic inverse problem, solving for a weighted kernel density estimate.

    :param discretization: Discretization on which to perform inversion.
    :type discretization: :class:`bet.sample.discretization`
    :param bw_method: bandwidth method for `scipy.stats.gaussian_kde`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    :type bw_method: str

    :return: marginal probabilities and cluster weights
    :rtype: list, `np.ndarray`
    """
    from scipy.stats import gaussian_kde

    predict_kdes, obs_kdes, num_clusters = generate_output_kdes(discretization, bw_method)

    rs, r, lam_ptr = invert(discretization, bw_method)

    obs_set = discretization.get_output_observed_set()

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
                param_marginals[i].append(gaussian_kde(params[lam_ptr[j], i], weights=r[j], bw_method=bw_method))
            else:
                param_marginals[i].append(None)
    discretization.get_input_sample_set().set_prob_type("kde_marginals")
    discretization.get_input_sample_set().set_prob_parameters((param_marginals, cluster_weights))
    print('Diagnostic for clusters [sample average of ratios in each cluster]: ', rs)
    return param_marginals, cluster_weights


def invert_rejection_sampling(discretization, bw_method=None):
    """
    Solve the data consistent stochastic inverse problem by rejection sampling.

    :param discretization: Discretization on which to perform inversion.
    :type discretization: :class:`bet.sample.discretization`
    :param bw_method: bandwidth method for `scipy.stats.gaussian_kde`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    :type bw_method: str

    :return: sample set containing samples
    :rtype: :class:`bet.sample.sample_set`
    """
    predict_kdes, obs_kdes, num_clusters = generate_output_kdes(discretization, bw_method)

    rs, r, lam_ptr = invert(discretization, bw_method)

    discretization.get_input_sample_set().local_to_global()
    new_vals = []
    for i in range(num_clusters):
        check = np.random.uniform(low=0, high=1, size=r[i].size)  # create random uniform weights to check r against
        new_r = r[i] / np.max(r[i])  # normalize weights
        idx = np.where(new_r >= check)[0]  # rejection criterion
        new_vals.append(discretization.get_input_sample_set().get_values()[lam_ptr[i][idx]])
    vals = np.vstack(new_vals)
    new_set = bet.sample.sample_set(discretization.get_input_sample_set().get_dim())
    new_set.set_values(vals)
    n = vals.shape[0]
    probs = np.ones((n, )) / float(n)
    new_set.set_probabilities(probs)
    domain = []
    for i in range(new_set.get_dim()):
        x_max = np.max(vals[:, i])
        x_min = np.min(vals[:, i])
        domain.append([x_min, x_max])
    domain = np.array(domain)
    new_set.set_domain(domain)
    new_set.global_to_local()

    return new_set


def invert_to_gmm(discretization, bw_method=None):
    """
    Solve the data consistent stochastic inverse problem, solving for a Gaussian mixture model.

    :param discretization: Discretization on which to perform inversion.
    :type discretization: :class:`bet.sample.discretization`
    :param bw_method: bandwidth method for `scipy.stats.gaussian_kde`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    :type bw_method: str

    :return: means, covariances, and weights for Gaussians
    :rtype: list, list, list
    """
    def weighted_mean_and_cov(x, weights):
        sum_weights = np.sum(weights)
        mean1 = []
        for i in range(x.shape[1]):
            mean1.append((np.sum(x[:, i] * weights)/sum_weights))
        mean1 = np.array(mean1)

        cov1 = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            val = x[i, :] - mean1
            cov1 += weights[i] * np.outer(val, val)
        cov1 = cov1 / sum_weights
        return mean1, cov1

    predict_kdes, obs_kdes, num_clusters = generate_output_kdes(discretization, bw_method)

    rs, r, lam_ptr = invert(discretization, bw_method)

    obs_set = discretization.get_output_observed_set()

    # Compute multivariate normal for each cluster
    means = []
    covariances = []
    cluster_weights = []
    num_obs = obs_set.check_num()

    input_dim = discretization.get_input_sample_set().get_dim()
    params = discretization.get_input_sample_set().get_values()

    for i in range(num_clusters):
        cluster_weights.append(len(np.where(obs_set.get_region() == i)[0]) / num_obs)
        if r[i] is not None:
            mean, cov = weighted_mean_and_cov(params[lam_ptr[i], :], r[i])
            means.append(mean)
            covariances.append(cov)
        else:
            means.append(None)
            covariances.append(None)

    discretization.get_input_sample_set().set_prob_type("gmm")
    discretization.get_input_sample_set().set_prob_parameters((means, covariances, cluster_weights))
    print('Diagnostic for clusters [sample average of ratios in each cluster]: ', rs)
    return means, covariances, cluster_weights


def invert_to_multivariate_gaussian(discretization, bw_method=None):
    """
    Solve the data consistent stochastic inverse problem, solving for a multivariate Gaussian.

    :param discretization: Discretization on which to perform inversion.
    :type discretization: :class:`bet.sample.discretization`
    :param bw_method: bandwidth method for `scipy.stats.gaussian_kde`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    :type bw_method: str

    :return: marginal probabilities and cluster weights
    :rtype: list, `np.ndarray`
    """
    def weighted_mean_and_cov(x, weights):
        sum_weights = np.sum(weights)
        mean1 = []
        for i in range(x.shape[1]):
            mean1.append((np.sum(x[:, i] * weights)/sum_weights))
        mean1 = np.array(mean1)

        cov1 = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            val = x[i, :] - mean1
            cov1 += weights[i] * np.outer(val, val)
        cov1 = cov1 / sum_weights
        return mean1, cov1

    predict_kdes, obs_kdes, num_clusters = generate_output_kdes(discretization, bw_method)

    rs, r, lam_ptr = invert(discretization, bw_method)

    obs_set = discretization.get_output_observed_set()

    # Compute multivariate normal
    cluster_weights = []
    num_obs = obs_set.check_num()

    params = discretization.get_input_sample_set().get_values()
    total_weights = np.zeros((discretization.get_input_sample_set().check_num(), ))

    for i in range(num_clusters):
        cluster_weights.append(len(np.where(obs_set.get_region() == i)[0]) / num_obs)
        total_weights[lam_ptr[i]] = r[i] * cluster_weights[i]
    mean, cov = weighted_mean_and_cov(params, total_weights)
    means = [mean]
    covariances = [cov]
    cluster_weights = [1.0]

    discretization.get_input_sample_set().set_prob_type("gmm")
    discretization.get_input_sample_set().set_prob_parameters((means, covariances, cluster_weights))
    print('Diagnostic for clusters [sample average of ratios in each cluster]: ', rs)
    return means, covariances, cluster_weights


def invert_to_random_variable(discretization, rv, num_reweighted=10000, bw_method=None):
    """
    Solve the data consistent stochastic inverse problem, fitting a random variable.

    `rv` can take multiple types of formats depending on type of distribution.

    A string is used for the same distribution with default parameters in each dimension.
    ex. rv = 'uniform' or rv = 'beta'

    A list or tuple of length 2 is used for the same distribution with  fixed user-defined parameters in each dimension
    as a dictionary.
    ex. rv = ['uniform', {'floc':-2, 'fscale':5}] or rv = ['beta', {'fa': 2, 'fb':5, 'floc':-2, 'fscale':5}]

    A list of length dim which entries of lists or tuples of length 2 is used for different distributions with fixed
    user-defined parameters in each dimension as a dictionary.
    ex. rv = [['uniform', {'floc':-2, 'fscale':5}],
              ['beta', {'fa': 2, 'fb':5, 'floc':-2, 'fscale':5}]]

    :param discretization: Discretization on which to perform inversion.
    :type discretization: :class:`bet.sample.discretization`
    :param rv: Type and parameters for continuous random variables.
    :type rv: str, list, or tuple
    :param num_reweighted: number of reweighted samples for fitting
    :type num_reweighted: int
    :param bw_method: bandwidth method for `scipy.stats.gaussian_kde`.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    :type bw_method: str

    :return: marginal probabilities and cluster weights
    :rtype: list, `np.ndarray`
    """
    import scipy.stats as stats

    dim = discretization.get_input_sample_set().get_dim()
    if type(rv) is str:
        rv = [[rv, {}]] * dim
    elif type(rv) in (list, tuple):
        if len(rv) == 2 and type(rv[0]) is str and type(rv[1]) is dict:
            rv = [rv] * dim
        elif len(rv) != dim:
            raise bet.sample.dim_not_matching("rv has fewer entries than the dimension.")
    else:
        raise bet.sample.wrong_input("rv must be a string, list, or tuple.")

    predict_kdes, obs_kdes, num_clusters = generate_output_kdes(discretization, bw_method)

    rs, r, lam_ptr = invert(discretization, bw_method)

    obs_set = discretization.get_output_observed_set()

    # Compute multivariate normal
    cluster_weights = []
    num_obs = obs_set.check_num()

    params = discretization.get_input_sample_set().get_values()
    total_weights = np.zeros((discretization.get_input_sample_set().check_num(), ))

    for i in range(num_clusters):
        cluster_weights.append(len(np.where(obs_set.get_region() == i)[0]) / num_obs)
        total_weights[lam_ptr[i]] = r[i] * cluster_weights[i]
    total_weights = np.round(num_reweighted * total_weights/np.sum(total_weights)).astype(int)
    reweighted_vals = np.repeat(params, total_weights, axis=0)

    prob_params = []
    for i in range(dim):
        pp = [rv[i][0], {}]
        rv_continuous = getattr(stats, rv[i][0])
        A = rv_continuous.fit(reweighted_vals[:, i], **rv[i][1])
        if len(A) == 2:
            pp[1]['loc'] = A[0]
            pp[1]['scale'] = A[1]
        elif len(A) == 3:
            pp[1]['a'] = A[0]
            pp[1]['loc'] = A[1]
            pp[1]['scale'] = A[2]
        elif len(A) == 4:
            pp[1]['a'] = A[0]
            pp[1]['b'] = A[1]
            pp[1]['loc'] = A[2]
            pp[1]['scale'] = A[3]
        else:
            raise bet.sample.wrong_input("Type of random variable is not currently supported.")
        prob_params.append(pp)
    discretization.get_input_sample_set().set_prob_type('rv')
    discretization.get_input_sample_set().set_prob_parameters(prob_params)
    print('Random variable fits: ', prob_params)
    print('Diagnostic for clusters [sample average of ratios in each cluster]: ', rs)
    return prob_params
