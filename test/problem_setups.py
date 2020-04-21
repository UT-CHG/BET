import bet.sample as samp
import bet.sampling.basicSampling as bsam
import numpy as np
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.calculateP.dataConsistent as dataConsistent


def random_voronoi(rv='uniform', dim=1, out_dim=1, num_samples=1000, globalize=True, level=1):
    if level == 1:
        return bsam.random_sample_set(rv, dim, num_samples, globalize)
    elif level == 2:
        def my_model(samples):
            A = np.eye(dim, out_dim)
            return np.dot(samples, A)
        sampler = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler.random_sample_set(rv, dim, num_samples, globalize)
        disc = sampler.compute_qoi_and_create_discretization()
        input_samples = disc.get_input_sample_set()
        input_samples.estimate_volume_mc()

        param_ref = np.array([0.5] * dim)
        q_ref = my_model(param_ref)
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
            data_set=disc, Q_ref=q_ref, rect_scale=0.25,
            cells_per_dimension=1)
        # calculate probabilities
        calculateP.prob(disc)
        return disc

def random_kde(rv='uniform', dim=1, out_dim=1, num_samples=1000, globalize=True, level=1, rv2="norm"):
    if level == 1:
        return bsam.random_sample_set(rv, dim, num_samples, globalize)
    elif level == 2:
        def my_model(samples):
            A = np.eye(dim, out_dim)
            return np.dot(samples, A)

        sampler1 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler1.random_sample_set(rv, dim, num_samples, globalize)
        disc1 = sampler1.compute_qoi_and_create_discretization()

        sampler2 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler2.random_sample_set(rv2, dim, num_samples, globalize)
        disc2 = sampler1.compute_qoi_and_create_discretization()

        disc1.set_output_probability_set(disc2.get_output_sample_set())
        dataConsistent.dc_inverse_kde(disc1)
        return disc1, disc2


def random_gmm(rv='uniform', dim=1, out_dim=1, num_samples=1000, globalize=True, level=1, rv2="norm"):
    if level == 1:
        return bsam.random_sample_set(rv, dim, num_samples, globalize)
    elif level == 2:
        def my_model(samples):
            A = np.eye(dim, out_dim)
            return np.dot(samples, A)

        sampler1 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler1.random_sample_set(rv, dim, num_samples, globalize)
        disc1 = sampler1.compute_qoi_and_create_discretization()

        sampler2 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler2.random_sample_set(rv2, dim, num_samples, globalize)
        disc2 = sampler1.compute_qoi_and_create_discretization()

        disc1.set_output_probability_set(disc2.get_output_sample_set())
        dataConsistent.dc_inverse_gmm(disc1)
        return disc1, disc2

def random_gmm(rv='uniform', dim=1, out_dim=1, num_samples=1000, globalize=True, level=1, rv2="norm"):
    if level == 1:
        return bsam.random_sample_set(rv, dim, num_samples, globalize)
    elif level == 2:
        def my_model(samples):
            A = np.eye(dim, out_dim)
            return np.dot(samples, A)

        sampler1 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler1.random_sample_set(rv, dim, num_samples, globalize)
        disc1 = sampler1.compute_qoi_and_create_discretization()

        sampler2 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler2.random_sample_set(rv2, dim, num_samples, globalize)
        disc2 = sampler1.compute_qoi_and_create_discretization()

        disc1.set_output_probability_set(disc2.get_output_sample_set())
        dataConsistent.dc_inverse_gmm(disc1)
        return disc1, disc2

def random_multivariate_gaussian(rv='uniform', dim=1, out_dim=1, num_samples=1000,
                                 globalize=True, level=1, rv2="norm"):
    if level == 1:
        return bsam.random_sample_set(rv, dim, num_samples, globalize)
    elif level == 2:
        def my_model(samples):
            A = np.eye(dim, out_dim)
            return np.dot(samples, A)

        sampler1 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler1.random_sample_set(rv, dim, num_samples, globalize)
        disc1 = sampler1.compute_qoi_and_create_discretization()

        sampler2 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler2.random_sample_set(rv2, dim, num_samples, globalize)
        disc2 = sampler1.compute_qoi_and_create_discretization()

        disc1.set_output_probability_set(disc2.get_output_sample_set())
        dataConsistent.dc_inverse_multivariate_gaussian(disc1)
        return disc1, disc2

def random_rv(rv='uniform', dim=1, out_dim=1, num_samples=1000,
                                 globalize=True, level=1, rv2="norm"):
    if level == 1:
        return bsam.random_sample_set(rv, dim, num_samples, globalize)
    elif level == 2:
        def my_model(samples):
            A = np.eye(dim, out_dim)
            return np.dot(samples, A)

        sampler1 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler1.random_sample_set(rv, dim, num_samples, globalize)
        disc1 = sampler1.compute_qoi_and_create_discretization()

        sampler2 = bsam.sampler(lb_model=my_model, error_estimates=False, jacobians=False)
        sampler2.random_sample_set(rv2, dim, num_samples, globalize)
        disc2 = sampler1.compute_qoi_and_create_discretization()

        disc1.set_output_probability_set(disc2.get_output_sample_set())
        dataConsistent.dc_inverse_random_variable(disc1, rv="norm")
        return disc1, disc2



