# Copyright (C) 2014-2020 The BET Development Team


import bet.sampling.basicSampling as bsam
import bet.postProcess.compareP as compP

"""
Compare marginals of two probability measures based on random variables with certain properties.
"""

# Initialize two sample sets
set1 = bsam.random_sample_set(rv=[['beta', {'loc': 0, 'scale': 2, 'a': 3, 'b': 2}],
                              ['beta', {'loc': 0, 'scale': 2, 'a': 3, 'b': 2}]],
                              input_obj=2, num_samples=300)
set2 = bsam.random_sample_set(rv=[['beta', {'loc': 0, 'scale': 2, 'a': 2, 'b': 3}],
                              ['beta', {'loc': 0, 'scale': 2, 'a': 2, 'b': 3}]],
                              input_obj=2, num_samples=300)

# Initialize metric
mm = compP.compare(set1, set2, set2_init=True, set1_init=True)
mm.set_compare_set()

# Test different distance metrics with discrete distances and by integrating with quadrature.
print(mm.distance('tv'))
print('Total Variation')
print(mm.distance_marginal(i=0, functional='tv', normalize=False))
print(mm.distance_marginal_quad(i=0, functional='tv'))

print('KL Divergence')
print(mm.distance_marginal(i=0, functional='kl', normalize=False))
print(mm.distance_marginal_quad(i=0, functional='kl'))

print('Hellinger Distance')
print(mm.distance_marginal(i=0, functional='hell', normalize=False))
print(mm.distance_marginal_quad(i=0, functional='hell'))

print('Euclidean Norm')
print(mm.distance_marginal(i=0, functional='2', normalize=False))
print(mm.distance_marginal_quad(i=0, functional='2'))
