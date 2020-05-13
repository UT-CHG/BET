from scipy.stats import entropy as kl_div
import bet.postProcess.compareP as compP
from helpers import *

"""
The ``helpers.py`` file contains functions that define
sample sets of an arbitrary dimension with probabilities
uniformly distributed in a hypercube of sidelength ``delta``.
The hypercube can be in three locations:
- corner at [0, 0, ..., 0]  in ``unit_bottom_set``
- centered at [0.5, 0.5, ... 0.5] in ``unit_center_set``
- corner in [1, 1, ..., 1] in `` unit_top_set``

and the number of samples will determine the fidelity of the
approximation since we are using voronoi-cell approximations.
"""
num_samples1 = 50
num_samples2 = 50
delta1 = 0.5  # width of measure's support per dimension
delta2 = 0.45
dim = 2
# define two sets that will be compared
set1 = unit_center_set(dim, num_samples1, delta1)
set2 = unit_center_set(dim, num_samples2, delta2)

# choose a reference sigma-algebra to compare both solutions
# against (using nearest-neighbor query).
num_comparison_samples = 2000
# the compP.compare method instantiates the compP.comparison class.
mm = compP.compare(set1, set2)  # initialize metric

# Use existing common library functions

# Use a function of your own!


def inftynorm(x, y):
    """
    Infinity norm between two vectors.
    """
    return np.max(np.abs(x - y))


mm.set_compare_set(compare_set=10000, compare_factor=0.1)

print(mm.distance('tv'))
print(mm.distance(inftynorm, normalize=False))
print(mm.distance('mink', w=0.5, p=1))
print(mm.distance('hell'))


"""
mm.set_left(unit_center_set(2, 1000, delta / 2))
mm.set_right(unit_center_set(2, 1000, delta))
print([mm.value(kl_div),
       mm.value(inftynorm),
       mm.value('tv'),
       mm.value('totvar'),
       mm.value('mink', w=0.5, p=1),
       mm.value('norm'),
       mm.value('sqhell'),
       mm.value('hell'),
       mm.value('hellinger')])
"""
