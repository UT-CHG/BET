

# define samples in parameter space.

# define a set of K anchor points

# perform nearest neighbor searches.

# want an object with lists of indices into each anchor point.

# feed each list of indices into samples and data, perform chooseQoIs

# for each anchor point, record best_sets (accessing [0] for the best one).

# define a quantity of interest map of Lambda_dim where we run nearest neighbor
# on the point against anchor points, then access the best QoI for that anchor point,
# use it to access the appropriate quantity of interest component maps.

# map parameter space under this new map.

# solve inverse problem with this piecwise-defined quantity of interest map.
