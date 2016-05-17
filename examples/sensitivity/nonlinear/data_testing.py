import numpy as np
import matplotlib.pyplot as plt
# Data_dim = 3
# rand_int = 124

np.random.seed(0)
rand_int_list = [int(np.round(np.random.random(1) * i)) for i in range(0,1000,100)]
np.random.seed(0)
# rand_int_list += [int(np.round(np.random.random(1) * i)) for i in range(5,1005,100)]
print rand_int_list, '\n'
# selection = range(8) # num_anchors 1, 2, 5, 10, 25, 50, 75, 100 (0 ... 7)
selection = range(0,5)
for Data_dim in [3, 5, 7, 9]:
    highest_prob = []
    for rand_int in rand_int_list:
        highest_prob.append(np.load('prob_reduction_results_seed%d_dimD%d.npy'%(rand_int, Data_dim)))
    highest_prob = np.array(highest_prob)
    for i in range(len(highest_prob)):
        print highest_prob[i][:,[0,2]], '\n'
        plt.plot(highest_prob[i][selection,0], highest_prob[i][selection,2])
    plt.ylabel('Number of Samples with P>0 \n (Out of 100,000)')
    plt.xlabel('Number of Anchor Points')
    plt.title('%d Random Maps with %d Random Seeds'%(Data_dim, len(highest_prob)))
    plt.axis([0, highest_prob[i][max(selection),0], 2E3, 2E4])
    plt.show()
