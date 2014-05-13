"""
This module provides methods for plotting probabilities. 
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_voronoi_probs(P_samples,
                       samples,
                       lam_domain,
                       nbins=20,
                       plot_surface = False):
    """
    This makes plots of the joint probabilies of
    input probability measure defined by P_samples for 2d cases. post_process - 
    is an input that only applies to the 2d case  w.r.t. the Voronoi cells.

    :param P_samples: Probabilities.
    :type P_samples: :class:`~numpy.ndarray` of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)
    :param nbins: Number of bins in each direction.
    :type nbins: :int

    """
    lam_dim=lam_domain.shape[0]
    
    if lam_dim == 2: # Plot Voronoi tesselations, otherwise plot 2d 
        #projections/marginals of the joint inverse measure
        num_samples = samples.shape[0]
        #Add fake samples outside of lam_domain to close Voronoi 
        #tesselations at infinity

def plot_marginal_probs(P_samples,
                        samples,
                        lam_domain,
                        nbins=20,
                        filename = "file",
                        plot_surface= False):
        
    """
    This makes plots of every pair of marginals (or joint in 2d case) of
    input probability measure defined by P_samples on a rectangular grid.

    :param P_samples: Probabilities.
    :type P_samples: :class:`~numpy.ndarray` of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:`~numpy.ndarray` of shape (ndim, 2)
    :param nbins: Number of bins in each direction.
    :type nbins: :int

    """
    if plot_surface:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    num_samples = samples.shape[0]
    num_dim = samples.shape[1]
    if isinstance(nbins, int):
        nbins =nbins*np.ones(num_dim, dtype=np.int)
    # histograms=[]
    bins = []
    for i in range(num_dim):
        bins.append(np.linspace(lam_domain[i][0], lam_domain[i][1], nbins[i]+1))
    bin_ptr = np.zeros((num_samples, num_dim), dtype=np.int)
    # for i in range(num_dim):
    #     hist, bin = np.histogram(samples[:,i],nbins)
    #     histograms.append(hist)
    #     bins.append(bin)
    #     for j in range(num_samples):
    #         bin_ptr[j][i]
    #histograms, bins = np.histogramdd(samples, nbins)
    for i in range(num_samples):
        for j in range(num_dim):
            go = True
            k = 0
            while go: #for k in range(nbins[j]):
                if samples[i][j] <= bins[j][k+1]:
                    bin_ptr[i][j] = k
                    go = False
                else:
                    k += 1
                    
    marginals = {}
    for i in range(num_dim):
        for j in range(i+1, num_dim):
            marg = np.zeros((nbins[i]+1, nbins[j]+1))
            for k in range(num_samples):
                marg[bin_ptr[k][i]][bin_ptr[k][j]] += P_samples[k]
            marginals[(i,j)] = marg

    # for k,(i,j) in enumerate(marginals.keys()):
    #     fig = plt.figure(k)
    #     ax = fig.add_subplot(111)
    #     X = bins[i]
    #     Y = bins[j]
    #     X,Y = np.meshgrid(X,Y)
    #     ax.plot_grid(X, Y, marginals[(i,j)])


    if plot_surface:
        for k,(i,j) in enumerate(marginals.keys()):
            fig = plt.figure(k)
            ax = fig.gca(projection='3d')
            X = bins[i]
            Y = bins[j]
            X,Y = np.meshgrid(X,Y)
            # import pdb
            # pdb.set_trace()

            #zz=np.vstack((marginals[(i,j)], np.zeros((nbins[i],))))#,np.zeros((nbins[j]+1,)).transpose()))
            #pdb.set_trace()
            surf = ax.plot_surface(X, Y, marginals[(i,j)], rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            fig.colorbar(surf, shrink=0.5, aspect=5)
            fig.savefig(filename + "_surf_"+ `i` + "_" +`j` + ".eps")
            plt.show()
