"""
This module provides methods for plotting probabilities. 
"""

import matplotlib.pyplot as plt
import numpy as np
import copy
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def get_global_values(array):
    """
    Concatenates local array into global arrays.

    :param array: Array.
    :type P_samples: :class:'~numpy.ndarray' 
    """

    array = comm.allgather(array, array)   
    return np.vstack(array)

def plot_voronoi_probs(P_samples, samples, lam_domain, nbins=20,
        plot_surface=False):
    """
    This makes plots of the joint probabilies of input probability measure
    defined by P_samples for 2d cases. post_process - is an input that only
    applies to the 2d case  w.r.t. the Voronoi cells.

    :param P_samples: Probabilities.
    :type P_samples: :class:'~numpy.ndarray' of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:'~numpy.ndarray' of shape (num_samples, ndim)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:'~numpy.ndarray' of shape (ndim, 2)
    :param nbins: Number of bins in each direction.
    :type nbins: :int

    """
    lam_dim = lam_domain.shape[0]
    
    if lam_dim == 2: # Plot Voronoi tesselations, otherwise plot 2d 
        #projections/marginals of the joint inverse measure
        num_samples = samples.shape[0]
        #Add fake samples outside of lam_domain to close Voronoi 
        #tesselations at infinity

def plot_marginal_probs(P_samples, samples, lam_domain, nbins=20,
        filename="file", lam_true=None, plot_surface=False, interactive=True,
        lambda_label=None):
        
    """
    This makes plots of every pair of marginals (or joint in 2d case) of
    input probability measure defined by P_samples on a rectangular grid.

    :param P_samples: Probabilities.
    :type P_samples: :class:'~numpy.ndarray' of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:'~numpy.ndarray' of shape (num_samples, ndim)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:'~numpy.ndarray' of shape (ndim, 2)
    :param nbins: Number of bins in each direction.
    :type nbins: :int

    """
    if plot_surface:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

    num_samples = samples.shape[0]
    num_dim = samples.shape[1]

    # Make list of bins if only an integer is given
    if isinstance(nbins, int):
        nbins = nbins*np.ones(num_dim, dtype=np.int)
    
    # Create bins
    bins = []
    for i in range(num_dim):
        bins.append(np.linspace(lam_domain[i][0], lam_domain[i][1], nbins[i]+1))
    bin_ptr = np.zeros((num_samples, num_dim), dtype=np.int)
    # Bin samples
    for j in range(num_dim):
        bin_ptr[:, j] = np.searchsorted(bins[j], samples[:, j])
    bin_ptr -= 1
         
    # Calculate marginal probabilities 
    marginals = {}
    for i in range(num_dim):
        for j in range(i+1, num_dim):
            marg = np.zeros((nbins[i]+1, nbins[j]+1))
            # This may be sped up with logical indices
            for k in range(num_samples):
                marg[bin_ptr[k][i]][bin_ptr[k][j]] += P_samples[k]
            marg = comm.allreduce(marg, marg, op=MPI.SUM)
            marginals[(i, j)] = marg

    if rank == 0:
        pairs = copy.deepcopy(marginals.keys())
        pairs.sort()
        for k, (i, j) in enumerate(pairs):
            fig = plt.figure(k)
            ax = fig.add_subplot(111)
            X = bins[i]
            Y = bins[j]
            X, Y = np.meshgrid(X, Y, indexing='ij')
            quadmesh = ax.pcolormesh(X, Y, marginals[(i, j)], cmap=cm.coolwarm)

            if lam_true != None:
                ax.plot(lam_true[i], lam_true[j], 'ko', markersize=10)
            if lambda_label == None:
                label1 = '$lambda_{' + 'i+1' + '}$'
                label2 = '$lambda_{' + 'j+1' + '}$'
            else:
                label1 = lambda_label[i]
                label2 = lambda_label[j]
            ax.set_xlabel(label1) 
            ax.set_ylabel(label2)
            fig.colorbar(quadmesh, ax=ax, label='$P$')
            plt.axis([lam_domain[i][0], lam_domain[i][1], lam_domain[j][0],
                lam_domain[j][1]]) 
            fig.savefig(filename + "_2D_" + 'i' + "_" + 'j' + ".eps")
            if interactive:
                plt.show()
 
        if plot_surface:
            for k, (i, j) in enumerate(pairs):
                fig = plt.figure(k)
                ax = fig.gca(projection='3d')
                X = bins[i]
                Y = bins[j]
                X, Y = np.meshgrid(X, Y, indexing='ij')
                surf = ax.plot_surface(X, Y, marginals[(i, j)], rstride=1,
                        cstride=1, cmap=cm.coolwarm, linewidth=0,
                        antialiased=False)
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                ax.set_xlabel('$lambda_{' + 'i+1' + '}$') 
                ax.set_ylabel('$lambda_{' + 'j+1' + '}$')
                ax.set_zlabel('$P$')
                plt.backgroundcolor = 'w'
                fig.colorbar(surf, shrink=0.5, aspect=5, label=r'$P$')
                fig.savefig(filename + "_surf_"+ 'i' + "_" +'j' + ".eps")
                if interactive:
                    plt.show()
