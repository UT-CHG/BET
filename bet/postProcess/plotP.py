"""
This module provides methods for plotting probabilities. 
"""

from bet.Comm import *
import matplotlib.pyplot as plt
import numpy as np
import copy, math


def calculate_1D_marginal_probs(P_samples, samples, lam_domain, nbins=20):
        
    """
    This calculates every single marginal of
    input probability measure defined by P_samples on a 1D grid.

    :param P_samples: Probabilities.
    :type P_samples: :class:'~numpy.ndarray' of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:'~numpy.ndarray' of shape (num_samples, ndim)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:'~numpy.ndarray' of shape (ndim, 2)
    :param nbins: Number of bins in each direction.
    :type nbins: :int or :class:'~numpy.ndarray' of shape (ndim,)
    :rtype: tuple
    :returns: (bins, marginals)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1)
    num_samples = samples.shape[0]
    num_dim = samples.shape[1]

    # Make list of bins if only an integer is given
    if isinstance(nbins, int):
        nbins = nbins*np.ones(num_dim, dtype=np.int)
 
    # Create bins
    bins = []
    for i in range(num_dim):
        bins.append(np.linspace(lam_domain[i][0], lam_domain[i][1], nbins[i]+1))
        
    # Calculate marginals
    marginals = {}
    for i in range(num_dim):
        [marg, _] = np.histogram(samples[:,i], bins=bins[i], weights=P_samples)
        marg_temp = np.copy(marg)
        comm.Allreduce([marg, MPI.DOUBLE], [marg_temp, MPI.DOUBLE], op=MPI.SUM)
        marginals[i] = marg_temp

    return (bins, marginals)

def calculate_2D_marginal_probs(P_samples, samples, lam_domain, nbins=20):
        
    """
    This calculates every pair of marginals (or joint in 2d case) of
    input probability measure defined by P_samples on a rectangular grid.

    :param P_samples: Probabilities.
    :type P_samples: :class:'~numpy.ndarray' of shape (num_samples,)
    :param samples: The samples in parameter space for which the model was run.
    :type samples: :class:'~numpy.ndarray' of shape (num_samples, ndim)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:'~numpy.ndarray' of shape (ndim, 2)
    :param nbins: Number of bins in each direction.
    :type nbins: :int or :class:'~numpy.ndarray' of shape (ndim,)
    :rtype: tuple
    :returns: (bins, marginals)

    """
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1)
    num_samples = samples.shape[0]
    num_dim = samples.shape[1]

    # Make list of bins if only an integer is given
    if isinstance(nbins, int):
        nbins = nbins*np.ones(num_dim, dtype=np.int)

    # Create bins
    bins = []
    for i in range(num_dim):
        bins.append(np.linspace(lam_domain[i][0], lam_domain[i][1], nbins[i]+1))
    
    # Calculate marginals
    marginals = {}
    for i in range(num_dim):
        for j in range(i+1, num_dim):
            (marg, _) = np.histogramdd(samples[:,[i,j]], bins = [bins[i], bins[j]], weights = P_samples)
            marg=np.ascontiguousarray(marg)
            marg_temp = np.copy(marg)
            comm.Allreduce([marg, MPI.DOUBLE], [marg_temp, MPI.DOUBLE],
                    op=MPI.SUM) 
            marginals[(i, j)] = marg_temp

    return (bins, marginals)

def plot_1D_marginal_probs(marginals, bins, lam_domain,
        filename="file", lam_ref=None, interactive=False,
        lambda_label=None):
        
    """
    This makes plots of every single marginal probability of
    input probability measure defined by P_samples on a 1D  grid.

    :param marginals: 1D marginal probabilities
    :type marginals: dictionary with int as keys and :class:'~numpy.ndarray' of
        shape (nbins+1,) as values :param bins: Endpoints of bins used in
        calculating marginals
    :type bins: :class:'~numpy.ndarray' of shape (nbins+1,)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:'~numpy.ndarray' of shape (ndim, 2)
    :param filename: Prefix for output files.
    :type filename: str
    :param lam_ref: True parameters.
    :type lam_ref: :class:'~numpy.ndarray' of shape (ndim,) or None
    :param interactive: Whether or not to display interactive plots.
    :type interactive: boolean
    :param lambda_label: Label for each parameter for plots.
    :type lambda_label: list of length nbins of strings or None

    """
    if rank == 0:
        index = copy.deepcopy(marginals.keys())
        index.sort()
        for i in index:
            x_range = np.linspace(lam_domain[i, 0], lam_domain[i, 1],
                    len(bins[i])-1) 
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            ax.plot(x_range, marginals[i]/(bins[i][1]-bins[i][0]))
            if lam_ref != None:
                ax.plot(lam_ref[i], 0.0, 'ko', markersize=10)
            if lambda_label == None:
                label1 = r'$\lambda_{' + str(i+1) + '}$'
            else:
                label1 = lambda_label[i]
            ax.set_xlabel(label1) 
            ax.set_ylabel(r'$\rho$')
            fig.savefig(filename + "_1D_" + str(i) + ".eps", transparent=True)
            if interactive:
                plt.show()
            else:
                plt.close()
            plt.clf()

def plot_2D_marginal_probs(marginals, bins, lam_domain,
        filename="file", lam_ref=None, plot_surface=False, interactive=False,
        lambda_label=None):
        
    """
    This makes plots of every pair of marginals (or joint in 2d case) of
    input probability measure defined by P_samples on a rectangular grid.

    :param marginals: 2D marginal probabilities
    :type marginals: dictionary with tuples of 2 integers as keys and
        :class:'~numpy.ndarray' of shape (nbins+1,) as values 
    :param bins: Endpoints of bins used in calculating marginals
    :type bins: :class:'~numpy.ndarray' of shape (nbins+1,2)
    :param lam_domain: The domain for each parameter for the model.
    :type lam_domain: :class:'~numpy.ndarray' of shape (ndim, 2)
    :param filename: Prefix for output files.
    :type filename: str
    :param lam_ref: True parameters.
    :type lam_ref: :class:'~numpy.ndarray' of shape (ndim,) or None
    :param interactive: Whether or not to display interactive plots.
    :type interactive: boolean
    :param lambda_label: Label for each parameter for plots.
    :type lambda_label: list of length nbins of strings or None

    """
    from matplotlib import cm
    if plot_surface:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
    if rank == 0:
        pairs = copy.deepcopy(marginals.keys())
        pairs.sort()
        for k, (i, j) in enumerate(pairs):
            fig = plt.figure(k)
            ax = fig.add_subplot(111)
            boxSize = (bins[i][1]-bins[i][0])*(bins[j][1]-bins[j][0])
            quadmesh = ax.imshow(marginals[(i, j)].transpose()/boxSize,
                    interpolation='bicubic', cmap=cm.jet, 
                    extent=[lam_domain[i][0], lam_domain[i][1],
                    lam_domain[j][0], lam_domain[j][1]], origin='lower',
                    vmax=marginals[(i, j)].max()/boxSize, vmin=marginals[(i,
                        j)].min()/boxSize, aspect='auto')
            if lam_ref != None:
                ax.plot(lam_ref[i], lam_ref[j], 'ko', markersize=10)
            if lambda_label == None:
                label1 = r'$\lambda_{' + str(i+1) + '}$'
                label2 = r'$\lambda_{' + str(j+1) + '}$'
            else:
                label1 = lambda_label[i]
                label2 = lambda_label[j]
            ax.set_xlabel(label1) 
            ax.set_ylabel(label2)
            label_cbar = r'$\rho_{\lambda_{' + str(i+1) + '}, ' 
            label_cbar += r'\lambda_{' + str(j+1) + '}' + '}$ (Lesbesgue)'
            fig.colorbar(quadmesh, ax=ax, label=label_cbar)
            plt.axis([lam_domain[i][0], lam_domain[i][1], lam_domain[j][0],
                lam_domain[j][1]]) 
            fig.savefig(filename + "_2D_" + str(i) + "_" + str(j) + ".eps", transparent=True)
            if interactive:
                plt.show()
            else:
                plt.close()
 
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
                ax.set_xlabel(r'$\lambda_{' + str(i+1) + '}$') 
                ax.set_ylabel(r'$\lambda_{' + str(j+1) + '}$')
                ax.set_zlabel(r'$P$')
                plt.backgroundcolor = 'w'
                fig.colorbar(surf, shrink=0.5, aspect=5, label=r'$P$')
                fig.savefig(filename + "_surf_"+str(i)+"_"+str(j)+".eps", transparent=True)
                if interactive:
                    plt.show()
                else:
                    plt.close()
                plt.clf()

def smooth_marginals_1D(marginals, bins, sigma=10.0):
    """
    This function smooths 1D marginal probabilities.

    :param marginals: 1D marginal probabilities
    :type marginals: dictionary with int as keys and :class:'~numpy.ndarray' of
        shape (nbins+1,) as values :param bins: Endpoints of bins used in
        calculating marginals
    :type bins: :class:'~numpy.ndarray' of shape (nbins+1,)
    :param sigma: Smoothing parameter in each direction.
    :type sigma: :float or :class:'~numpy.ndarray' of shape (ndim,)
    :rtype: dict
    :returns: marginals_smooth
    """
    from scipy.fftpack import fftshift, ifft, fft 

    if isinstance(sigma, float):
        sigma = sigma*np.ones(len(bins), dtype=np.int)
    marginals_smooth = {}
    index = copy.deepcopy(marginals.keys())
    index.sort()
    for i in index:    
        nx = len(bins[i])-1
        dx = bins[i][1] - bins[i][0]
        augx = math.ceil(3*sigma[i]/dx)
        x_kernel = np.linspace(-nx*dx/2, nx*dx/2, nx)
        kernel = np.exp(-(x_kernel/sigma[i])**2)
        aug_kernel = np.zeros((nx+2*augx,))
        aug_marginals = np.zeros((nx+2*augx,))

        aug_kernel[augx:augx+nx] = kernel
        aug_marginals[augx:augx+nx] = marginals[i]

        aug_kernel = fftshift(aug_kernel)       

        aug_marginals_smooth = np.real(ifft(fft(aug_kernel)*fft(aug_marginals)))
        marginals_smooth[i] = aug_marginals_smooth[augx:augx+nx]
        marginals_smooth[i] = marginals_smooth[i]/np.sum(marginals_smooth[i])

    return marginals_smooth

def smooth_marginals_2D(marginals, bins, sigma=10.0):
    """
    This function smooths 2D marginal probabilities.

    :param marginals: 2D marginal probabilities
    :type marginals: dictionary with tuples of 2 integers as keys and
        :class:'~numpy.ndarray' of shape (nbins+1,) as values 
    :param bins: Endpoints of bins used in calculating marginals
    :type bins: :class:'~numpy.ndarray' of shape (nbins+1,)
    :param sigma: Smoothing parameter in each direction.
    :type sigma: :float or :class:'~numpy.ndarray' of shape (ndim,)
    :rtype: dict
    :returns: marginals_smooth
    """
    from scipy.fftpack import fftshift, ifft2, fft2 

    if isinstance(sigma, float):
        sigma = sigma*np.ones(len(bins), dtype=np.int)
    marginals_smooth = {}
    pairs = copy.deepcopy(marginals.keys())
    pairs.sort()
    for (i, j) in pairs:   
        nx = len(bins[i])-1
        ny = len(bins[j])-1
        dx = bins[i][1] - bins[i][0]
        dy = bins[j][1] - bins[j][0]

        augx = math.ceil(3*sigma[i]/dx)
        augy = math.ceil(3*sigma[j]/dy)

        x_kernel = np.linspace(-nx*dx/2, nx*dx/2, nx)
        y_kernel = np.linspace(-ny*dy/2, ny*dy/2, ny)
        X, Y = np.meshgrid(x_kernel, y_kernel, indexing='ij')

        kernel = np.exp(-(X/sigma[i])**2-(Y/sigma[j])**2)
        aug_kernel = np.zeros((nx+2*augx, ny+2*augy))
        aug_marginals = np.zeros((nx+2*augx, ny+2*augy))

        aug_kernel[augx:augx+nx, augy:augy+ny] = kernel
        aug_marginals[augx:augx+nx, augy:augy+ny] = marginals[(i, j)]

        aug_kernel = fftshift(aug_kernel, 0) 
        aug_kernel = fftshift(aug_kernel, 1)

        aug_marginals_smooth = ifft2(fft2(aug_kernel)*fft2(aug_marginals))
        aug_marginals_smooth = np.real(aug_marginals_smooth)
        marginals_smooth[(i, j)] = aug_marginals_smooth[augx:augx+nx, 
                augy:augy+ny]
        marginals_smooth[(i, j)] = marginals_smooth[(i, 
            j)]/np.sum(marginals_smooth[(i, j)])

    return marginals_smooth
