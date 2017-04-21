# Copyright (C) 2014-2016 The BET Development Team

"""
This module provides methods for plotting probabilities. 
"""

import copy, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
from bet.Comm import comm, MPI 
import bet.sample as sample


class dim_not_matching(Exception):
    """
    Exception for when the dimension is inconsistent.
    """

class bad_object(Exception):
    """
    Exception for when the wrong type of object is used.
    """

class missing_attribute(Exception):
    """
    Exception for missing attribute.
    """

def calculate_1D_marginal_probs(sample_set, nbins=20):
        
    r"""
    This calculates every single marginal of the probability measure
    described by the probabilities within the sample_set object.
    If the sample_set object is a discretization object, we assume
    that the probabilities to be plotted are from the input space on the
    emulated samples
    (``discretization._emulated_input_sample_set._probabilties_local``).

    This assumes that the user has already run
    :meth:`~bet.calculateP.calculateP.prob_emulated`.

    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base` or 
        :class:`~bet.sample.discretization`
    :param nbins: Number of bins in each direction.
    :type nbins: :int or :class:`~numpy.ndarray` of shape (ndim,)
    :rtype: tuple
    :returns: (bins, marginals)

    """
    if isinstance(sample_set, sample.discretization):
        sample_obj = sample_set._emulated_input_sample_set
        if sample_obj is None:
            raise missing_attribute("Missing emulated_input_sample_set")
    elif isinstance(sample_set, sample.sample_set_base):
        sample_obj = sample_set
    else:
        raise bad_object("Improper sample object")

    # Check for local probabilities
    if sample_obj._probabilities_local is None:
        if sample_obj._probabilities is None:
            raise missing_attribute("Missing probabilities")
        else:
            sample_obj.global_to_local()

    # Make list of bins if only an integer is given
    if isinstance(nbins, int):
        nbins = nbins*np.ones(sample_obj.get_dim(), dtype=np.int)
 
    # Create bins
    bins = []
    for i in range(sample_obj.get_dim()):
        bins.append(np.linspace(sample_obj.get_domain()[i][0],
                                sample_obj.get_domain()[i][1],
                                nbins[i]+1))
        
    # Calculate marginals
    marginals = {}
    for i in range(sample_obj.get_dim()):
        [marg, _] = np.histogram(sample_obj.get_values_local()[:, i],
                bins=bins[i], weights=sample_obj.get_probabilities_local())
        marg_temp = np.copy(marg)
        comm.Allreduce([marg, MPI.DOUBLE], [marg_temp, MPI.DOUBLE], op=MPI.SUM)
        marginals[i] = marg_temp

    return (bins, marginals)

def calculate_2D_marginal_probs(sample_set, nbins=20):
        
    """
    This calculates every pair of marginals (or joint in 2d case) of
    input probability measure defined on a rectangular grid.
    If the sample_set object is a discretization object, we assume
    that the probabilities to be plotted are from the input space on the
    emulated samples
    (``discretization._emulated_input_sample_set._probabilties_local``).

    This assumes that the user has already run
    :meth:`~bet.calculateP.calculateP.prob_emulated`.


    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base` 
        or :class:`~bet.sample.discretization`
    :param nbins: Number of bins in each direction.
    :type nbins: :int or :class:`~numpy.ndarray` of shape (ndim,)
    :rtype: tuple
    :returns: (bins, marginals)

    """
    if isinstance(sample_set, sample.discretization):
        sample_obj = sample_set._emulated_input_sample_set
        if sample_obj is None:
            raise missing_attribute("Missing emulated_input_sample_set")
    elif isinstance(sample_set, sample.sample_set_base):
        sample_obj = sample_set
    else:
        raise bad_object("Improper sample object")

    # Check for local probabilities
    if sample_obj._probabilities_local is None:
        if sample_obj._probabilities is None:
            raise missing_attribute("Missing probabilities")
        else:
            sample_obj.global_to_local()

    if sample_obj.get_dim() < 2:
        raise dim_not_matching("Incompatible dimensions of sample set"
                               " for plotting")

    # Make list of bins if only an integer is given
    if isinstance(nbins, int):
        nbins = nbins*np.ones(sample_obj.get_dim(), dtype=np.int)

    # Create bins
    bins = []
    for i in range(sample_obj.get_dim()):
        bins.append(np.linspace(sample_obj.get_domain()[i][0],
                                sample_obj.get_domain()[i][1],
                                nbins[i]+1))

    # Calculate marginals
    marginals = {}
    for i in range(sample_obj.get_dim()):
        for j in range(i+1, sample_obj.get_dim()):
            (marg, _) = np.histogramdd(sample_obj.get_values_local()[:, [i, j]],
                    bins=[bins[i], bins[j]],
                    weights=sample_obj.get_probabilities_local())
            marg = np.ascontiguousarray(marg)
            marg_temp = np.copy(marg)
            comm.Allreduce([marg, MPI.DOUBLE], [marg_temp, MPI.DOUBLE],
                    op=MPI.SUM) 
            marginals[(i, j)] = marg_temp

    return (bins, marginals)

def plot_1D_marginal_probs(marginals, bins, sample_set,
        filename="file", lam_ref=None, interactive=False,
        lambda_label=None, file_extension=".png"):
        
    """
    This makes plots of every single marginal probability of
    input probability measure on a 1D  grid.
    If the sample_set object is a discretization object, we assume
    that the probabilities to be plotted are from the input space.

    .. note::

        Do not specify the file extension in the file name.

    :param marginals: 1D marginal probabilities
    :type marginals: dictionary with int as keys and :class:`~numpy.ndarray` of
        shape (nbins+1,) as values :param bins: Endpoints of bins used in
        calculating marginals
    :type bins: :class:`~numpy.ndarray` of shape (nbins+1,)
    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base` 
        or :class:`~bet.sample.discretization`
    :param filename: Prefix for output files.
    :type filename: str
    :param lam_ref: True parameters.
    :type lam_ref: :class:`~numpy.ndarray` of shape (ndim,) or None
    :param interactive: Whether or not to display interactive plots.
    :type interactive: bool
    :param lambda_label: Label for each parameter for plots.
    :type lambda_label: list of length nbins of strings or None
    :param string file_extension: file extenstion

    """
    if isinstance(sample_set, sample.discretization):
        sample_obj = sample_set._input_sample_set
    elif isinstance(sample_set, sample.sample_set_base):
        sample_obj = sample_set
    else:
        raise bad_object("Improper sample object")

    if lam_ref is None:
        lam_ref = sample_obj._reference_value

    lam_domain = sample_obj.get_domain()

    if comm.rank == 0:
        index = copy.deepcopy(marginals.keys())
        index.sort()
        for i in index:
            x_range = np.linspace(lam_domain[i, 0], lam_domain[i, 1],
                    len(bins[i])-1) 
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            ax.plot(x_range, marginals[i]/(bins[i][1]-bins[i][0]))
            ax.set_ylim([0, 1.05*np.max(marginals[i]/(bins[i][1]-bins[i][0]))])
            if lam_ref is not None:
                ax.plot(lam_ref[i], 0.0, 'ko', markersize=10)
            if lambda_label is None:
                label1 = r'$\lambda_{' + str(i+1) + '}$'
            else:
                label1 = lambda_label[i]
            ax.set_xlabel(label1) 
            ax.set_ylabel(r'$\rho$')
            fig.savefig(filename + "_1D_" + str(i) + file_extension,
                    transparent=True) 
            if interactive:
                plt.show()
            else:
                plt.close()
            plt.clf()
    comm.barrier()

def plot_2D_marginal_probs(marginals, bins, sample_set,
        filename="file", lam_ref=None, plot_surface=False, interactive=False,
        lambda_label=None, file_extension=".png"):
        
    """
    This makes plots of every pair of marginals (or joint in 2d case) of
    input probability measure on a rectangular grid.
    If the sample_set object is a discretization object, we assume
    that the probabilities to be plotted are from the input space.

    .. note::

        Do not specify the file extension in the file name.

    :param marginals: 2D marginal probabilities
    :type marginals: dictionary with tuples of 2 integers as keys and
        :class:`~numpy.ndarray` of shape (nbins+1,) as values 
    :param bins: Endpoints of bins used in calculating marginals
    :type bins: :class:`~numpy.ndarray` of shape (nbins+1,2)
    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base` 
        or :class:`~bet.sample.discretization`
    :param filename: Prefix for output files.
    :type filename: str
    :param lam_ref: True parameters.
    :type lam_ref: :class:`~numpy.ndarray` of shape (ndim,) or None
    :param interactive: Whether or not to display interactive plots.
    :type interactive: bool
    :param lambda_label: Label for each parameter for plots.
    :type lambda_label: list of length nbins of strings or None
    :param string file_extension: file extenstion

    """
    if isinstance(sample_set, sample.discretization):
        sample_obj = sample_set._input_sample_set
    elif isinstance(sample_set, sample.sample_set_base):
        sample_obj = sample_set
    else:
        raise bad_object("Improper sample object")

    if lam_ref is None:
        lam_ref = sample_obj._reference_value

    lam_domain = sample_obj.get_domain()

    from matplotlib import cm
    if plot_surface:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
    if comm.rank == 0:
        pairs = copy.deepcopy(marginals.keys())
        pairs.sort()
        for k, (i, j) in enumerate(pairs):
            fig = plt.figure(k)
            ax = fig.add_subplot(111)
            boxSize = (bins[i][1]-bins[i][0])*(bins[j][1]-bins[j][0])
            quadmesh = ax.imshow(marginals[(i, j)].transpose()/boxSize,
                    interpolation='bicubic', cmap=cm.CMRmap_r, 
                    extent=[lam_domain[i][0], lam_domain[i][1],
                    lam_domain[j][0], lam_domain[j][1]], origin='lower',
                    vmax=marginals[(i, j)].max()/boxSize, vmin=0, aspect='auto')
            if lam_ref is not None:
                ax.plot(lam_ref[i], lam_ref[j], 'wo', markersize=10)
            if lambda_label is None:
                label1 = r'$\lambda_{' + str(i+1) + '}$'
                label2 = r'$\lambda_{' + str(j+1) + '}$'
            else:
                label1 = lambda_label[i]
                label2 = lambda_label[j]
            ax.set_xlabel(label1, fontsize=20) 
            ax.set_ylabel(label2, fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=14)
            label_cbar = r'$\rho_{\lambda_{' + str(i+1) + '}, ' 
            label_cbar += r'\lambda_{' + str(j+1) + '}' + '}$ (Lebesgue)'
            cb = fig.colorbar(quadmesh, ax=ax, label=label_cbar)
            cb.ax.tick_params(labelsize=14)
            cb.set_label(label_cbar, size=20)
            plt.axis([lam_domain[i][0], lam_domain[i][1], lam_domain[j][0],
                lam_domain[j][1]]) 
            fig.savefig(filename + "_2D_" + str(i) + "_" + str(j) +\
                    file_extension, transparent=True)
            if interactive:
                plt.show()
            else:
                plt.close()
 
        if plot_surface:
            for k, (i, j) in enumerate(pairs):
                fig = plt.figure(k)
                ax = fig.gca(projection='3d')
                X = bins[i][:-1] + np.diff(bins[i])/2 
                Y = bins[j][:-1] + np.diff(bins[j])/2
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
                fig.savefig(filename + "_surf_" + str(i) + "_" + str(j) + \
                        file_extension, transparent=True)

                if interactive:
                    plt.show()
                else:
                    plt.close()
                plt.clf()
    comm.barrier()
		
def smooth_marginals_1D(marginals, bins, sigma=10.0):
    """
    This function smooths 1D marginal probabilities.

    :param marginals: 1D marginal probabilities
    :type marginals: dictionary with int as keys and :class:`~numpy.ndarray` of
        shape (nbins+1,) as values 
    :param bins: Endpoints of bins used in calculating marginals
    :type bins: :class:`~numpy.ndarray` of shape (nbins+1,)
    :param sigma: Smoothing parameter in each direction.
    :type sigma: float or :class:`~numpy.ndarray` of shape (ndim,)
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
        augx = int(math.ceil(3*sigma[i]/dx))
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
        :class:`~numpy.ndarray` of shape (nbins+1,) as values 
    :param bins: Endpoints of bins used in calculating marginals
    :type bins: :class:`~numpy.ndarray` of shape (nbins+1,)
    :param sigma: Smoothing parameter in each direction.
    :type sigma: float or :class:`~numpy.ndarray` of shape (ndim,)
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

        augx = int(math.ceil(3*sigma[i]/dx))
        augy = int(math.ceil(3*sigma[j]/dy))

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


