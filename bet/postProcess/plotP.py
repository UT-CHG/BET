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

def plot_1D_voronoi(sample_set, density=False, filename="file", 
                    lam_ref=None, interactive=False,
                    lambda_label=None, file_extension=".png"):
    if isinstance(sample_set, sample.discretization):
        sample_obj = sample_set._input_sample_set
        if sample_obj is None:
            raise missing_attribute("Missing input_sample_set")
    elif isinstance(sample_set, sample.sample_set_base):
        sample_obj = sample_set
    else:
        raise bad_object("Improper sample object")

    if sample_obj._dim != 1:
            raise dim_not_matching("Only applicable for 1D domains.")

    if lam_ref is None:
        lam_ref = sample_obj._reference_value

    ind_sort = np.argsort(sample_obj._values, axis=0)
    ends = 0.5 * (sample_obj._values[ind_sort][1::] + sample_obj._values[ind_sort][0:-1])
    ends = ends[:, 0, 0]
    mins = np.array([sample_obj._domain[0][0]] + list(ends))
    maxes = np.array(list(ends) + [sample_obj._domain[0][1]])
    #import pdb
    #pdb.set_trace()
    fig = plt.figure(0)
    if density:
        plt.hlines(sample_obj._probabilities[ind_sort]/(maxes-mins), mins, maxes)
        plt.ylabel(r'$\rho_{\lambda}$')
        #plt.xlabel(r'$\lambda$')
    else:
        plt.hlines(sample_obj._probabilities[ind_sort], mins, maxes)
        plt.ylabel(r'$P_{\Lambda}(\mathcal{V}_i)$')
        #plt.xlabel(r'$\lambda$')
    if lam_ref is not None:
        plt.plot(lam_ref[0], 0.0, 'ko', markersize=10)
    if lambda_label is None:
        label1 = r'$\lambda$'
    else:
        label1 = lambda_label[0]
    plt.xlabel(label1)
    if interactive:
        plt.show()
    fig.savefig(filename + file_extension)

def plot_2D_voronoi(sample_set, density=False, colormap_type='BuGn'):
    from scipy.spatial import Voronoi, voronoi_plot_2d
    if isinstance(sample_set, sample.discretization):
        sample_obj = sample_set._input_sample_set
        if sample_obj is None:
            raise missing_attribute("Missing input_sample_set")
    elif isinstance(sample_set, sample.sample_set_base):
        sample_obj = sample_set
    else:
        raise bad_object("Improper sample object")

    if sample_obj._dim != 2:
            raise dim_not_matching("Only applicable for 2D domains.")

    vor = Voronoi(sample_obj._values)
    regions, vertices =  voronoi_finite_polygons_2d(vor)
    points = sample_obj._values
    cmap = matplotlib.cm.get_cmap(colormap_type)
    if density:
        P = sample_obj._probabilities/sample_obj._volumes
    else:
        P = sample_obj._probabilities
    P_max = np.max(P)

    #fig = plt.figure(0)
    #ax = fig.add_subplot(111)
    for i,region in enumerate(regions):
        polygon = vertices[region]
        plt.fill(*zip(*polygon),color=cmap(P[i]/P_max), edgecolor = 'k', linewidth = 0.005)

    #plt.plot(points[:,0], points[:,1], 'ko')
    #plt.xlim(vor.min_bound[0], vor.max_bound[0])
    #plt.ylim(vor.min_bound[1] , vor.max_bound[1])
    plt.axis([sample_obj._domain[0][0], sample_obj._domain[0][1], sample_obj._domain[1][0], sample_obj._domain[1][1]])
    #fig.colorbar(ax=ax, mappable=sample_obj._probabilities[i])
    #plt.colorbar()
    plt.xlabel(r'$\lambda_1$')
    plt.ylabel(r'$\lambda_2$')
    ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=0.9)
    if density:
        cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                            norm=matplotlib.colors.Normalize(vmin=0.0, vmax=P_max), label=r'$\rho_{\Lambda}$')
    else:
        cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                            norm=matplotlib.colors.Normalize(vmin=0.0, vmax=P_max), label=r'$P_{\Lambda}(\mathcal{V}_i)$')
    #cbar.set_clim(-2, 2)

    #plt.tight_layout()
    plt.show()
    #voronoi_plot_2d(vor, show_points=False, show_vertices=False)
    # fig = plt.figure()
    # plt.axis([sample_obj._domain[0][0], sample_obj._domain[0][1], sample_obj._domain[1][0], sample_obj._domain[1][1]])
    # plt.hold(True)
    # cmap = matplotlib.cm.get_cmap('BuGn')
    # P_max = np.max(sample_obj._probabilities)
    # for i,val in enumerate(vor.point_region):
    #     region = vor.regions[val]
    #     if not -1 in region:
    #         polygon = [vor.vertices[i] for i in region]
    #         z = zip(*polygon)
    #         #plt.fill(z[0], z[1],color=cmap(sample_obj._probabilities[i]/P_max) , edgecolor = 'k', linewidth = 0.005)
    #         plt.fill(z[0], z[1], color='r' , edgecolor = 'k', linewidth = 0.005)

    #plt.show()

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

