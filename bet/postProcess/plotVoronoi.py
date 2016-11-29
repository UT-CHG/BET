# Copyright (C) 2014-2016 The BET Development Team

"""
This module provides methods for Voronoi plots. 
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

def plot_1D_voronoi(sample_set, density=True, filename="file", 
                    lam_ref=None, interactive=False,
                    lambda_label=None, file_extension=".png"):
    """
    This makes a 1d Voronoi plot of the input probability measure for a 
    1D Voronoi sample set. If the sample_set object is a discretization 
    object, we assume that the probabilities to be plotted are from 
    the input space.

    .. note::

        Do not specify the file extension in the file name.

    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base` 
        or :class:`~bet.sample.discretization`
    :param density: Plot prob. density instead of prob. measure.
    :type density: bool
    :param filename: Prefix for output files.
    :type filename: str
    :param lam_ref: Reference parameters.
    :type lam_ref: :class:`~numpy.ndarray` of shape (ndim,) or None
    :param interactive: Whether or not to display interactive plots.
    :type interactive: bool
    :param lambda_label: Label for each parameter for plots.
    :type lambda_label: list of length nbins of strings or None
    :param string file_extension: file extenstion

    """

    # Check inputs
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

    # Check for global probabilities
    if sample_obj._probabilities is None:
        if sample_obj._probabilities_local is None:
            raise missing_attribute("Missing probabilities")
        else:
            sample_obj.local_to_global()

    if lam_ref is None:
        lam_ref = sample_obj._reference_value

    # Form 1D Voronoi
    ind_sort = np.argsort(sample_obj._values, axis=0)
    ends = 0.5 * (sample_obj._values[ind_sort][1::] + sample_obj._values[ind_sort][0:-1])
    ends = ends[:, 0, 0]
    mins = np.array([sample_obj._domain[0][0]] + list(ends))
    maxes = np.array(list(ends) + [sample_obj._domain[0][1]])

    # Make plot
    if comm.rank == 0:
        fig = plt.figure(0)
        if density:
            plt.hlines(sample_obj._probabilities[ind_sort]/(maxes-mins), mins, maxes)
            plt.ylabel(r'$\rho_{\lambda}$', fontsize=20)
        else:
            plt.hlines(sample_obj._probabilities[ind_sort], mins, maxes)
            plt.ylabel(r'$P_{\Lambda}(\mathcal{V}_i)$', fontsize=20)

        if lam_ref is not None:
            plt.plot(lam_ref[0], 0.0, 'ko', markersize=10)

        if lambda_label is None:
            label1 = r'$\lambda$'
        else:
            label1 = lambda_label[0]
        plt.xlabel(label1, fontsize=20)

        if interactive:
            plt.show()

        fig.savefig(filename + file_extension)

def plot_2D_voronoi(sample_set, density=True, colormap_type='BuGn', 
                    filename="file", 
                    lam_ref=None, interactive=False,
                    lambda_label=None, file_extension=".png"):

    """
    This makes a 2D Voronoi plot of the input probability measure for a 
    2D Voronoi sample set. If the sample_set object is a discretization 
    object, we assume that the probabilities to be plotted are from 
    the input space.

    .. note::

        Do not specify the file extension in the file name.

    :param sample_set: Object containing samples and probabilities
    :type sample_set: :class:`~bet.sample.sample_set_base` 
        or :class:`~bet.sample.discretization`
    :param density: Plot prob. density instead of prob. measure.
    :type density: bool
    :param colormap_type: type of color map to use
    :type colormap_type: str
    :param filename: Prefix for output files.
    :type filename: str
    :param lam_ref: Reference parameters.
    :type lam_ref: :class:`~numpy.ndarray` of shape (ndim,) or None
    :param interactive: Whether or not to display interactive plots.
    :type interactive: bool
    :param lambda_label: Label for each parameter for plots.
    :type lambda_label: list of length nbins of strings or None
    :param string file_extension: file extenstion

    """

    from scipy.spatial import Voronoi, voronoi_plot_2d

    # Check inputs
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

    # Check for global probabilities
    if sample_obj._probabilities is None:
        if sample_obj._probabilities_local is None:
            raise missing_attribute("Missing probabilities")
        else:
            sample_obj.local_to_global()
    if sample_obj._values is None:
        sample_obj.local_to_global()

    if lam_ref is None:
        lam_ref = sample_obj._reference_value

    # Form Voronoi 
    if comm.rank == 0:
        vor = Voronoi(sample_obj._values)
        regions, vertices =  voronoi_finite_polygons_2d(vor)
        points = sample_obj._values

        # Make plot
        fig = plt.figure(0)
        cmap = matplotlib.cm.get_cmap(colormap_type)
        if density:
            P = sample_obj._probabilities/sample_obj._volumes
        else:
            P = sample_obj._probabilities
        P_max = np.max(P)

        # plot each cell
        for i,region in enumerate(regions):
            polygon = vertices[region]
            plt.fill(*zip(*polygon),color=cmap(P[i]/P_max), edgecolor = 'k', linewidth = 0.005)

        plt.axis([sample_obj._domain[0][0], sample_obj._domain[0][1], sample_obj._domain[1][0], sample_obj._domain[1][1]])
        if lam_ref is not None:
            plt.plot(lam_ref[0], lam_ref[1], 'ro', markersize=10)

        if lambda_label is None:
            label1 = r'$\lambda_1$'
            label2 = r'$\lambda_2$'
        else:
            label1 = lambda_label[0]
            label1 = lambda_label[1]
        plt.xlabel(label1, fontsize=20)
        plt.ylabel(label2, fontsize=20)

        # plot colorbar
        ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=0.9)
        if density:
            cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                                    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=P_max), label=r'$\rho_{\Lambda}$')
        else:
            cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                                    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=P_max), label=r'$P_{\Lambda}(\mathcal{V}_i)$')
        text = cbar.ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=20)
        text.set_font_properties(font)
        if interactive:
            plt.show()

        fig.savefig(filename + file_extension)
    
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    :param vor: Voronoi input diagram
    :type vor: :class:`scipy.spatial.Voronoi`
    :param radius: Distance to points at infinity. Optional.
    :type radius: float
    :param regions: Indices of vertices in each revised Voronoi regions.
    :type regions: list of tuples
    :param vertices: Coordinates for revised Voronoi vertices. 
    Same as coordinates
    of input vertices, with 'points at infinity' appended to the end.
    :type vertices: list of tuples
    :rtype: tuple
    :returns (regions, vertices) 

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
