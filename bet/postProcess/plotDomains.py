# Copyright (C) 2014-2016 The BET Development Team

"""
This module provides methods used to plot two-dimensional domains and/or
two-dimensional slices/projections of domains.

"""

import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
from matplotlib.lines import Line2D
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import bet.util as util
import os
import bet.sample as sample

markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')


class dim_not_matching(Exception):
    """
    Exception for when the dimension is inconsistent.
    """


class bad_object(Exception):
    """
    Exception for when the wrong type of object is used.
    """


def scatter_2D(sample_obj, sample_nos=None, color=None, ref_sample=None, save=True,
               interactive=False, xlabel='x', ylabel='y',
               filename='scatter2d', file_extension="png"):
    r"""
    Creates a two-dimensional scatter plot of the samples within the sample
    object colored by ``color`` (usually an array of pointwise probability
    density values).  A reference sample (``ref_sample``) can be chosen by the user.
    This reference sample will be plotted as a mauve circle twice the size of
    the other markers.

    :param sample_obj: contains samples to create scatter plot
    :type sample_obj: :class:`~bet.sample.sample_set_base`
    :param list sample_nos: indicies of the samples to plot
    :param color: values to color the samples by
    :type color: :class:`numpy.ndarray`
    :param ref_sample: reference parameter value
    :type ref_sample: :class:`numpy.ndarray` of shape (ndim,)
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param string filename: filename to save the figure as
    :param string file_extension: file extension

    """
    if not isinstance(sample_obj, sample.sample_set_base):
        raise bad_object("Improper sample object")
    # check dimension of data to plot
    if sample_obj.get_dim() != 2:
        raise dim_not_matching("Cannot create 2D plot of non-2D sample "
                               "object")

    # plot all of the samples by default
    if sample_nos is None:
        sample_nos = np.arange(sample_obj.get_values().shape[0])
    # color all of the samples uniformly by default and set the default
    # to the default colormap of matplotlib
    if color is None:
        color = np.ones((sample_obj.get_values().shape[0],))
        cmap = None
    else:
        cmap = plt.cm.PuBu
    markersize = 75
    color = color[sample_nos]
    # create the scatter plot for the samples specified by sample_nos
    plt.scatter(sample_obj.get_values()[sample_nos, 0],
                sample_obj.get_values()[sample_nos, 1],
                c=color, s=markersize, alpha=.75, linewidth=.1, cmap=cmap)
    # add a colorbar and label for the colorbar usually we just assume the
    # samples are colored by the pointwise probability density on the data
    # space
    cbar = plt.colorbar()
    cbar.set_label(r'$\rho_\mathcal{D}(q)$')
    # if there is a reference value plot it with a notiable marker
    if ref_sample is not None:
        plt.scatter(ref_sample[0], ref_sample[1], c='m', s=2 * markersize)
    if save:
        plt.autoscale(tight=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename, bbox_inches='tight', transparent=True,
                    pad_inches=0, format=file_extension)
    if interactive:
        plt.show()
    else:
        plt.close()


def scatter_3D(sample_obj, sample_nos=None, color=None, ref_sample=None, save=True,
               interactive=False, xlabel='x', ylabel='y', zlabel='z',
               filename="scatter3d", file_extension="png"):
    r"""
    Creates a three-dimensional scatter plot of samples within the sample
    object colored by ``color`` (usually an array of pointwise probability
    density values). A reference sample (``ref_sample``) can be chosen by the user.
    This reference sample will be plotted as a mauve circle twice the size of
    the other markers.

    :param sample_obj: Object containing the samples to plot
    :type sample_obj: :class:`~bet.sample.sample_set_base`
    :param list sample_nos: indicies of the samples to plot
    :param color: values to color the samples by
    :type color: :class:`numpy.ndarray`
    :param ref_sample: reference parameter value
    :type ref_sample: :class:`numpy.ndarray` of shape (ndim,)
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param string zlabel: z-axis label
    :param string filename: filename to save the figure as
    :param string file_extension: file extension

    """
    if not isinstance(sample_obj, sample.sample_set_base):
        raise bad_object("Improper sample object")
    # check dimension of data to plot
    if sample_obj.get_dim() != 3:
        raise dim_not_matching("Cannot create 3D plot of non-3D sample "
                               "object")

    # plot all of the samples by default
    if sample_nos is None:
        sample_nos = np.arange(sample_obj.get_values().shape[0])
    # color all of the samples uniformly by default and set the default
    # to the default colormap of matplotlib
    if color is None:
        color = np.ones((sample_obj.get_values().shape[0],))
        cmap = None
    else:
        cmap = plt.cm.PuBu
    markersize = 75
    color = color[sample_nos]
    # create the scatter plot for the samples specified by sample_nos
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(sample_obj.get_values()[sample_nos, 0],
                   sample_obj.get_values()[sample_nos, 1],
                   sample_obj.get_values()[sample_nos, 2],
                   alpha=.75, linewidth=.1, c=color, s=markersize, cmap=cmap)
    # add a colorbar and label for the colorbar usually we just assume the
    # samples are colored by the pointwise probability density on the data
    # space
    cbar = fig.colorbar(p)
    cbar.set_label(r'$\rho_\mathcal{D}(q)$')
    # if there is a reference value plot it with a notiable marker
    if ref_sample is not None:
        ax.scatter(ref_sample[0], ref_sample[1], ref_sample[2], c='m', s=2 * markersize)
    ax.autoscale(tight=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if save:
        plt.savefig(filename, bbox_inches='tight', transparent=True,
                    pad_inches=0, format=file_extension)
    if interactive:
        plt.show()
    else:
        plt.close()

def scatter_rhoD(sample_obj, ref_sample=None, sample_nos=None, io_flag='input',
        rho_D=None, dim_nums=None, label_char=None, showdim=None, save=True,
        interactive=False, file_extension="png"):
    r"""
    Create scatter plots of samples within the sample object colored by
    ``color`` (usually an array of pointwise probability density values).  A
    reference sample (``ref_sample``) can be chosen by the user.  This reference
    sample will be plotted as a mauve circle twice the size of the other
    markers.

    :param sample_obj: Object containing the samples to plot
    :type sample_obj: :class:`~bet.sample.discretization` 
        or :class:`~bet.sample.sample_set_base`
    :param ref_sample: reference parameter value
    :type ref_sample: :class:`numpy.ndarray` of shape (ndim,)
    :param list sample_nos: sample numbers to plot
    :param string io_flag: Either `input` or `output`. If ``sample_obj`` is a
        :class:`~bet.sample.discretization` object flag whether or not put plot
        input or output.
    :param rho_D: probability density function on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`numpy.ndarray`
    :param list dim_nums: integers representing domain coordinate
        numbers to plot (e.g. i, where :math:`\x_i` is a coordinate in the
        input/output space).
    :param string label_char: character to use to label coordinate axes
    :param int showdim: 2 or 3, flag to determine whether or not to show
        pairwise or tripletwise parameter sample scatter plots in 2 or 3
        dimensions
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param string file_extension: file extension

    """
    # If there is density function given determine the pointwise probability
    # values of each sample based on the value in the data space. Otherwise,
    # color the samples in numerical order.
    rD = None
    if isinstance(sample_obj, sample.discretization):
        if rho_D is not None:
            rD = rho_D(sample_obj._output_sample_set.get_values())
        if io_flag == 'input':
            sample_obj = sample_obj._input_sample_set
        else:
            sample_obj = sample_obj._output_sample_set
    elif isinstance(sample_obj, sample.sample_set_base):
        if io_flag == 'output':
            rD = rho_D(sample_obj.get_values())
    else:
        raise bad_object("Improper sample object")
    
    if rD is None:
        rD = np.ones(sample_obj.get_values().shape[0])

    if label_char is not None:
        if io_flag == 'input':
            label_char = r'$\lambda_'
            prefix = 'input_'
        elif io_flag == 'output':
            label_char = r'$q_'
            prefix = 'output_'
    else:
        label_char = r'$x_'
        prefix = 'rhoD_'

    # If no specific coordinate numbers are given for the parameter coordinates
    # (e.g. i, where \lambda_i is a coordinate in the parameter space), then
    # set them to be the the counting numbers.
    if dim_nums is None:
        dim_nums = 1 + np.array(range(sample_obj.get_values().shape[1]))
    # Create the labels based on the user selected parameter coordinates
    print label_char, dim_nums
    xlabel = label_char+r'{' + str(dim_nums[0]) + '}$'
    ylabel = label_char+r'{' + str(dim_nums[1]) + '}$'
    savename = prefix+'samples_cs.eps'
    # Plot 2 or 3 dimensional scatter plots of the samples colored by rD.
    if sample_obj.get_dim() == 2:
        scatter_2D(sample_obj, sample_nos, rD, ref_sample, save,
                   interactive, xlabel, ylabel, savename)
    elif sample_obj.get_dim() == 3:
        zlabel = label_char+r'{' + str(dim_nums[2]) + '}$'
        scatter_3D(sample_obj, sample_nos, rD, ref_sample, save,
                   interactive, xlabel, ylabel, zlabel, savename)
    elif sample_obj.get_dim() > 2 and showdim == 2:
        temp_obj = sample.sample_set(2)
        for x, y in combinations(dim_nums, 2):
            xlabel = label_char+r'{' + str(x) + '}$'
            ylabel = label_char+r'{' + str(y) + '}$'
            savename = prefix+'samples_x' + str(x) + 'x' + str(y) + '_cs.eps'
            temp_obj.set_values(sample_obj.get_values()[:, [x - 1, y - 1]])
            scatter_2D(temp_obj, sample_nos, rD, ref_sample, save,
                       interactive, xlabel, ylabel, savename)
    elif sample_obj.get_dim() > 3 and showdim == 3:
        temp_obj = sample.sample_set(3)
        for x, y, z in combinations(dim_nums, 3):
            xlabel = label_char+r'{' + str(x) + '}$'
            ylabel = label_char+r'{' + str(y) + '}$'
            zlabel = label_char+r'{' + str(z) + '}$'
            savename = prefix+'samples_x' + str(x) + 'x' + str(y) + 'x' + str(z) + \
                       '_cs.eps'
            temp_obj.set_values(sample_obj.get_values()[:, [x - 1, y - 1, z - 1]])
            scatter_3D(temp_obj, sample_nos, rD, ref_sample, save,
                       interactive, xlabel, ylabel, zlabel, savename)

def show_data_domain_multi(sample_disc, Q_ref=None, Q_nums=None,
                           img_folder='figs/', ref_markers=None,
                           ref_colors=None, showdim=None, file_extension="png"):
    r"""
    Plots 2-D projections of the data domain D using a triangulation based on
    the first two coordinates (parameters) of the generating samples where
    :math:`Q={q_1, q_i}` for ``i=Q_nums``, with a marker for various
    :math:`Q_{ref}`.

    :param sample_disc: Object containing the samples to plot
    :type sample_disc: :class:`~bet.sample.discretization` 
    :param Q_ref: reference data value
    :type Q_ref: :class:`numpy.ndarray` of shape (M, mdim)
    :param list Q_nums: dimensions of the QoI to plot
    :param string img_folder: folder to save the plots to
    :param list ref_markers: list of marker types for :math:`Q_{ref}`
    :param list ref_colors: list of colors for :math:`Q_{ref}`
    :param showdim: default 1. If int then flag to show all combinations with a
        given dimension (:math:`q_i`) or if ``all`` show all combinations.
    :type showdim: int or string
    :param string file_extension: file extension

    """
    if not isinstance(sample_disc, sample.discretization):
        raise bad_object("Improper sample object")

    # Set the default marker and colors
    if ref_markers == None:
        ref_markers = markers
    if ref_colors == None:
        ref_colors = colors

    data_obj = sample_disc._output_sample_set
    sample_obj = sample_disc._input_sample_set
    # If no specific coordinate numbers are given for the data coordinates
    # (e.g. i, where \q_i is a coordinate in the data space), then
    # set them to be the the counting numbers.
    if Q_nums is None:
        Q_nums = range(data_obj.get_dim())
    # If no specific coordinate number of choice is given set to be the first
    # coordinate direction.
    if showdim == None:
        showdim = 0

    # Create a folder for these figures if it doesn't already exist
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    # Make sure the shape of Q_ref is correct
    if Q_ref is not None:
        Q_ref = util.fix_dimensions_data(Q_ref, data_obj.get_dim())

    # Create the triangulization to use to define the topology of the samples
    # in the data space from the first two parameters in the parameter space
    triangulation = tri.Triangulation(sample_obj.get_values()[:, 0],
                                      sample_obj.get_values()[:, 1])
    triangles = triangulation.triangles

    # Create plots of the showdim^th QoI (q_{showdim}) with all other QoI (q_i)
    if isinstance(showdim, int):
        for i in Q_nums:
            xlabel = r'$q_{' + str(showdim + 1) + r'}$'
            ylabel = r'$q_{' + str(i + 1) + r'}$'

            filenames = [img_folder + 'domain_q' + str(showdim + 1) + '_q' + \
                         str(i + 1) + '.eps', img_folder + 'q' + str(showdim + 1) + \
                         '_q' + str(i + 1) + '_domain_Q_cs.eps']

            data_obj_temp = sample.sample_set(2)
            data_obj_temp.set_values(data_obj.get_values()[:, [showdim, i]])
            sample_disc_temp = sample.discretization(sample_obj, data_obj_temp)

            if Q_ref is not None:
                show_data_domain_2D(sample_disc_temp, Q_ref[:, [showdim, i]],
                                    ref_markers, ref_colors, xlabel=xlabel,
                                    ylabel=ylabel, triangles=triangles, save=True,
                                    interactive=False, filenames=filenames)

            else:
                show_data_domain_2D(sample_disc_temp, None,
                                    ref_markers, ref_colors, xlabel=xlabel, ylabel=ylabel,
                                    triangles=triangles, save=True, interactive=False,
                                    filenames=filenames)
    # Create plots of all combinations of QoI in 2D
    elif showdim == 'all' or showdim == 'ALL':
        for x, y in combinations(Q_nums, 2):
            xlabel = r'$q_{' + str(x + 1) + r'}$'
            ylabel = r'$q_{' + str(y + 1) + r'}$'

            filenames = [img_folder + 'domain_q' + str(x + 1) + '_q' + str(y + 1) + '.eps',
                         img_folder + 'q' + str(x + 1) + '_q' + str(y + 1) + '_domain_Q_cs.eps']

            data_obj_temp = sample.sample_set(2)
            data_obj_temp.set_values(data_obj.get_values()[:, [x, y]])
            sample_disc_temp = sample.discretization(sample_obj, data_obj_temp)

            if Q_ref is not None:
                show_data_domain_2D(sample_disc_temp, Q_ref[:, [x, y]],
                                    ref_markers, ref_colors, xlabel=xlabel, ylabel=ylabel,
                                    triangles=triangles, save=True, interactive=False,
                                    filenames=filenames)
            else:
                show_data_domain_2D(sample_disc_temp, None,
                                    ref_markers, ref_colors, xlabel=xlabel, ylabel=ylabel,
                                    triangles=triangles, save=True, interactive=False,
                                    filenames=filenames)


def show_data_domain_2D(sample_disc, Q_ref=None, ref_markers=None,
                        ref_colors=None, xlabel=r'$q_1$', ylabel=r'$q_2$',
                        triangles=None, save=True, interactive=False,
                        filenames=None, file_extension="png"):
    r"""
    Plots 2-D a single data domain D using a triangulation based on the first
    two coordinates (parameters) of the generating samples where :math:`Q={q_1,
    q_i}` for ``i=Q_nums``, with a marker for various :math:`Q_{ref}`. Assumes
    that the first dimension of data is :math:`q_1`.


    :param sample_disc: Object containing the samples to plot
    :type sample_disc: :class:`~bet.sample.discretization` 
        or :class:`~bet.sample.sample_set_base`
    :param Q_ref: reference data value
    :type Q_ref: :class:`numpy.ndarray` of shape (M, 2)
    :param list ref_markers: list of marker types for :math:`Q_{ref}`
    :param list ref_colors: list of colors for :math:`Q_{ref}`
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param triangles: triangulation defined by ``samples``
    :type triangles: :class:`tri.Triuangulation.triangles`
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param list filenames: file names for the unmarked and marked domain plots
    :param string file_extension: file extension

    """
    if not isinstance(sample_disc, sample.discretization):
        raise bad_object("Improper sample object")

    data_obj = sample_disc._output_sample_set
    sample_obj = sample_disc._input_sample_set

    # Set the default marker and colors
    if ref_markers == None:
        ref_markers = markers
    if ref_colors == None:
        ref_colors = colors
    # If no specific coordinate numbers are given for the data coordinates
    # (e.g. i, where \q_i is a coordinate in the data space), then
    # set them to be the the counting numbers.
    if triangles is None:
        triangulation = tri.Triangulation(sample_obj.get_values()[:, 0],
                                          sample_obj.get_values()[:, 1])
        triangles = triangulation.triangles
    # Set default file names
    if filenames == None:
        filenames = ['domain_q1_q2_cs.eps', 'q1_q2_domain_Q_cs.eps']

    # Make sure the shape of Q_ref is correct
    if Q_ref is not None:
        Q_ref = util.fix_dimensions_data(Q_ref, 2)

    # Create figure
    plt.tricontourf(data_obj.get_values()[:, 0], data_obj.get_values()[:, 1],
                    np.zeros((data_obj.get_values().shape[0],)),
                    triangles=triangles, colors='grey')
    plt.autoscale(tight=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filenames[0], bbox_inches='tight', transparent=True,
                pad_inches=.2, format=file_extension)
    # Add truth markers
    if Q_ref is not None:
        for i in xrange(Q_ref.shape[0]):
            plt.scatter(Q_ref[i, 0], Q_ref[i, 1], s=60, c=ref_colors[i],
                        marker=ref_markers[i])
    if save:
        plt.savefig(filenames[1], bbox_inches='tight', transparent=True,
                    pad_inches=.2, format=file_extension)
    if interactive:
        plt.show()
    else:
        plt.close()

def scatter_2D_multi(sample_obj, color=None, ref_sample=None, img_folder='figs/',
                    filename="scatter2Dm", label_char=r'$\lambda',
                    showdim=None, file_extension="png"):
    r"""
    Creates two-dimensional projections of scatter plots of samples colored
    by ``color`` (usually an array of pointwise probability density values). A
    reference sample (``ref_sample``) can be chosen by the user. This reference
    sample will be plotted as a mauve circle twice the size of the other
    markers.

    :param sample_obj: Object containing the samples to plot
    :type sample_obj: :class:`~bet.sample.sample_set_base`
    :param color: values to color the ``samples`` by
    :type color: :class:`numpy.ndarray`
    :param string filename: filename to save the figure as
    :param string label_char: character to use to label coordinate axes
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param string img_folder: folder to save the plots to
    :param showdim: default 1. If int then flag to show all combinations with a
        given dimension (:math:`\lambda_i`) or if ``all`` show all combinations.
    :type showdim: int or string
    :param string filename: filename to save the figure as

    """
    if not isinstance(sample_obj, sample.sample_set_base):
        raise bad_object("Improper sample object")
    # If no specific coordinate number of choice is given set to be the first
    # coordinate direction.
    if showdim == None:
        showdim = 0
    # Create a folder for these figures if it doesn't already exist
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)
    # Create list of all the parameter coordinates
    p_nums = range(sample_obj.get_dim())

    # Create plots of the showdim^th parameter (\lambda_{showdim}) with all the
    # other parameters
    if isinstance(showdim, int):
        for i in p_nums:
            xlabel = label_char + r'_{' + str(showdim + 1) + r'}$'
            ylabel = label_char + r'_{' + str(i + 1) + r'}$'

            postfix = '_d' + str(showdim + 1) + '_d' + str(i + 1) + '.eps'
            myfilename = os.path.join(img_folder, filename + postfix)

            sample_obj_temp = sample.sample_set(2)
            sample_obj_temp.set_values(sample_obj.get_values()[:, [showdim, i]])

            if ref_sample is not None:
                scatter_2D(sample_obj_temp, sample_nos=None,
                           color=color, ref_sample=ref_sample[[showdim, i]], save=True,
                           interactive=False, xlabel=xlabel, ylabel=ylabel,
                           filename=myfilename)
            else:
                scatter_2D(sample_obj_temp, sample_nos=None,
                           color=color, ref_sample=None, save=True,
                           interactive=False, xlabel=xlabel, ylabel=ylabel,
                           filename=myfilename)

    # Create plots of all of the possible pairwise combinations of parameters
    elif showdim == 'all' or showdim == 'ALL':
        for x, y in combinations(p_nums, 2):
            xlabel = label_char + r'_{' + str(x + 1) + r'}$'
            ylabel = label_char + r'_{' + str(y + 1) + r'}$'

            postfix = '_d' + str(x + 1) + '_d' + str(y + 1) + '.eps'
            myfilename = os.path.join(img_folder, filename + postfix)

            sample_obj_temp = sample.sample_set(2)
            sample_obj_temp.set_values(sample_obj.get_values()[:, [x, y]])

            if ref_sample is not None:
                scatter_2D(sample_obj_temp, sample_nos=None, color=color,
                           ref_sample=ref_sample[[x, y]], save=True, interactive=False,
                           xlabel=xlabel, ylabel=ylabel, filename=myfilename)
            else:
                scatter_2D(sample_obj_temp, sample_nos=None, color=color,
                           ref_sample=None, save=True, interactive=False,
                           xlabel=xlabel, ylabel=ylabel, filename=myfilename)
