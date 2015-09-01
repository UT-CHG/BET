# Copyright (C) 2014-2015 The BET Development Team

"""
This module provides methods used to plot two-dimensional domains and/or
two-dimensional slices/projections of domains.

"""

import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from matplotlib.lines import Line2D
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import bet.util as util
import os

markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

def scatter_2D(samples, sample_nos=None, color=None, p_ref=None, save=True,
               interactive=False, xlabel='x', ylabel='y',
               filename='scatter2d'): 
    """
    Creates a two-dimensional scatter plot of ``samples`` colored by ``color``
    (usually an array of pointwise probability density values). A reference
    ``sample`` (``p_ref``) can be chosen by the user. This reference ``sample``
    will be plotted as a mauve circle twice the size of the other markers.
    
    :param samples: Samples to plot. These are the locations in the x-axis and
        y-axis.
    :type samples: :class:`numpy.ndarray`
    :param list sample_nos: indicies of the ``samples`` to plot
    :param color: values to color the ``samples`` by
    :type color: :class:`numpy.ndarray`
    :param p_ref: reference parameter(``sample``) value
    :type p_ref: :class:`numpy.ndarray` of shape (ndim,)
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param string filename: filename to save the figure as

    """
    # plot all of the samples by default
    if type(sample_nos) == type(None):
        sample_nos = np.arange(samples.shape[0])
    # color all of the samples uniformly by default and set the default
    # to the default colormap of matplotlib
    if color is None:
        color = np.ones((samples.shape[0],))
        cmap = None
    else:
        cmap = plt.cm.PuBu
    markersize = 75
    color = color[sample_nos]
    # create the scatter plot for the samples specified by sample_nos
    plt.scatter(samples[sample_nos, 0], samples[sample_nos, 1], c=color,
            s=markersize,
            alpha=.75, linewidth=.1, cmap=cmap)
    # add a colorbar and label for the colorbar usually we just assume the
    # samples are colored by the pointwise probability density on the data
    # space
    cbar = plt.colorbar()
    cbar.set_label(r'$\rho_\mathcal{D}(q)$')
    # if there is a reference value plot it with a notiable marker
    if type(p_ref) != type(None):
        plt.scatter(p_ref[0], p_ref[1], c='m', s=2*markersize)
    if save:
        plt.autoscale(tight=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename, bbox_inches='tight', transparent=True,
                pad_inches=0)
    if interactive:
        plt.show()
    else:
        plt.close()

def scatter_3D(samples, sample_nos=None, color=None, p_ref=None, save=True,
               interactive=False, xlabel='x', ylabel='y', zlabel='z',
               filename="scatter3d"):
    """ 
    Creates a three-dimensional scatter plot of ``samples`` colored by
    ``color`` (usually an array of pointwise probability density values). A
    reference ``sample`` (``p_ref``) can be chosen by the user. This reference
    ``sample`` will be plotted as a mauve circle twice the size of the other
    markers.
    
    :param samples: Samples to plot. These are the locations in the x-axis,
        y-axis, and z-axis.
    :type samples: :class:`numpy.ndarray`
    :param list sample_nos: indicies of the ``samples`` to plot
    :param color: values to color the ``samples`` by
    :type color: :class:`numpy.ndarray`
    :param p_ref: reference parameter(``sample``) value
    :type p_ref: :class:`numpy.ndarray` of shape (ndim,)
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param string zlabel: z-axis label
    :param string filename: filename to save the figure as

    """

    # plot all of the samples by default
    if type(sample_nos) == type(None):
        sample_nos = np.arange(samples.shape[0])
    # color all of the samples uniformly by default and set the default
    # to the default colormap of matplotlib
    if color is None:
        color = np.ones((samples.shape[0],))
        cmap = None
    else:
        cmap = plt.cm.PuBu
    markersize = 75
    color = color[sample_nos]
    # create the scatter plot for the samples specified by sample_nos
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[sample_nos, 0], samples[sample_nos, 1],
            samples[sample_nos, 2], alpha=.75, linewidth=.1, c=color,
            s=markersize,
            cmap=cmap)
    # add a colorbar and label for the colorbar usually we just assume the
    # samples are colored by the pointwise probability density on the data
    # space
    cbar = plt.colorbar()
    cbar.set_label(r'$\rho_\mathcal{D}(q)$') 
    # if there is a reference value plot it with a notiable marker
    if type(p_ref) != type(None):
        ax.scatter(p_ref[0], p_ref[1], p_ref[2], c='m', s=2*markersize)       
    ax.autoscale(tight=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if save:
        plt.savefig(filename, bbox_inches='tight', transparent=True,
                pad_inches=0)
    if interactive:
        plt.show()
    else:
        plt.close()
   
def show_param(samples, data, rho_D=None, p_ref=None, sample_nos=None,
        save=True, interactive=False, lnums=None, showdim=None):
    r"""
    Create scatter plots of ``samples`` colored by ``color`` (usually
    an array of pointwise probability density values). A reference ``sample``
    (``p_ref``) can be chosen by the user. This reference ``sample`` will be
    plotted as a mauve circle twice the size of the other markers.

    :param samples: Samples to plot
    :type samples: :class:`numpy.ndarray`
    :param data: Data value(s) associated with ``samples``
    :type data: :class:`numpy.ndarray`
    :param list sample_nos: sample numbers to plot
    :param rho_D: probability density function on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`numpy.ndarray`
    :param p_ref: reference parameter value
    :type p_ref: :class:`numpy.ndarray` of shape (ndim,)
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param list lnums: integers representing parameter domain coordinate
        numbers to plot (e.g. i, where :math:`\lambda_i` is a coordinate in the
        parameter space).
    :param int showdim: 2 or 3, flag to determine whether or not to show
        pairwise or tripletwise parameter sample scatter plots in 2 or 3
        dimensions

    """
    # If there is density function given determine the pointwise probability
    # values of each sample based on the value in the data space. Otherwise,
    # color the samples in numerical order.
    if rho_D is not None and data is not None:
        rD = rho_D(data)
    else:
        rD = np.ones(samples.shape[0])
    # If no specific coordinate numbers are given for the parameter coordinates
    # (e.g. i, where \lambda_i is a coordinate in the parameter space), then
    # set them to be the the counting numbers.
    if type(lnums) == type(None):
        lnums = 1+np.array(range(samples.shape[1]))
    # Create the labels based on the user selected parameter coordinates
    xlabel = r'$\lambda_{'+str(lnums[0])+'}$'
    ylabel = r'$\lambda_{'+str(lnums[1])+'}$'
    savename = 'param_samples_cs.eps'
    # Plot 2 or 3 dimensional scatter plots of the samples colored by rD.
    if samples.shape[1] == 2:
        scatter_2D(samples, sample_nos, rD, p_ref, save, interactive, xlabel,
                ylabel, savename)
    elif samples.shape[1] == 3:
        zlabel = r'$\lambda_{'+str(lnums[2])+'}$'
        scatter_3D(samples, sample_nos, rD, p_ref, save, interactive, xlabel,
                ylabel, zlabel, savename)
    elif samples.shape[1] > 2 and showdim == 2:
        for x, y in combinations(lnums, 2):
            xlabel = r'$\lambda_{'+str(x)+'}$'
            ylabel = r'$\lambda_{'+str(y)+'}$'
            savename = 'param_samples_l'+str(x)+'l'+str(y)+'_cs.eps'
            scatter_2D(samples[:, [x-1, y-1]], sample_nos, rD, p_ref, save,
                    interactive, xlabel, ylabel, savename)
    elif samples.shape[1] > 3 and showdim == 3:
        for x, y, z in combinations(lnums, 3):
            xlabel = r'$\lambda_{'+str(x)+'}$'
            ylabel = r'$\lambda_{'+str(y)+'}$'
            zlabel = r'$\lambda_{'+str(z)+'}$'
            savename = 'param_samples_l'+str(x)+'l'+str(y)+'l'+str(z)+'_cs.eps'
            scatter_3D(samples[:, [x-1, y-1, z-1]], sample_nos, rD, p_ref, save,
                    interactive, xlabel, ylabel, zlabel, savename)

def show_data(data, rho_D=None, Q_ref=None, sample_nos=None,
        save=True, interactive=False, Q_nums=None, showdim=None):
    r"""
    Create scatter plots of ``data`` colored by ``color`` (usually
    an array of pointwise probability density values). A reference ``data``
    point (``Q_ref``) can be chosen by the user. This reference ``data`` will
    be plotted as a mauve circle twice the size of the other markers.

    :param data: Data (the data associated with a given set of samples in the
        data space)
    :type data: :class:`numpy.ndarray`
    :param list sample_nos: sample numbers to plot
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`numpy.ndarray`
    :param Q_ref: reference data value
    :type Q_ref: :class:`numpy.ndarray` of shape (mdim,)
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param list lnums: integers representing data domain coordinate
        numbers to plot (e.g. i, where :math:`\q_i` is a coordinate in the
        data space).
    :param int showdim: 2 or 3, flag to determine whether or not to show
        pairwise or tripletwise data sample scatter plots in 2 or 3
        dimensions

    """  
    # If there is density function given determine the pointwise probability
    # values of each sample based on the value in the data space. Otherwise,
    # color the samples in numerical order.
    if rho_D != None:
        rD = rho_D(data)
    else:
        rD = np.ones(data.shape[0])
    # If no specific coordinate numbers are given for the data coordinates
    # (e.g. i, where \q_i is a coordinate in the data space), then
    # set them to be the the counting numbers.
    if  type(Q_nums) == type(None):
        Q_nums = range(data.shape[1])
    # Create the labels based on the user selected data coordinates
    xlabel = r'$q_{'+str(Q_nums[0]+1)+'}$'
    ylabel = r'$q_{'+str(Q_nums[1]+1)+'}$'
    savename = 'data_samples_cs.eps'
    # Plot 2 or 3 dimensional scatter plots of the data colored by rD.
    if data.shape[1] == 2:
        q_ref = None
        if type(Q_ref) == np.ndarray:
            q_ref = Q_ref[Q_nums[:2]]
        scatter_2D(data, sample_nos, rD, q_ref, save, interactive, xlabel,
                ylabel, savename)
    elif data.shape[1] == 3:
        zlabel = r'$q_{'+str(Q_nums[2]+1)+'}$'
        if type(Q_ref) == np.ndarray:
            q_ref = Q_ref[Q_nums[:3]]
        scatter_3D(data, sample_nos, rD, q_ref, save, interactive, xlabel,
                ylabel, zlabel, savename)
    elif data.shape[1] > 2 and showdim == 2:
        for x, y in combinations(Q_nums, 2):
            xlabel = r'$q_{'+str(x+1)+'}$'
            ylabel = r'$q_{'+str(y+1)+'}$'
            savename = 'data_samples_q'+str(x+1)+'q'+str(y+1)+'_cs.eps'
            q_ref = None
            if type(Q_ref) == np.ndarray:
                q_ref = Q_ref[[x, y]]
            scatter_2D(data[:, [x, y]], sample_nos, rD, q_ref, save,
                    interactive, xlabel, ylabel, savename)
    elif data.shape[1] > 3 and showdim == 3:
        for x, y, z in combinations(Q_nums, 3):
            xlabel = r'$q_{'+str(x+1)+'}$'
            ylabel = r'$q_{'+str(y+1)+'}$'
            zlabel = r'$q_{'+str(z+1)+'}$'
            q_ref = None
            if type(Q_ref) == np.ndarray:
                q_ref = Q_ref[[x, y, z]]
            savename = 'data_samples_q'+str(x+1)+'q'+str(y+1)+'q'\
                       +str(z+1)+'_cs.eps'
            scatter_3D(data[:, [x, y, z]], sample_nos, rD, q_ref, save,
                    interactive, xlabel, ylabel, zlabel, savename)

def show_data_domain_multi(samples, data, Q_ref=None, Q_nums=None,
        img_folder='figs/', ref_markers=None,
        ref_colors=None, showdim=None):
    r"""
    Plots 2-D projections of the data domain D using a triangulation based on
    the first two coordinates (parameters) of the generating samples where
    :math:`Q={q_1, q_i}` for ``i=Q_nums``, with a marker for various
    :math:`Q_{ref}`. 

    :param samples: Samples to plot
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim). Only
        uses the first two dimensions.
    :param data: Data associated with ``samples``
    :type data: :class:`numpy.ndarray`
    :param Q_ref: reference data value
    :type Q_ref: :class:`numpy.ndarray` of shape (M, mdim)
    :param list Q_nums: dimensions of the QoI to plot
    :param string img_folder: folder to save the plots to
    :param list ref_markers: list of marker types for :math:`Q_{ref}`
    :param list ref_colors: list of colors for :math:`Q_{ref}`
    :param showdim: default 1. If int then flag to show all combinations with a
        given dimension (:math:`q_i`) or if ``all`` show all combinations.
    :type showdim: int or string

    """
    # Set the default marker and colors
    if ref_markers == None:
        ref_markers = markers
    if ref_colors == None:
        ref_colors = colors
    # If no specific coordinate numbers are given for the data coordinates
    # (e.g. i, where \q_i is a coordinate in the data space), then
    # set them to be the the counting numbers.
    if  type(Q_nums) == type(None):
        Q_nums = range(data.shape[1])
    # If no specific coordinate number of choice is given set to be the first
    # coordinate direction.
    if showdim == None:
        showdim = 0

    # Create a folder for these figures if it doesn't already exist
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    # Make sure the shape of Q_ref is correct
    if Q_ref is not None:
        Q_ref = util.fix_dimensions_data(Q_ref, data.shape[1])

    # Create the triangulization to use to define the topology of the samples
    # in the data space from the first two parameters in the parameter space
    triangulation = tri.Triangulation(samples[:, 0], samples[:, 1])
    triangles = triangulation.triangles

    # Create plots of the showdim^th QoI (q_{showdim}) with all other QoI (q_i)
    if type(showdim) == int:
        for i in Q_nums:
            xlabel = r'$q_{'+str(showdim+1)+r'}$'
            ylabel = r'$q_{'+str(i+1)+r'}$'

            filenames = [img_folder+'domain_q'+str(showdim+1)+'_q'+\
                    str(i+1)+'.eps', img_folder+'q'+str(showdim+1)+\
                    '_q'+str(i+1)+'_domain_Q_cs.eps']
            if Q_ref is not None:    
                show_data_domain_2D(samples, data[:, [showdim, i]], Q_ref[:,
                    [showdim, i]], ref_markers, ref_colors, xlabel=xlabel,
                    ylabel=ylabel, triangles=triangles, save=True,
                    interactive=False, filenames=filenames)
            else:
                show_data_domain_2D(samples, data[:, [showdim, i]], None,
                        ref_markers, ref_colors, xlabel=xlabel, ylabel=ylabel,
                        triangles=triangles, save=True, interactive=False,
                        filenames=filenames)
    # Create plots of all combinations of QoI in 2D 
    elif showdim == 'all' or showdim == 'ALL':
        for x, y in combinations(Q_nums, 2):
            xlabel = r'$q_{'+str(x+1)+r'}$'
            ylabel = r'$q_{'+str(y+1)+r'}$'

            filenames = [img_folder+'domain_q'+str(x+1)+'_q'+str(y+1)+'.eps',
                    img_folder+'q'+str(x+1)+'_q'+str(y+1)+'_domain_Q_cs.eps']
            
            if Q_ref is not None:
                show_data_domain_2D(samples, data[:, [x, y]], Q_ref[:, [x, y]],
                        ref_markers, ref_colors, xlabel=xlabel, ylabel=ylabel,
                        triangles=triangles, save=True, interactive=False,
                        filenames=filenames)
            else:
                show_data_domain_2D(samples, data[:, [x, y]], None,
                        ref_markers, ref_colors, xlabel=xlabel, ylabel=ylabel,
                        triangles=triangles, save=True, interactive=False,
                        filenames=filenames)

def show_data_domain_2D(samples, data, Q_ref=None, ref_markers=None,
        ref_colors=None, xlabel=r'$q_1$', ylabel=r'$q_2$',
        triangles=None, save=True, interactive=False, filenames=None):
    r"""
    Plots 2-D a single data domain D using a triangulation based on the first
    two coordinates (parameters) of the generating samples where :math:`Q={q_1,
    q_i}` for ``i=Q_nums``, with a marker for various :math:`Q_{ref}`. Assumes
    that the first dimension of data is :math:`q_1`.


    :param samples: Samples to plot
    :type samples: :class:`~numpy.ndarray` of shape (num_samples, ndim). Only
        uses the first two dimensions.
    :param data: Data associated with ``samples``
    :type data: :class:`numpy.ndarray`
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

    """
    # Set the default marker and colors
    if ref_markers == None:
        ref_markers = markers
    if ref_colors == None:
        ref_colors = colors
    # If no specific coordinate numbers are given for the data coordinates
    # (e.g. i, where \q_i is a coordinate in the data space), then
    # set them to be the the counting numbers.
    if type(triangles) == type(None):
        triangulation = tri.Triangulation(samples[:, 0], samples[:, 1])
        triangles = triangulation.triangles
    # Set default file names
    if filenames == None:
        filenames = ['domain_q1_q2_cs.eps', 'q1_q2_domain_Q_cs.eps']
    
    # Make sure the shape of Q_ref is correct
    if Q_ref is not None:
        Q_ref = util.fix_dimensions_data(Q_ref, 2)

    # Create figure
    plt.tricontourf(data[:, 0], data[:, 1], np.zeros((data.shape[0],)),
            triangles=triangles, colors='grey') 
    plt.autoscale(tight=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filenames[0], bbox_inches='tight', transparent=True,
            pad_inches=.2)
    # Add truth markers
    if Q_ref is not None:
        for i in xrange(Q_ref.shape[0]):
            plt.scatter(Q_ref[i, 0], Q_ref[i, 1], s=60, c=ref_colors[i],
                marker=ref_markers[i])
    if save:
        plt.savefig(filenames[1], bbox_inches='tight', transparent=True,
            pad_inches=.2)
    if interactive:
        plt.show()
    else:
        plt.close()

def scatter_param_multi(samples, img_folder='figs/', showdim='all', save=True,
        interactive=False):
    r"""

    Creates two-dimensional projections of scatter plots of ``samples``.
    
    :param samples: Samples to plot. 
    :type samples: :class:`numpy.ndarray`
    :param bool save: flag whether or not to save the figure
    :param bool interactive: flag whether or not to show the figure
    :param string img_folder: folder to save the plots to
    :param showdim: default 1. If int then flag to show all combinations with a
        given dimension (:math:`\lambda_i`) or if ``all`` show all combinations.
    :type showdim: int or string

    """
    # If no specific coordinate number of choice is given set to be the first
    # coordinate direction.
    if showdim == None:
        showdim = 0
    # Create a folder for these figures if it doesn't already exist
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)
    # Create list of all the parameter coordinates
    L_nums = range(samples.shape[1])

   # Create plots of the showdim^th parameter (\lambda_{showdim}) with all the
   # other parameters
    if type(showdim) == int:
        for i in L_nums:
            xlabel = r'$\lambda_{'+str(showdim+1)+r'}$'
            ylabel = r'$\lambda_{'+str(i+1)+r'}$'

            filenames = [img_folder+'domain_l'+str(showdim+1)+'_l'+\
                    str(i+1)+'.eps', img_folder+'l'+str(showdim+1)+\
                    '_l'+str(i+1)+'_domain_L_cs.eps']
            filename = filenames[0]
            plt.scatter(samples[:, 0], samples[:, 1])
            if save:
                plt.autoscale(tight=True)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.savefig(filename, bbox_inches='tight', transparent=True,
                        pad_inches=0)
            if interactive:
                plt.show()
            else:
                plt.close()
    # Create plots of all of the possible pairwise combinations of parameters
    elif showdim == 'all' or showdim == 'ALL':
        for x, y in combinations(L_nums, 2):
            xlabel = r'$\lambda_{'+str(x+1)+r'}$'
            ylabel = r'$\lambda_{'+str(y+1)+r'}$'

            filenames = [img_folder+'domain_l'+str(x+1)+'_l'+\
                    str(y+1)+'.eps', img_folder+'l'+str(x+1)+\
                    '_l'+str(y+1)+'_domain_L_cs.eps']
            filename = filenames[0]
            plt.scatter(samples[:, x], samples[:, y])
            if save:
                plt.autoscale(tight=True)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.savefig(filename, bbox_inches='tight', transparent=True,
                        pad_inches=0)
            if interactive:
                plt.show()
            else:
                plt.close()

def scatter2D_multi(samples, color=None, p_ref=None, img_folder='figs/',
                    filename=None, label_char=r'$\lambda', showdim=None):
    r"""
    Creates two-dimensional projections of scatter plots of ``samples`` colored
    by ``color`` (usually an array of pointwise probability density values). A
    reference ``sample`` (``p_ref``) can be chosen by the user. This reference
    ``sample`` will be plotted as a mauve circle twice the size of the other
    markers.

    :param samples: Samples to plot. 
    :type samples: :class:`numpy.ndarray`
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

    """
    # If no specific coordinate number of choice is given set to be the first
    # coordinate direction.
    if showdim == None:
        showdim = 0
    # Create a folder for these figures if it doesn't already exist
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)
    # Create list of all the parameter coordinates    
    p_nums = range(samples.shape[1])

   # Create plots of the showdim^th parameter (\lambda_{showdim}) with all the
   # other parameters
    if type(showdim) == int:
        for i in p_nums:
            xlabel = label_char+r'_{'+str(showdim+1)+r'}$'
            ylabel = label_char+r'_{'+str(i+1)+r'}$'

            postfix = '_d'+str(showdim+1)+'_d'+str(i+1)+'.eps'
            myfilename = os.path.join(img_folder, filename+postfix)

            scatter_2D(samples[:, [showdim, i]], sample_nos=None, color=color,
                       p_ref=p_ref[[showdim, i]], save=True, interactive=False,
                       xlabel=xlabel, ylabel=ylabel, filename=myfilename)
    # Create plots of all of the possible pairwise combinations of parameters
    elif showdim == 'all' or showdim == 'ALL':
        for x, y in combinations(p_nums, 2):
            xlabel = label_char+r'_{'+str(x+1)+r'}$'
            ylabel = label_char+r'_{'+str(y+1)+r'}$'

            postfix = '_d'+str(x+1)+'_d'+str(y+1)+'.eps'
            myfilename = os.path.join(img_folder, filename+postfix)

            scatter_2D(samples[:, [x, y]], sample_nos=None, color=color,
                       p_ref=p_ref[[x, y]], save=True, interactive=False,
                       xlabel=xlabel, ylabel=ylabel, filename=myfilename)
