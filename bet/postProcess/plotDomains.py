"""
This module provides methods used to plot two-dimensional domains and/or
two-dimensional slices/projections of domains.

TODO : match method signatures with style in plotP
"""

import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

def scatter_2D(samples, sample_nos, color, p_ref, save, show,
        xlabel, ylabel, filename):
    """
    Two-dimensional scatter plot of ``samples`` colored by ``color`` (usually
    an array of rho_D values).
    
    :param samples: Samples to plot
    :type samples: :class:`np.ndarray`
    :param list sample_nos: sample numbers to plot
    :param color: array to color the samples by
    :type color: :class:`np.ndarray`
    :param p_ref: reference parameter value
    :type p_ref: :class:`np.ndarray`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param string filename: filename to save the figure as

    """
    if sample_nos == None:
        sample_nos = range(samples.shape[0])
    color = color[sample_nos]
    plt.scatter(samples[sample_nos, 0], samples[sample_nos, 1], c=color, s=10,
            alpha=.75, linewidth=.1, cmap=plt.cm.Oranges)
    cbar = plt.colorbar()
    cbar.set_label(r'$\rho_\mathcal{D}(Q)$')
    if p_ref != None:
        plt.scatter(p_ref[0], p_ref[1], c='g')
    if save:
        plt.autoscale(tight=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename, bbox_inches='tight', transparent=True,
                pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()

def scatter_3D(samples, sample_nos, color, p_ref, save, show,
        xlabel, ylabel, zlabel, filename):
    """
    Three-dimensional scatter plot of ``samples`` colored by ``color`` (usually
    an array of rho_D values).
    
    :param samples: Samples to plot
    :type samples: :class:`np.ndarray`
    :param list sample_nos: sample numbers to plot
    :param color: array to color the samples by
    :type color: :class:`np.ndarray`
    :param p_ref: reference parameter value
    :type p_ref: :class:`np.ndarray`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param string zlabel: z-axis label
    :param string filename: filename to save the figure as

    """
    if sample_nos == None:
        sample_nos = range(samples.shape[0])
    color = color[sample_nos]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[sample_nos, 0], samples[sample_nos, 1],
            samples[sample_nos, 2], s=10, alpha=.75, linewidth=.1, c=color,
            cmap=plt.cm.Oranges)
    if p_ref != None:
        ax.scatter(p_ref[0], p_ref[1], p_ref[2], c='g')
        
    ax.autoscale(tight=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if save:
        plt.savefig(filename, bbox_inches='tight', transparent=True,
                pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()
   
def show_param(samples, data, rho_D=None, p_ref=None, sample_nos=None,
        save=True, show=False, lnums=None):
    """
    TODO: add options for higher dimensional visualization or at least for
    showing pairwise or tripletwise parameter point clouds

    Plot samples in parameter space and colors them either by rho_D or by
    sample batch number.

    :param samples: Samples to plot
    :type samples: :class:`np.ndarray`
    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param list sample_nos: sample numbers to plot
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`np.ndarray`
    :param p_ref: reference parameter value
    :type p_ref: :class:`np.ndarray`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure

    """
   
    if rho_D != None:
        rD = rho_D(data)
    if lnums == None:
        lnums = 1+np.array(range(data.shape[1]))
    xlabel = r'$\lambda_{'+str(lnums[0])+'}$'
    ylabel = r'$\lambda_{'+str(lnums[1])+'}$'
    savename = 'param_samples_cs.eps'
    if data.shape[1] == 2:
        scatter_2D(samples, sample_nos, rD, p_ref, save, show, xlabel, ylabel,
                savename)
    elif data.shape[1] == 3:
        zlabel = r'$\lambda_{'+str(lnums[2])+'}$'
        scatter_3D(samples, sample_nos, rD, p_ref, save, show, xlabel, ylabel,
                zlabel, savename)

def show_data(data, rho_D=None, Q_ref=None, sample_nos=None,
        save=True, show=False, Q_nums=None):
    """
    TODO: add options for higher dimensional visualization or at least for
    showing pairwise or tripletwise sample point clouds

    Plot samples in data space and colors them either by rho_D or by
    sample batch number.

    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param list sample_nos: sample numbers to plot
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`np.ndarray`
    :param Q_ref: reference data value
    :type Q_ref: :class:`np.ndarray`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure

    """   
    if rho_D != None:
        rD = rho_D(data)
    if  Q_nums == None:
        Q_nums = range(data.shape[1])
    xlabel = r'$q_{'+str(Q_nums[0]+1)+'}$'
    ylabel = r'$q_{'+str(Q_nums[1]+1)+'}$'
    savename = 'data_samples_cs.eps'
    if data.shape[1] == 2:
        scatter_2D(data, sample_nos, rD, Q_ref, save, show, xlabel, ylabel,
            savename)
    elif data.shape[1] == 3:
        zlabel = r'$q_{'+str(Q_nums[2]+1)+'}$'
        scatter_3D(data, sample_nos, rD, Q_ref, save, show, xlabel, ylabel,
                zlabel, savename)

def show_data_domain_multi(samples, data, Q_ref, Q_nums=None,
        img_folder='figs/', ref_markers=None,
        ref_colors=None):
    r"""
    TODO: add option for showing all pairs or altering which index for which
    all pairs are shown

    Plot the data domain D using a triangulation based on the generating
    samples where :math:`Q={q_1, q_i}` for ``i=Q_nums``, with a marker for
    various :math:`Q_{ref}`. 

    :param samples: Samples to plot
    :type samples: :class:`~numpy.ndarray` of shape (ndim, num_samples)
    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param Q_ref: reference data value
    :type Q_ref: :class:`np.ndarray`
    :param list Q_nums: dimensions of the QoI to plot
    :param string img_folder: folder to save the plots to
    :param list ref_markers: list of marker types for :math:`Q_{ref}`
    :param list ref_colors: list of colors for :math:`Q_{ref}`

    """
    if ref_markers == None:
        ref_markers = markers
    if ref_colors == None:
        ref_colors = colors
    if  Q_nums == None:
        Q_nums = range(data.shape[1])

    triangulation = tri.Triangulation(samples[0, :], samples[1, :])
    triangles = triangulation.triangles

    for i in Q_nums:
        
        plt.tricontourf(data[:, 0], data[:, i], np.zeros((data.shape[0],)),
                triangles=triangles, colors='grey') 
        plt.autoscale(tight=True)
        plt.xlabel(r'$q_1$')
        plt.ylabel(r'$q_{'+str(i+1)+r'}$')

        filenames = [img_folder+'domain_q1_'+str(i)+'.eps',
                img_folder+'q1_q'+str(i)+'_domain_Q_cs.eps']
            
        show_data_domain_2D(samples, data[:, [0, i]], Q_ref[:, [0, i]],
            ref_markers, ref_colors, ylabel=r'$q_{'+str(i+1)+r'}$',
            triangles=triangles, save=True, show=False, filenames=filenames)

def show_data_domain_2D(samples, data, Q_ref, ref_markers=None,
        ref_colors=None, xlabel=r'$q_1$', ylabel=r'$q_2',
        triangles=None, save=True, show=True, filenames=None):
    r"""
    Plot the data domain D using a triangulation based on the generating
    samples with a marker for various :math:`Q_{ref}`. Assumes that the first
    dimension of data is :math:`q_1`.

    :param samples: Samples to plot
    :type samples: :class:`~numpy.ndarray` of shape (ndim, num_samples)
    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param Q_ref: reference data value
    :type Q_ref: :class:`np.ndarray`
    :param list ref_markers: list of marker types for :math:`Q_{ref}`
    :param list ref_colors: list of colors for :math:`Q_{ref}`
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param triangles: triangulation defined by ``samples``
    :type triangles: :class:`tri.Triuangulation.triangles`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure
    :param list filenames: file names for the unmarked and marked domain plots

    """
    if ref_markers == None:
        ref_markers = markers
    if ref_colors == None:
        ref_colors = colors
    if triangles == None:
        triangulation = tri.Triangulation(samples[:, 0], samples[:, 1])
        triangles = triangulation.triangles
    if filenames == None:
        filenames = ['domain_q1_q2_cs.eps', 'q1_q2_domain_Q_cs.eps']
    # Create figure
    plt.tricontourf(data[:, 0], data[:, 1], np.zeros((data.shape[0],)),
            triangles=triangles, colors='grey') 
    plt.autoscale(tight=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filenames[0], bbox_inches='tight', transparent=True,
            pad_inches=0)
    # Add truth markers
    for i in xrange(Q_ref.shape[0]):
        plt.scatter(Q_ref[i, 0], Q_ref[i, 1], s=60, c=ref_colors[i],
                marker=ref_markers[i])
    if save:
        plt.savefig(filenames[1], bbox_inches='tight', transparent=True,
            pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()




