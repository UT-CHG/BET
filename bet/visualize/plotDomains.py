import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Methods used to plot two-dimensional domains and/or two-dimensional
slices/projections of domains.
"""

def scatter_2D(samples, sample_nos, color, p_true, save, show,
        xlabel, ylabel, filename):
    """
    Two-dimensional scatter plot of ``samples`` colored by ``color`` (usually
    an array of rho_D values).
    
    :param samples: Samples to plot
    :type samples: :class:`np.ndarray`
    :param list sample_nos: sample numbers to plot
    :param color: array to color the samples by
    :type color: :class:`np.ndarray`
    :param p_true: true parameter value
    :type p_true: :class:`np.ndarray`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param string filename: filename to save the figure as

    """
    if sample_nos==None:
        sample_nos = range(samples.shape[0])
    color = color[sample_nos]
    plt.scatter(samples[sample_nos, 0],samples[sample_nos, 1],c=color,
            cmap=plt.cm.Oranges_r)
    plt.colorbar()
    if p_true != None:
        plt.scatter(p_true[0], p_true[1], c='b')
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

def scatter_3D(samples, sample_nos, color, p_true, save, show,
        xlabel, ylabel, zlabel, filename):
    """
    Three-dimensional scatter plot of ``samples`` colored by ``color`` (usually
    an array of rho_D values).
    
    :param samples: Samples to plot
    :type samples: :class:`np.ndarray`
    :param list sample_nos: sample numbers to plot
    :param color: array to color the samples by
    :type color: :class:`np.ndarray`
    :param p_true: true parameter value
    :type p_true: :class:`np.ndarray`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param string zlabel: z-axis label
    :param string filename: filename to save the figure as

    """
    if sample_nos==None:
        sample_nos = range(samples.shape[0])
    color = color[sample_nos]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[sample_nos, 0], samples[sample_nos, 1],
            samples[sample_nos, 2], c=color, cmap=plt.cm.Oranges_r)
    #ax.colorbar()
    if p_true != None:
        ax.scatter(p_true[0], p_true[1], p_true[2], c='b')
    if save:
        ax.autoscale(tight=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.savefig(filename, bbox_inches='tight', transparent=True,
                pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()
   
def show_param(samples, data, rho_D = None, p_true=None,
        sample_nos=None, save=True, show=True):
    """
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
    :param p_true: true parameter value
    :type p_true: :class:`np.ndarray`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure

    """
   
    if rho_D!=None:
        rD = rho_D(data)
    xlabel = r'$\lambda_1$'
    ylabel = r'$\lambda_2$'
    savename = 'param_samples_cs.eps'
    if data.shape[1]==2:
        scatter_2D(samples, sample_nos, rD, p_true, save, show, xlabel, ylabel,
                savename)
    elif data.shape[1]==3:
        zlabel = r'$\lambda_3$'
        scatter_3D(samples, sample_nos, rD, p_true, save, show, xlabel, ylabel,
                zlabel, savename)

def show_data(data, rho_D=None, Q_true=None, sample_nos=None,
        save=True, show=True):
    """
    Plot samples in data space and colors them either by rho_D or by
    sample batch number.

    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param list sample_nos: sample numbers to plot
    :param rho_D: probability density on D
    :type rho_D: callable function that takes a :class:`np.array` and returns a
        :class:`np.ndarray`
    :param Q_true: true data value
    :type Q_true: :class:`np.ndarray`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure

    """   
    if rho_D!=None:
        rD = rho_D(data)
    xlabel = r'$q_1$'
    ylabel = r'$q_2$'
    savename = 'data_samples_cs.eps'
    if data.shape[1]==2:
        scatter_2D(data, sample_nos, rD, Q_true, save, show, xlabel, ylabel,
            savename)
    elif data.shape[1]==3:
        zlabel = r'$q_3$'
        scatter_3D(data, sample_nos, rD, Q_true, save, show, xlabel, ylabel,
                zlabel, savename)

def show_data_domain_multi(samples, data, Q_true, Q_nums=None,
        img_folder='figs/', true_markers=['^', 's', 'o'],
        true_colors=['r', 'g', 'b']):
    """
    Plot the data domain D using a triangulation based on the generating
    samples where $Q={q_1, q_i}$ for ``i=Q_nums``, with a marker for various
    ``Q_true``. 

    :param samples: Samples to plot
    :type samples: :class:`~numpy.ndarray` of shape (ndim, num_samples)
    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param Q_true: true data value
    :type Q_true: :class:`np.ndarray`
    :param list Q_nums: dimensions of the QoI to plot
    :param string img_folder: folder to save the plots to
    :param list true_markers: list of marker types for ``Q_true``
    :param list true_colors: list of colors for ``Q_true``

    """
    
    if  Q_nums == None:
        Q_nums = range(data.shape[1])

    triangulation = tri.Triangulation(samples[0,:], samples[1,:])
    triangles = triangulation.triangles

    for i in Q_nums:
        
        plt.tricontourf(data[:,0],data[:,i], np.zeros((data.shape[0],)),
                triangles=triangles, colors = 'grey') 
        plt.autoscale(tight=True)
        plt.xlabel(r'$q_1$')
        ylabel=r'$q_{'+str(i+1)+r'}$'

        filenames = [img_folder+'domain_q1_'+str(i)+'.eps',
                img_folder+'q1_q'+str(i)+'_domain_Q_cs.eps']
            
        show_data_domain_2D(samples, data[:,[0, i]], Q_true[:,[0, i]],
            true_markers, true_colors, ylabel=r'$q_{'+str(i+1)+r'}$',
            triangles=triangles, save=True, show=False, filenames=filenames)

def show_data_domain_2D(samples, data, Q_true, true_markers=['^', 's', 'o'],
        true_colors=['r', 'g', 'b'], xlabel=r'$q_1$', ylabel=r'$q_2',
        triangles=None, save=True, show=True, filenames=None):
    """
    Plot the data domain D using a triangulation based on the generating
    samples with a marker for various ``Q_true``. Assumes that the first
    dimension of data is $q_1$.

    :param samples: Samples to plot
    :type samples: :class:`~numpy.ndarray` of shape (ndim, num_samples)
    :param data: Data associated with ``samples``
    :type data: :class:`np.ndarray`
    :param Q_true: true data value
    :type Q_true: :class:`np.ndarray`
    :param list true_markers: list of marker types for ``Q_true``
    :param list true_colors: list of colors for ``Q_true``
    :param string xlabel: x-axis label
    :param string ylabel: y-axis label
    :param triangles: triangulation defined by ``samples``
    :type triangles: :class:`tri.Triuangulation.triangles`
    :param boolean save: flag whether or not to save the figure
    :param boolean show: flag whether or not to show the figure
    :param list filenames: file names for the unmarked and marked domain plots

    """

    if triangles==None:
        triangulation = tri.Triangulation(samples[:,0],samples[:,1])
        triangles = triangulation.triangles
    if filenames==None:
        filenames = ['domain_q1_q2_cs.eps', 'q1_q2_domain_Q_cs.eps']
    # Create figure
    plt.tricontourf(data[:,0],data[:,1], np.zeros((data.shape[0],)),
            triangles=triangles, colors = 'grey') 
    plt.autoscale(tight=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filenames[0], bbox_inches='tight', transparent=True,
            pad_inches=0)
    # Add truth markers
    for i in xrange(Q_true.shape[0]):
        plt.scatter(Q_true[i,0],Q_true[i,1], s = 60, c=true_colors[i],
                marker=true_markers[i])
    if save:
        plt.savefig(filenames[1], bbox_inches='tight', transparent=True,
            pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()




