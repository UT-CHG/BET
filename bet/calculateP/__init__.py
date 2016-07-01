# Copyright (C) 2014-2015 The BET Development Team

r""" 
This subpackage provides classes and methods for calulating the
probability measure :math:`P_{\Lambda}`.

* :mod:`~bet.calculateP.calculateP` provides methods for approximating
     probability densities 
* :mod:`~bet.calculateP.simpleFunP` provides methods for creating simple
    function approximations of probability densisties
* :mod:`~bet.calculateP.voronoiHistogram` provides methods for creating the
    generating points for cells that define a regular grid for use by
    :meth:`numpy.histogramdd` and for determining their volumes, etc. This
    module is only for use by :mod:`~bet.calculateP.simpleFunP`.  
* :mod:`~bet.calculateP.indicatorFunctions` provides methods for creating
    indicator functions for use by various other classes.
"""
__all__ = ['calculateP', 'simpleFunP', 'voronoiHistogram', 'indicatorFunctions',
           'calculateError']
