# Copyright (C) 2014-2015 The BET Development Team

r"""
This subpackage contains

* :mod:`~bet.postProcess.plotP` plots :math:`P` and/or volumes (:math:`\mu`)
    of voronoi cells
* :mod:`~bet.postProcess.plotDomains` plots the data domain
    :math:`\mathcal{D}` in 2D
* :mod:`~bet.postProcess.postTools` has tools for postprocessing
* :mod:`~bet.postProcess.dev_volume2d` has tools for determining the exact
    volume of :math:`A := Q^{-1}(B) \in \Lambda` where :math:`B \in
    \mathcal{D}` is a polygon defined by the Delunay triangulation of a set of
    points. Both the parameter and data space must be 2D.

"""
__all__ = ['plotP', 'plotDomains', 'postTools', 'dev_volume2d']
