# Copyright (C) 2014-2020 The BET Development Team

r"""
This subpackage provides classes and methods for calculating the
probability measure :math:`P_{\Lambda}`.

* :mod:`~bet.calculateP.calculateP` provides methods for approximating probability densities in the measure-theoretic framework.
* :mod:`~bet.calculateP.simpleFunP` provides methods for creating simple function approximations of probability densities for the measure-theoretic framework.
* :mod:`~bet.calculateP.dataConsistent` provides methods for data-consistent stochastic inversion.
* :mod:`~bet.calculateP.calculateError` provides methods for approximating numerical and sampling errors.
"""
__all__ = ['calculateP', 'simpleFunP', 'calculateError', 'dataConsistent']
