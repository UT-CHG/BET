# Copyright (C) 2014-2020 The BET Development Team

"""
Butler, Estep, Tavener Method

This package provides tools for solving stochastic inverse problems using a
measure-theoretic. It is named for the developers of the key algorithm in
:mod:`bet.calculateP.calculateP`.

* :mod:`~bet.Comm` provides a work around for users who do not which to install :program:``mpi4py``.
* :mod:`~bet.util` provides some general use methods for creating grids, checking/fixing dimensions, and globalizing arrays.
* :mod:`~bet.calculateP` provides tools to approximate probabilities.
* :mod:`~bet.sampling` provides various sampling algorithms.
* :mod:`~bet.sensitivity` provides tools for approximating derivatives and optimally choosing quantities of interest.
* :mod:`~bet.postProcess` provides plotting tools and tools to sort samples by probabilities.
* :mod:`~bet.sample` provides data structures to store sets of samples and their associated arrays.
* :mod:`~bet.surrogates` provides methods for generating and using
    surrogate models.

"""

__all__ = ['sampling', 'calculateP', 'postProcess', 'sensitivity', 'util',
           'Comm', 'sample', 'surrogates']
