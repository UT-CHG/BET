# Copyright (C) 2014-2016 The BET Development Team

"""
Butler, Estep, Tavener Method

This package provides tools for solving stochastic inverse problems using a
measure-theoretic. It is named for the developers of the key algorithm in
:mod:`bet.calculateP.calculateP`.

Comm :mod:`~bet.Comm` provides a work around for users who do not which to install
    :program:``mpi4py``.

util :mod:`~bet.util` provides some general use methods for creating grids,
    checking/fixing dimensions, and globalizing arrays.

calculateP :mod:`~bet.calculateP` provides tools to approximate probabilities.

sampling :mod:`~bet.sampling` provides various sampling algorithms.

sensitivity :mod:`~bet.sensitivity` provides tools for approximating
    derivatives and optimally choosing quantities of interest.

postProcess :mod:`~bet.postProcess` provides plotting tools and tools to sort
    samples by probabilities.

sample :mod:`~bet.sample` provides data structures to store sets of samples and
    their associated arrays.

"""

__all__ = ['sampling', 'calculateP', 'postProcess', 'sensitivity', 'util', 
           'Comm', 'sample', 'surrogates']
