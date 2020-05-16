# Copyright (C) 2014-2020 The BET Development Team

"""
This subpackage contains

* :mod:`~bet.sampling.basicSampling` a general class and associated set of methods that sample spaces and solve models through an interface.
* :class:`~bet.sampling.basicSampling.sampler` requests data (QoI) at a specified set of parameter samples.
* :mod:`~bet.sampling.LpGeneralizedSamples` provides methods for sampling on balls in Lp spaces.
* :mod:`~bet.sampling.useLUQ` provides methods for interfacing with the LUQ package.
"""
__all__ = ['basicSampling', 'LpGeneralizedSamples', 'useLUQ']
