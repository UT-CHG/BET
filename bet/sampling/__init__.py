# Copyright (C) 2014-2015 The BET Development Team

"""
This subpackage contains

* :class:`~bet.sampling.basicSampling.sampler` a general class and associated
    set of methods that interogates a model through a
    :class:`~bet.loadBalance.load_balance` interface.
    :class:`~bet.sampling.basicSampling.sampler` requests data(QoI) at a
    specified set of parameter samples.
* :class:`bet.sampling.adaptiveSampling.sampler` inherits from
    :class:`~bet.sampling.basicSampling.sample` adaptively generates samples
    according to a probability density function.

The following developmental modules are OPERATIONAL:
    * :mod:`~bet.sampling.smoothIndicatorFunction`
    * :mod:`~bet.sampling.boundarySampling`
    * :mod:`~bet.sampling.gradientSampling`
    * :mod:`~bet.sampling.limitedMemorySampling`

The following developmental modules are OPERATIONAL but not recommended:
    * :mod:`~bet.sampling.slopedIndicatorFunction`
    * :mod:`~bet.sampling.reseedSampling`
    * :mod:`~bet.sampling.surrogateSampling` (possibly BUGGY)

The following developmental modules not NOT OPERATIONAL:
    * :mod:`~bet.sampling.dev_multi_dist_kernel`

"""
__all__ = ['basicSampling', 'adaptiveSampling', 'dev_multi_dist_kernel',
    'slopedIndicatorFunction', 'smoothedIndicatorFunction', 'boundarySampling',
    'limitedMemorySampling', 'surrogateSampling', 'reseedSampling',
    'gradientSampling']
