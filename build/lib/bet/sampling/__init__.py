# Copyright (C) 2014-2015 The BET Development Team

"""
This subpackage contains

* :class:`~bet.sampling.basicSampling` a general class and associated set of
    methods that interogates a model through a
    :class:`~bet.loadBalance.load_balance` interface.
    :class:`~bet.sampling.basicSampling.sampler` requests data(QoI) at a
    specified set of parameter samples.
* :class:`bet.sampling.adaptiveSampling` inherits from
    :class:`~bet.sampling.basicSampling` adaptively generates samples.
"""
__all__ = ['basicSampling', 'adaptiveSampling']
