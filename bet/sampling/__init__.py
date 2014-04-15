"""
This subpackage contains

* :class:`sampling` a general class and associated set of methods that
    interogates a model through a :class:`~bet.loadBalance.load_balance`
    interface.  :class:`sampling` requests data(QoI) at a specified set of
    parameter samples.
* :class:`adaptive_sampling` inherits from :class:`~bet.sampling.sampling`
    adaptively generates samples.
"""
__all__ = ['sampling', 'adaptive_sampling']
