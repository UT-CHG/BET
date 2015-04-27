.. _parallel:

========
Parallel
========

Installation
------------

Running this code in parallel requires the installation of `MPI for Python
<http://mpi4py.scipy.org/>`_ which requires that your system has mpi installed.

Affected Packages
-----------------

The modules that have parallel capabilities are as follows::

  bet/
    util
    calculateP/
      calculateP
      simpleFunP
    sampling/
      basicSampling 
      adaptiveSampling
    postProcess/
      plotP  
      postTools

util
~~~~
The module :mod:`~bet.util` provides the method
:meth:`~bet.util.get_global_values` to globalize local arrays into an array of
global values on all processors.

calculateP
~~~~~~~~~~
All methods in the module :mod:`~bet.calculateP.calculateP` benifit from
parallel execution. Only local arrays are returned for ``P``, use
:meth:`~bet.util.get_global_values` to globalize local arrays.

In the module :mod:`~bet.calculateP.simpleFunP` the methods
:meth:`~bet.calculateP.simpleFunP.unif_unif`,
:meth:`~bet.calculateP.simpleFunP.normal_normal`, and 
:meth:`~bet.calculateP.simpleFunP.unif_normal` benifit from parallel
execution.

sampling
~~~~~~~~
