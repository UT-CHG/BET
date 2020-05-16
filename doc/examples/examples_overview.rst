.. _examples:

=======================================
Examples
=======================================
All of the examples listed here and more are located in the ``BET/examples/`` directory.

Getting Started: Measure Theoretic Stochastic Inversion
=======================================

See :ref:`validation` for a basic example involving measure-theoretic stochastic inversion.

Getting Started: Data-Consistent Stochastic Inversion
=======================================

See `here <https://github.com/smattis/BET-1/blob/v3-steve/examples/linearMap/linearMapDataConsistent.py>`_ for a basic
example involving Data-Consistent Stochastic Inversion for a linear map.

Linear Map Example
=======================================

See :ref:`linearMap` for an example using a linear map involving measure-theoretic stochastic inversion.

Non-Linear Map Example
=======================================

See :ref:`nonlinearMap` for an example using a nonlinear map involving measure-theoretic stochastic inversion.

See `here <https://github.com/smattis/BET-1/blob/v3-steve/examples/nonlinearMap/nonlinearMapDataConsistent.py>`_  for an example using a nonlinear map with data-consistent inversion.

FEniCS Example (serial BET and serial model)
=============================================

A completely serial example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`fenicsExample` for an example using the `FEniCS package
<http://fenicsproject.org/>`_ that can be run with serial BET.

Using Launcher to run multiple serial models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`fenicsMultipleSerialExample` for an example that can be run with
serial BET and uses `Launcher <https://github.com/TACC/launcher>`_ to
launch multiple serial runs of the model in parallel.

ADCIRC on an Idealized Inlet Examples and Adaptive Sampling
===========================================================

The files for these examples can be found in ``examples/fromADCIRC_FileMap``.

For a description of the model, physical inlet domain, data space, and parameter
space for the examples using the idealized inlet see `Definition and solution
of a stochastic inverse problem for the Manning’s n parameter field in
hydrodynamic models <http://dx.doi.org/10.1016/j.advwatres.2015.01.011>`_.


Examples Estimating :math:`P_\Lambda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These illustrate how to actually take a discretization object
(loaded from file) and solve the stochastic inverse problem
using different QoI maps.

    * :ref:`q1D`
    * :ref:`q2D`
    * :ref:`q3D`

Contaminant Transport Example
==============================
See :ref:`contaminantTransport` for an example.

Choosing Optimal QoIs Examples
==============================
The files for these examples can be found in ``examples/sensitivity``.

See :ref:`chooseQoIs` for an example based on optimizing the space-time
locations of two temperature measurements on a thin metal plate with
spatially variable thermal diffusivity. The goal is to optimize the QoI map with
respect to a geometric property related to numerical accuracy in computing the
solution to the stochastic inverse problem with a finite number of samples.

See :ref:`linear_sensitivity` for an example based on optimizing
a QoI map from a space of linear QoI maps under different optimization criteria.


List of all examples
====================
.. toctree::
   :maxdepth: 1
   :glob:

   example_rst_files/*
 
