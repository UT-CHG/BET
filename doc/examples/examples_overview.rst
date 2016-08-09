.. _examples:

=======================================
Some References and Examples
=======================================

For more information about the method and algorithm, see `A Measure-Theoretic
Computational Method for Inverse Sensitivity Problems III: Multiple Quantities of Interest 
<http://dx.doi.org/10.1137/130930406>`_ for the formulation of the stochastic
inverse problem in a measure theoretic framework along with proofs of existence
and uniqueness of solutions, `Solving Stochastic Inverse Problems using Sigma-Algebras on Contour Maps 
<http://arxiv.org/abs/1407.3851>`_ for the convergence 
and error analysis of the non-intrusive algorithm, and
`Definition and solution of a stochastic inverse problem for the Manning’s n parameter field in 
hydrodynamic models <http://dx.doi.org/10.1016/j.advwatres.2015.01.011>`_ for a less technical description
of the method for engineers as well as application to a physically relevant problem
in coastal ocean modeling. 

All of the example listed here and more are located in the ``BET/examples/``
directory.


Validation example
=======================================

See :ref:`validation` for an example.


Linear Map Example
=======================================

See :ref:`linearMap` for an example using a linear map.

Non-Linear Map Example
=======================================

See :ref:`nonlinearMap` for an example using a nonlinear map.

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


(Batch) Adaptive Sampling Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These illustrate how to perform a specific type of goal-oriented
adaptive sampling on a linear interpolant
created from data read from file.
We also show how several methods within the module
:mod:`~bet.postProcess.plotDomains` can be used to
plot 2D domains and/or 2D slices and projections of higher dimensional domains.

    * :ref:`fromFile2DExample`
    * :ref:`fromFile3DExample`

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
 
