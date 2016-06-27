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

See :ref:`fenicsExample` for an example using the `FEniCS package
<http://fenicsproject.org/>`_ that can be run with serial BET.

See :ref:`fenicsMultipleSerialExample` for an example that can be run with
serial BET and uses `Launcher <https://github.com/TACC/launcher>`_ to
launch multiple serial runs of the model in parallel.

ADCIRC Based Examples
==============================================

The files for these examples can be found in ``examples/fromADCIRC_FileMap``.


Idealized Inlet Physical Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a description of the model, physical inlet domain, data space, and parameter
space for the examples using the idealized inlet see `Definition and solution
of a stochastic inverse problem for the Manning’s n parameter field in
hydrodynamic models <http://dx.doi.org/10.1016/j.advwatres.2015.01.011>`_.


Adaptive Sampling Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptively samples a linear interpolant using data read from file:

    * :ref:`fromFile2DExample`
    * :ref:`fromFile3DExample`

Visualization Examples
~~~~~~~~~~~~~~~~~~~~~~

The module :mod:`~bet.postProcess.plotDomains` provides several methods used to
plot 2D domains and/or 2D slices and projections of higher dimensional domains.

    * :ref:`domains2D`
    * :ref:`domains3D`

Examples Estimating :math:`P_\Lambda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    * :ref:`q1D`
    * :ref:`q2D`
    * :ref:`q3D`

Contaminant Transport Example:
==============================
See :ref:`contaminantTransport` for an example.

Choosing Optimal QoIs Examples:
==============================
The files for these examples can be found in ``examples/sensitivity``.

List of all examples
====================
.. toctree::
   :maxdepth: 1
   :glob:

   example_rst_files/*
 
