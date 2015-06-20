=======================================
Linear Map Example
=======================================

See :ref:`linearMap` for an example using a linear map.

=======================================
Non-Linear Map Example
=======================================

See :download:`non linear map example
<../../examples/nonlinearMap/nonlinearMapUniformSampling.py>` for an example using a non-linear
map.

==============================================
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

    * :ref:`fromFile2D`
    * :ref:`fromFile3D`

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
