=======================================
Linear Map Example
=======================================

See :download:`linear map example
<../../examples/linearMap/linearMapUniformSampling.py>` for an example using a linear
map.

==============================================
ADCIRC Based Examples
==============================================

The files for these examples can be found in ``examples/fromFileMap`` and
``examples/fromADCIRC``.


Idealized Inlet Physical Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo:: 

    Description of the model, physical inlet domain, data space, and parameter
    space for the examples using the idealized inlet.


Adaptive Sampling Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptively samples a linear interpolant using data read from file:

    * :ref:`fromFile2D`
    * :ref:`fromFile3D`

Adaptively samples PADCIRC using :mod:`polyadcirc.run_framework` modules.
Example submission files are included as ``filename.sbatch``:

    * :ref:`adaptive2D`
    * :ref:`adaptive3D`

Visualization Examples
~~~~~~~~~~~~~~~~~~~~~~

.. todo::

    Write visualization examples for :mod:`~bet.postProcess.plotP` and
    :mod:`~bet.postProcess.plotDomains`.

Examples Estimating :math:`P_\Lambda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    * :ref:`q1D`
    * :ref:`q2D`
    * :ref:`q3D`
