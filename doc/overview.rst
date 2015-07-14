.. _overview:

========
Overview
========

Installation
------------

The code currently resides at `GitHub
<https://github.com/UT-CHG/BET>`_.
If you have a 
`zip file <https://github.com/UT-CHG/BET/archive/master.zip>`_ you can install
BET using::

    python setup.py install

from the package root directory. The BET package is currently NOT avaiable in
the `Python Package Index <http://pypi.python.org/pypi/Sphinx>`_ this may
change in the future. This pacakge requires `matplotlib <http://http://matplotlib.org>`_, `scipy <scipy.org>`_, mpl_toolkits,  `numpy
<http://http://www.numpy.org>`_, and `pyDOE <http://pythonhosted.org/pyDOE/>`_. This package is written in `Python
<http://http://docs.python.org/2>`_.

If you have `nose <http://nose.readthedocs.org/en/latest/index.html>`_
installed you can run tests by typing::

    nosetests

in ``BET`` to run the serial tests or ::

    mpirun -np NPROC nosetests

to run the parallel tests.

Package Layout
--------------

The package layout is as follows::

  bet/
    calculateP/
      calculateP 
      simpleFunP
      voronoiHistogram
    sampling/
      basicSampling 
      adaptiveSampling
    postProcess/
      plotP
      plotDomains
      postTools
    sensitivity/
      gradients
      chooseQoIs

Code Overview
--------------

:mod:`calculateP` Package
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.calculateP

:mod:`sampling` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.sampling

:mod:`postProcess` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.postProcess

:mod:`sensitivity` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.sensitivity

.. seealso:: :ref:`modindex` for detailed documentation of modules, classes, etc.

Internal dependencies
---------------------
Dependencies via :keyword:`import` statements::

    bet 
      \-calculateP 
      | \-voronoiHistogram (bet.calculateP.simpleFunP)
      \-sampling 
        \-basicSampling (bet.sampling.adaptiveSampling)


External dependencies
---------------------
This pacakge requires `matplotlib <http://http://matplotlib.org>`_, `scipy <scipy.org>`_, mpl_toolkits,  `numpy
<http://http://www.numpy.org>`_, sys, itertools, and `pyDOE <http://pythonhosted.org/pyDOE/>`_. This package is written in `Python
<http://http://docs.python.org/2>`_.

::    
  
    matplotlib 
      \-cm (bet.postProcess.plotP)
      \-pyplot (bet.postProcess.plotDomains,bet.postProcess.plotP)
      \-ticker (bet.postProcess.plotP)
      \-tri (bet.postProcess.plotDomains)
    mpi4py 
      \-MPI (bet.calculateP.calculateP,bet.postProcess.plotP,bet.calculateP.simpleFunP)
    mpl_toolkits 
      \-mplot3d (bet.postProcess.plotP)
    numpy (bet.sampling.adaptiveSampling,bet.sampling.basicSampling,bet.postProcess.plotP,bet.calculateP.voronoiHistogram,bet.calculateP.calculateP,bet.postProcess.plotDomains,bet.calculateP.simpleFunP,bet.sensitivity.gradients,bet.sensitivity.chooseQoIs)
    pyDOE (bet.sampling.basicSampling)
    scipy 
      \-io (bet.sampling.basicSampling,bet.sampling.adaptiveSampling)
      \-spatial (bet.calculateP.voronoiHistogram,bet.calculateP.calculateP,bet.calculateP.simpleFunP,bet.sensitivity.gradients)
      \-stats (bet.calculateP.simpleFunP)
    itertools
      (bet.sensitivity.chooseQoIs)
    sys
      (bet.sensitivity.gradients)



