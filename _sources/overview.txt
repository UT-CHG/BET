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
    util
    Comm
    sample
    surrogates
    calculateP/
      calculateP 
      calculateError
      simpleFunP
      voronoiHistogram
      indicatorFunctions
    sampling/
      basicSampling 
      adaptiveSampling
      LpGeneralizedSamples
    postProcess/
      plotP
      plotDomains
      postTools
    sensitivity/
      gradients
      chooseQoIs

Code Overview
--------------
:mod:`bet.sample` module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.sample

:mod:`bet.util` module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.util

:mod:`bet.Comm` module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.Comm

:mod:`bet.calculateP` Sub-package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.calculateP

:mod:`bet.sampling` Sub-package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.sampling

:mod:`bet.postProcess` Sub-package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.postProcess

:mod:`bet.sensitivity` Sub-package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.sensitivity

.. seealso:: :ref:`modindex` for detailed documentation of modules, classes, etc.

Internal dependencies
---------------------
Dependencies via :keyword:`import` statements::

        bet 
          \-Comm (bet.sample,bet.surrogates,bet.sampling.adaptiveSampling,bet.sensitivity.chooseQoIs,bet.sampling.basicSampling,bet.util,bet.calculateP.calculateP,bet.postProcess.plotP,bet.calculateP.calculateError,bet.calculateP.simpleFunP)
          \-calculateP 
          | \-calculateError (bet.surrogates)
          | \-calculateP (bet.surrogates,bet.calculateP.calculateError)
          \-sample (bet.surrogates,bet.sampling.adaptiveSampling,bet.postProcess.plotDomains,bet.sampling.basicSampling,bet.sensitivity.gradients,,bet.postProcess.plotP,bet.postProcess.postTools,bet.calculateP.calculateError,bet.calculateP.simpleFunP)
          \-sampling 
          | \-LpGeneralizedSamples (bet.sample,bet.sensitivity.gradients)
          | \-basicSampling (bet.sampling.adaptiveSampling,bet.calculateP.calculateP)
          \-util (bet.sample,bet.sensitivity.gradients,bet.sampling.adaptiveSampling,bet.sensitivity.chooseQoIs,bet.postProcess.plotDomains,,bet.calculateP.calculateP,bet.calculateP.calculateError,bet.calculateP.simpleFunP)


External dependencies
---------------------
This pacakge requires `matplotlib <http://http://matplotlib.org>`_, `scipy
<scipy.org>`_, mpl_toolkits,  `numpy <http://http://www.numpy.org>`_, and
`pyDOE <http://pythonhosted.org/pyDOE/>`_. This package is written in `Python
<http://http://docs.python.org/2>`_.

::    
  
        matplotlib 
          \-cm (bet.postProcess.plotP)
          \-lines (bet.postProcess.plotDomains)
          \-pyplot (bet.postProcess.plotP,bet.postProcess.plotDomains)
          \-ticker (bet.postProcess.plotP)
          \-tri (bet.postProcess.plotDomains)
        mpl_toolkits 
          \-mplot3d (bet.postProcess.plotP,bet.postProcess.plotDomains)
        numpy (bet.sample,bet.surrogates,bet.sampling.adaptiveSampling,bet.sensitivity.chooseQoIs,bet.postProcess.plotDomains,bet.sampling.LpGeneralizedSamples,bet.sampling.basicSampling,bet.sensitivity.gradients,bet.calculateP.indicatorFunctions,bet.util,,bet.calculateP.calculateP,bet.postProcess.plotP,bet.postProcess.postTools,bet.calculateP.calculateError,bet.calculateP.simpleFunP)
          \-linalg (bet.sample,bet.calculateP.calculateError)
        pyDOE (bet.sampling.basicSampling)
        scipy 
          \-fftpack (bet.postProcess.plotP)
          \-io (bet.sample,bet.sampling.basicSampling,bet.sampling.adaptiveSampling)
          \-spatial (bet.sample,bet.sensitivity.gradients,bet.calculateP.calculateError)
          \-stats (bet.sample,bet.sensitivity.chooseQoIs,bet.calculateP.simpleFunP)




