.. _overview:

========
Overview
========

Installation
------------

The code currently resides at `GitHub
<https://github.com/smattis/BET>`_.
If you have a 
`zip file <https://github.com/smattis/BET/archive/master.zip>`_ you can install
BET using::

    python setup.py install

from the package root directory. The BET package is currently NOT avaiable in
the `Python Package Index <http://pypi.python.org/pypi/Sphinx>`_ this may
change in the future. This pacakge requires `matplotlib <http://http://matplotlib.org>`_, `scipy <scipy.org>`_, mpl_toolkits,  `numpy
<http://http://www.numpy.org>`_, and `pyDOE <http://pythonhosted.org/pyDOE/>`_. This package is written in `Python
<http://http://docs.python.org/2>`_.

Package Layout
--------------

The package layout is as follows::

  bet/
    calculateP/
      calculateP.py  
      simpleFunP.py  
      voronoiHistogram.py
    sampling/
      basicSampling.py  
      adaptiveSampling.py
    vis/
      plotP.py  
      plotDomains.py 
    loadBalance/        # !!! IN ACTIVE DEVELOPMENT !!!
      lb_PADCIRC.py  
      load_balance.py  
      lb_DIAMOND.py

Code Overview
--------------

:mod:`calculateP` Package
~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.calculateP

:mod:`sampling` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.sampling

:mod:`vis` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.vis

:mod:`loadBalance` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: bet.loadBalance

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
<http://http://www.numpy.org>`_, and `pyDOE <http://pythonhosted.org/pyDOE/>`_. This package is written in `Python
<http://http://docs.python.org/2>`_.

::    
  
    matplotlib 
      \-cm (bet.vis.plotP)
      \-pyplot (bet.vis.plotDomains,bet.vis.plotP)
      \-ticker (bet.vis.plotP)
      \-tri (bet.vis.plotDomains)
    mpi4py 
      \-MPI (bet.calculateP.calculateP,bet.vis.plotP,bet.calculateP.simpleFunP)
    mpl_toolkits 
      \-mplot3d (bet.vis.plotP)
    numpy (bet.sampling.adaptiveSampling,bet.sampling.basicSampling,bet.vis.plotP,bet.calculateP.voronoiHistogram,bet.calculateP.calculateP,bet.vis.plotDomains,bet.calculateP.simpleFunP)
    pyDOE (bet.sampling.basicSampling)
    scipy 
      \-io (bet.sampling.basicSampling,bet.sampling.adaptiveSampling)
      \-spatial (bet.calculateP.voronoiHistogram,bet.calculateP.calculateP,bet.calculateP.simpleFunP)
      \-stats (bet.calculateP.simpleFunP)



