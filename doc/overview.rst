.. _overview:

========
Overview
========

BET is an initialism of Butler, Estep and Tavener, the primary authors of a
`series <https://epubs.siam.org/doi/abs/10.1137/100785946>`_
`of <https://epubs.siam.org/doi/abs/10.1137/100785958>`_
`papers <https://epubs.siam.org/doi/abs/10.1137/130930406>`_
that introduced the mathematical framework for measure-based data-consistent stochastic inversion, for which BET included
a computational implementation. However, since it's initial inception it has grown to include a broad range of
`data- <https://iopscience.iop.org/article/10.1088/1361-6420/ab8f83/meta>`_
`consistent <https://epubs.siam.org/doi/abs/10.1137/16M1087229>`_
`methods <https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.6078>`_
that can also be density-based.
It has been applied to a wide variety of application problems, many of which can be found
`here.
<https://scholar.google.com/scholar?oi=bibs&hl=en&cites=915741139550333528,6038673497778212734,182199236207122617>`_


Mathematical Theory
------------
For more information about the methods and algorithms for the Measure-Based Data-Consistent framework, see `A Measure-Theoretic
Computational Method for Inverse Sensitivity Problems III: Multiple Quantities of Interest
<http://dx.doi.org/10.1137/130930406>`_ for the formulation of the stochastic
inverse problem  along with proofs of existence
and uniqueness of solutions, `Solving Stochastic Inverse Problems using Sigma-Algebras on Contour Maps
<http://arxiv.org/abs/1407.3851>`_ for the convergence
and error analysis of the non-intrusive algorithm, and
`Definition and solution of a stochastic inverse problem for the Manning’s n parameter field in
hydrodynamic models <http://dx.doi.org/10.1016/j.advwatres.2015.01.011>`_ for a less technical description
of the method for engineers as well as application to a physically relevant problem
in coastal ocean modeling.

For more information about the methods and algorithms for Density-Based Data-Consistent framework see
`Combining Push-Forward Measures and Bayes' Rule to Construct Consistent Solutions to Stochastic Inverse Problems
<https://doi.org/10.1137/16M1087229>`_ and
`Data-Consistent Inversion for Stochastic Input-to-Output Maps
<https://iopscience.iop.org/article/10.1088/1361-6420/ab8f83/meta>`_.


Installation
------------

The code currently resides at `GitHub
<https://github.com/UT-CHG/BET>`_.
The current development branch of BET can be installed from GitHub,  using ``pip``::

    $ pip install git+https://github.com/UT-CHG/BET

Another option is to clone the repository and install BET using::

    $ python setup.py install

Dependencies
------------
BET is tested on Python 3.6 and 3.7 (but should work on most recent Python 3 versions) and depends on
`NumPy <http://www.numpy.org/>`_, `SciPy <http://www.scipy.org/>`_,
`matplotlib <http://matplotlib.org/>`_, `pyDOE <https://pythonhosted.org/pyDOE/>`_,
`pytest <https://docs.pytest.org/>`_, and
`mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ (optional) (see ``requirements.txt`` for version information).
For some optional features `LUQ <https://github.com/CU-Denver-UQ/LUQ>`_ is also required.

License
------------
`GNU Lesser General Public License (LGPL) <https://github.com/UT-CHG/BET/blob/master/LICENSE.txt>`_

Citing BET
------------
Please include the citation:

Lindley Graham, Steven Mattis, Scott Walsh, Troy Butler, Michael Pilosov, and Damon McDougall.
“BET: Butler, Estep, Tavener Method V2.0.0”. Zenodo, August 10, 2016.
`doi:10.5281/zenodo.59964 <https://doi.org/10.5281/zenodo.59964>`_

Lindley Graham, Steven Mattis, Michael Pilosov, Scott Walsh, Troy Butler, Michael Pilosov, … Damon McDougall. (2020, July 9). UT-CHG/BET: BET v3.0.0 (Version v3.0.0). Zenodo. http://doi.org/10.5281/zenodo.3936258

or in BibTEX::

    @software{lindley_graham_2020_3936258,
    author       = {Lindley Graham and
                    Steven Mattis and
                    Michael Pilosov and
                    Scott Walsh and
                    Troy Butler and
                    Wenjuan Zhang and
                    Damon McDougall},
  title          =  UT-CHG/BET: BET v3.0.0},
  month          =  jul,
  year           =  2020,
  publisher      =  {Zenodo},
  version        =  {v3.0.0},
  doi            =  {10.5281/zenodo.3936258},
  url            =  {https://doi.org/10.5281/zenodo.3936258}
  }


Documentation
------------

This code has been documented with sphinx. the documentation is available online at http://ut-chg.github.io/BET.
To build documentation run
``make html`` in the ``doc/`` folder.

To build/update the documentation use the following commands::

    sphinx-apidoc -f -o doc bet
    cd doc/
    make html
    make html

This creates the relevant documentation at ``bet/gh-pages/html``.
To change the build location of the documentation you will need to update ``doc/makefile``.

You will need to run ``sphinx-apidoc`` and reinstall bet anytime a new module or method in the source code has been added.
If only the ``*.rst`` files have changed then you can simply run ``make html`` twice in the doc folder.

Testing
------------

To run the tests in the root directory with ``pytest`` in serial call::

    $ pytest ./test/

Some features of BET (primarily those associated with the measure-based approach) have the ability to work in parallel. To run tests in parallel call::

    $ mpirun -np NPROC pytest ./test/

Make sure to have a working MPI environment (we recommend `mpich <http://www.mpich.org/downloads/>`_).
if you want to use parallel features.

Contributors
------------

See the `GitHub contributors page <https://github.com/UT-CHG/BET/graphs/contributors>`_.

Contact
------------

BET is in active development. Hence, some features are still being added and you may find bugs we have overlooked.
If you find something please report these problems to us through GitHub so that we can fix them. Thanks!

Please note that we are using continuous integration and issues for bug tracking.

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
      calculateR
    sampling/
      basicSampling
      useLUQ
      LpGeneralizedSamples
    postProcess/
      plotP
      plotDomains
      postTools
      plotVoronoi
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
