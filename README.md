BET
===

BET is in active development. Hence, some features are still being added and you may find bugs we have overlooked. If you find something please report these problems to us through github so that we can fix them. Thanks!

Please note that we are using continuous integration and issues for bug tracking.

Butler, Estep, Tavener Method

This code has been documented with Sphinx. The documentation is available online ta http://ut-chg.github.io/BET. To build documentation run 
``make html`` in the ``doc/`` folder.
To build/update the documentation use the following commands::

    sphinx-apidoc -f -o doc bet
    cd doc/
    make html
    make html

This creates the relevant documentation at ``BET/gh-pages/html``. To change the build location of the documentation you will need to update ``doc/Makefile``.

You will need to run sphinx-apidoc AND reinstall BET anytime a new module or method in the source code has been added. If only the `*.rst` files have changed then you can simply run ``make html`` twice in the doc folder.

Useful scripts are contained in ``examples/``

Tests
-----

To run tests in serial call::

    nosetests tests

To run tests in parallel call::

    mpirun -np NPROC nosetets tests

Dependencies
------------

`BET` requires the following packages:

1. [numpy](http://www.numpy.org/)
2. [scipy](http://www.scipy.org/)
3. [nose](https://nose.readthedocs.org/en/latest/)
4. [pyDOE](https://pythonhosted.org/pyDOE/)
5. [matplotlib](http://matplotlib.org/)
