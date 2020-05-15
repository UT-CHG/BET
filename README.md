# BET
[![Build Status](https://travis-ci.org/UT-CHG/BET.svg?branch=master)](https://travis-ci.org/UT-CHG/BET) [![DOI](https://zenodo.org/badge/18813599.svg)](https://zenodo.org/badge/latestdoi/18813599) [![codecov](https://codecov.io/gh/UT-CHG/BET/branch/master/graph/badge.svg)](https://codecov.io/gh/UT-CHG/BET) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UT-CHG/BET/master)

BET is a Python package for measure-theoretic and data-consistent stochastic forward and inverse problems. The package is very flexible and is applicable to a wide variety of problems.

## Installation
The current development branch of BET can be installed from GitHub,  using ``pip``:

    pip install git+https://github.com/UT-CHG/BET
    
Another option is to clone the repository and install BET using
``python setup.py install``


## Dependencies
BET is tested on Python 3.6 and 3.7 (but should work on most recent Python 3 versions) and depends on [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [matplotlib](http://matplotlib.org/), [pyDOE](https://pythonhosted.org/pyDOE/), [pytest](https://docs.pytest.org/), and [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (optional) (see ``requirements.txt`` for version information). For some optional features [LUQ](https://github.com/CU-Denver-UQ/LUQ) is also required.

## License
[GNU Lesser General Public License (LGPL)](https://github.com/UT-CHG/BET/blob/master/LICENSE.txt)

## Documentation

This code has been documented with sphinx. the documentation is available online at http://ut-chg.github.io/BET. to build documentation run 
``make html`` in the ``doc/`` folder.

To build/update the documentation use the following commands::

    sphinx-apidoc -f -o doc bet
    cd doc/
    make html
    make html

This creates the relevant documentation at ``bet/gh-pages/html``. 
To change the build location of the documentation you will need to update ``doc/makefile``.

You will need to run sphinx-apidoc and reinstall bet anytime a new module or method in the source code has been added. 
If only the `*.rst` files have changed then you can simply run ``make html`` twice in the doc folder.

## Examples
Useful scripts are contained in ``examples/``, as are the following sets of example Jupyter Notebooks:

- [Plotting](./examples/plotting/Plotting_Examples.ipynb)
    (this allows execution any of the following examples and plots the associated results)
- [Contaminant Transport](./examples/contaminantTransport/contaminant.ipynb)
- [Validation Example](./examples/validationExample/linearMap.ipynb)
- [Linear (QoI) Sensitivity](./examples/sensitivity/linear_sensitivity.ipynb)
- [Linear Map](./examples/linearMap/linearMapUniformSampling.ipynb)

Furthermore, the `examples/templates` directory contains a [notebook](./examples/templates/Example_Notebook_Template.ipynb) that serves as a template for the examples.
You can also try out BET in your browser using [Binder](https://mybinder.org/v2/gh/UT-CHG/BET/master).

## Testing

To run the tests in the root directory with `pytest` in serial call::

    pytest

Some features of BET have the ability to work in parallel. To run tests in parallel call::

    mpirun -np nproc pytest

Make sure to have a working MPI environment (we recommend [mpich](http://www.mpich.org/downloads/)) if you want to use parallel features.


(Note: you may need to set `~/.config/matplotlib/matplotlibrc` to include `backend:agg` if there is no `DISPLAY` port in your environment). 

## Contact
BET is in active development. Hence, some features are still being added and you may find bugs we have overlooked. If you find something please report these problems to us through GitHub so that we can fix them. Thanks! 

Please note that we are using continuous integration and issues for bug tracking.
