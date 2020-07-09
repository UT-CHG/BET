# BET
[![Build Status](https://travis-ci.org/UT-CHG/BET.svg?branch=master)](https://travis-ci.org/UT-CHG/BET) [![DOI](https://zenodo.org/badge/18813599.svg)](https://zenodo.org/badge/latestdoi/18813599) [![codecov](https://codecov.io/gh/UT-CHG/BET/branch/master/graph/badge.svg)](https://codecov.io/gh/UT-CHG/BET) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UT-CHG/BET/master)

BET is a Python package for data-consistent stochastic forward and inverse problems. The package is very flexible and is applicable to a wide variety of problems. 

BET is an initialism of Butler, Estep and Tavener, the primary authors of a [series](https://epubs.siam.org/doi/abs/10.1137/100785946) [of](https://epubs.siam.org/doi/abs/10.1137/100785958) [papers](https://epubs.siam.org/doi/abs/10.1137/130930406) that introduced the mathematical framework for measure-based data-consistent stochastic inversion, for which BET included a computational implementation. However, since its initial inception it has grown to include a broad range of [data-](https://iopscience.iop.org/article/10.1088/1361-6420/ab8f83/meta)[consistent](https://epubs.siam.org/doi/abs/10.1137/16M1087229) [methods](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.6078). It has been applied to a wide variety of application problems, many of which can be found [here](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=915741139550333528,6038673497778212734,182199236207122617).

## Installation
The current development branch of BET can be installed from GitHub,  using ``pip``:

    pip install git+https://github.com/UT-CHG/BET
    
Another option is to clone the repository and install BET using
``python setup.py install``


## Dependencies
BET is tested on Python 3.6 and 3.7 (but should work on most recent Python 3 versions) and depends on [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [matplotlib](http://matplotlib.org/), [pyDOE](https://pythonhosted.org/pyDOE/), [pytest](https://docs.pytest.org/), and [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (optional) (see [requirements.txt](requirements.txt) for version information). For some optional features [LUQ](https://github.com/CU-Denver-UQ/LUQ) is also required. mpi4py is required to take advantage of parallel features and requires an mpi implementation. It can be installed by:

    pip install mpi4py


## License
[GNU Lesser General Public License (LGPL)](LICENSE.txt)

## Citing BET
Please include the citation:

Lindley Graham, Steven Mattis, Michael Pilosov, Scott Walsh, Troy Butler, Michael Pilosov, … Damon McDougall. (2020, July 9). UT-CHG/BET: BET v3.0.0 (Version v3.0.0). Zenodo. http://doi.org/10.5281/zenodo.3936258

or in BibTEX:

    @software{BET,
              author = {Lindley Graham and
                        Steven Mattis and
                        Michael Pilosov and
                        Scott Walsh and
                        Troy Butler and
                        Wenjuan Zhang and
                        Damon McDougall},
              title = {UT-CHG/BET: BET v3.0.0},
              month = jul,
              year = 2020,
              publisher = {Zenodo},
              version = {v3.0.0},
              doi = {10.5281/zenodo.3936258},
              url = {https://doi.org/10.5281/zenodo.3936258}
              }

## Documentation

This code has been documented with sphinx. the documentation is available online at http://ut-chg.github.io/BET. To build documentation run 
``make html`` in the ``doc/`` folder.

To build/update the documentation use the following commands:

    sphinx-apidoc -f -o doc bet
    cd doc/
    make html
    make html

This creates the relevant documentation at ``bet/gh-pages/html``. 
To change the build location of the documentation you will need to update ``doc/makefile``.

You will need to run sphinx-apidoc and reinstall bet anytime a new module or method in the source code has been added. 
If only the `*.rst` files have changed then you can simply run ``make html`` twice in the doc folder.
Building the docs requires Sphinx and the Read the Docs Sphinx theme, which can be installed with `pip` by:

    pip install Sphinx sphinx_rtd_theme

## Examples
Examples scripts are contained in [here](examples/). 

You can also try out BET in your browser using [Binder](https://mybinder.org/v2/gh/UT-CHG/BET/master).

## Testing

To run the tests in the root directory with `pytest` in serial call:

    pytest ./test/

Some features of BET (primarily those associated with the measure-based approach) have the ability to work in parallel. To run tests in parallel call:

    mpirun -np NPROC pytest ./test/

Make sure to have a working MPI environment (we recommend [mpich](http://www.mpich.org/downloads/)) if you want to use parallel features.


(Note: you may need to set `~/.config/matplotlib/matplotlibrc` to include `backend:agg` if there is no `DISPLAY` port in your environment). 

## Contributors
See the [GitHub contributors page](https://github.com/UT-CHG/BET/graphs/contributors).

## Contact
BET is in active development. Hence, some features are still being added and you may find bugs we have overlooked. If you find something please report these problems to us through GitHub so that we can fix them. Thanks! 

Please note that we are using continuous integration and issues for bug tracking.
