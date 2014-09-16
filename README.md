BET
===

BET is in active development. Hence, some features are still being added and you may find bugs we have overlooked. If you find something please report these problems to us through github so that we can fix them. Thanks!

Butler, Estep, Tavener Method

This code has been documented with Sphinx. To build documentation run 
``make html`` in the ``doc/`` folder.
To build/update the documentation use the following commands::

    sphinx-apidoc -f -o doc bet
    cd doc/
    make html
    make html

This creates the relevant documentation at ``BET/gh-pages/html``. To change the build location of the documentation you will need to update ``doc/Makefile``.

You will need to run sphinx-apidoc anytime a new module or method in the source code has been added. If only the *.rst files have changed then you can simply run ``make html`` twice in the doc folder.

Useful scripts are contained in ``examples/``

