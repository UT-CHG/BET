.. _parallel:

========
Parallel
========

Installation
------------

Running this code in parallel requires the installation of `MPI for Python
<http://mpi4py.scipy.org/>`_ which requires that your system has mpi
installed.

Running in parallel
-------------------

Depending on what modules, methods, and/or functions from BET your script uses
small or no changes will need to be made to run your script in parallel. To run
your script in parallel you will have to use::

    $ mpirun -np NUM_PROCS python YOUR_SCRIPT.py

instead of ::
    
    $ python YOUR_SCRIPT.py

You might need to make sure your globalize your arrays or make sure to choose
the ``parallel`` flag on some functions to ensure correct execution.

Affected Packages
-----------------

The modules that have parallel capabilities are as follows::

  bet/
    util
    calculateP/
      calculateP
      simpleFunP
    sampling/
      basicSampling 
      adaptiveSampling
    postProcess/
      plotP  
      postTools

util
~~~~
The module :mod:`~bet.util` provides the method
:meth:`~bet.util.get_global_values` to globalize local arrays into an array of
global values on all processors.

calculateP
~~~~~~~~~~
All methods in the module :mod:`~bet.calculateP.calculateP` benefit from
parallel execution. Only local arrays are returned for ``P``, use
:meth:`~bet.util.get_global_values` to globalize local arrays.

In the module :mod:`~bet.calculateP.simpleFunP` the methods
:meth:`~bet.calculateP.simpleFunP.unif_unif`,
:meth:`~bet.calculateP.simpleFunP.normal_normal`, and 
:meth:`~bet.calculateP.simpleFunP.unif_normal` benefit from parallel
execution.

sampling
~~~~~~~~
If you are using a model with parallel capabilities we recommend that you write
your own python interface to handle running multiple parallel copies of your
model simulatenously. If your model is serial then you might benefit from
parallel execution of scripts that use
:class:`bet.sampling.basicSampling.sampler` or
:class:`bet.sampling.adaptiveSampling.sampler`.  The method
:meth:`~bet.sampling.basicSampling.sampler.user_samples` has a parallel option
(must be specified in the method call) which will partition the samples over
several processors and return a globalized set of results.  The method
:meth:`~bet.sampling.adaptiveSampling.sampler.generalized_chains` divides up
the chains among the availiable processors and returns a globalized result.

postProcess
~~~~~~~~~~~
In :mod:`~bet.postProcess.plotP` the methods
:meth:`~bet.postProcess.plotP.calculate_1D_marginal_probs` and
:meth:`~bet.postProcess.plotP.calculate_2D_marginal_probs` benefit from
parallel execution. The methods :meth:`~bet.postProcess.plotP.plot_1D_marginal_probs` and
:meth:`~bet.postProcess.plotP.plot_2D_marginal_probs` will only execute on the
rank 0 processor.

In :mod:`~bet.postProcess.postTools` the methods
:meth:`~bet.postProcess.postTools.save_parallel_probs_csv`,
:meth:`~bet.postProcess.postTools.collect_parallel_probs_csv`,
:meth:`~bet.postProcess.postTools.save_parallel_probs_mat`, and
:meth:`~bet.postProcess.postTools.collect_parallel_probs_mat` provide tools to
save and collect probabitlies on separate processors as appropriately named files.

