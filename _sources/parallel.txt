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


Example scripts are availiable in ``examples/parallel_and_serial_sampling``
which demonstrate different modes of parallel and serial sampling. Also the
:ref:`fenicsMultipleSerialExample` is an example that can be run with serial
BET and uses `Launcher <https://github.com/TACC/launcher>`_ to launch multiple
serial runs of the model in parallel.

Parallel Enabled Modules
------------------------

The modules that have parallel capabilities are as follows::

  bet/
    util
    sample
    Comm
    surrogates
    calculateP/
      calculateP
      simpleFunP
      calculateError
    sampling/
      basicSampling 
      adaptiveSampling
    postProcess/
      plotP  
      postTools
    sensitivity/
      chooseQoIs

util
~~~~
The module :mod:`~bet.util` provides the method
:meth:`~bet.util.get_global_values` to globalize local arrays into an array of
global values on all processors.

sample
~~~~~~
The :class:`~bet.sample.sample_set_base` has methods which benifit from
parallel execution and has methods for localizing
(:meth:`~bet.sample.sample_set_base.local_to_local`) and localizing
(:meth:`~bet.sample.sample_set_base.local_to_global`) member attributes.
Localized attributes are denoted as ``*_local``. The
:class:`~bet.sample.discretization` aslo has methods which enable and benifit
from parallel execution.

calculateP
~~~~~~~~~~
All methods in the module :mod:`~bet.calculateP.calculateP` benefit from
parallel execution.

In the module :mod:`~bet.calculateP.simpleFunP` any method with a
``num_d_emulate`` option benefits greatly from parallel execution.

sampling
~~~~~~~~
If you are using a model with parallel capabilities we recommend that you write
your own python interface to handle running multiple parallel copies of your
model simulatenously. If your model is serial then you might benefit from
parallel execution of scripts that use
:class:`bet.sampling.basicSampling.sampler` or
:class:`bet.sampling.adaptiveSampling.sampler`.  The method
:meth:`~bet.sampling.basicSampling.sampler.compute_QoI_and_create_discretization`
and :meth:`~bet.sampling.basicSampling.sampler.create_random_discretization`
both  will partition the samples over several processors and have a globalize
option to return a globalized set of results. The method
:meth:`~bet.sampling.adaptiveSampling.sampler.generalized_chains` divides up
the chains among the availiable processors and returns a globalized result.
This method also has serial and parallel hotstart capabilties.

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

sensitivity
~~~~~~~~~~~
All methods in the module :mod:`~bet.sensitivity.chooseQoIs` benefit from parallel execution.

