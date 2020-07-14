
Profiler
========

Taichi's profiler can help you analyze the run-time cost of your program.

Currently there are three profiling systems in Taichi:

- ``PythonProfiler``
- ``KernelProfiler``
- ``ScopedProfiler`` (for developers)



PythonProfiler
##############

``PythonProfiler`` basically measures time spent between ``start()`` and ``stop()`` using
the Python-builtin function ``time.time()``.

There are 3 ways to use this profiler:

1. ``ti.profiler.start()`` and ``ti.profiler.stop()``, these are the most fundemental APIs:

.. code-block:: python

    from time import sleep
    import taichi as ti
    ti.init()

    def do_something_A():
        sleep(0.01)

    def do_something_B():
        sleep(0.1)

    ti.profiler.start('A')
    do_something_A()
    ti.profiler.stop('A')

    ti.profiler.start('B')
    do_something_B()
    ti.profiler.stop('B')

    ti.profiler.print()


2. ``ti.profiler()``, this one makes our code cleaner:

.. code-block:: python

    from time import sleep
    import taichi as ti
    ti.init()

    def do_something_A():
        sleep(0.01)

    def do_something_B():
        sleep(0.1)

    ti.profiler('A')  # start('A')
    do_something_A()

    ti.profiler('B')  # stop('A') & start('B')
    do_something_B()
    ti.profiler()     # stop('B')

    ti.profiler.print()


.. note::

    Running 1 and 2 should obtain something like:

    .. code-block:: none

          min   |   avg   |   max   |  num  |  total  |    name
         0.100s |  0.100s |  0.100s |    1x |  0.100s | B
        10.10ms | 10.10ms | 10.10ms |    1x | 10.10ms | A


3. ``@ti.profiler.timed``, this one is very intuitive when profiling kernels:

.. code-block:: python

    from time import sleep
    import taichi as ti
    ti.init()

    @ti.profiler.timed
    def do_something_A():
        sleep(0.01)

    @ti.profiler.timed
    def do_something_B():
        sleep(0.1)

    do_something_A()
    do_something_B()

    ti.profiler.print()


When combining ``@ti.profiler.timed`` with other decorators like ``@ti.kernel``,
then ``@ti.profiler.timed`` should be put **above** it, e.g.:

.. code-block:: python

        @ti.profiler.timed
        @ti.kernel
        def substep():
            ...


.. note::

    Running 3 should obtain something like:

    .. code-block:: none

          min   |   avg   |   max   |  num  |  total  |    name
         0.100s |  0.100s |  0.100s |    1x |  0.100s | do_something_B
        10.10ms | 10.10ms | 10.10ms |    1x | 10.10ms | do_something_A


See `misc/mpm99_timed.py <https://github.com/taichi-dev/taichi/blob/master/misc/mpm99_timed.py>`_ for their usage example.


.. warning::

    ``ti.profiler``, i.e. ``PythonProfiler``, **only works in Python-scope**, e.g.::

        @ti.func
        def substep():
            ti.profiler.start('hello')  # won't work as you expected...
            ...
            ti.profiler.stop('hello')

        @ti.profiler.timed  # won't work as you expected...
        @ti.func
        def hello():
            ...

    To do profiling **inside Taichi-scope**, please see the ``KernelProfiler`` section below.


KernelProfiler
##############

``KernelProfiler`` records the costs of Taichi kernels on devices.

To enable this profiler, please initialize Taichi using ``ti.init(kernel_profiler=True)``.

Call ``ti.kernel_profiler_print()`` to show the kernel profiling result. For example:

.. code-block:: python
    :emphasize-lines: 3, 13

    import taichi as ti

    ti.init(ti.cpu, kernel_profiler=True)
    var = ti.var(ti.f32, shape=1)


    @ti.kernel
    def compute():
        var[0] = 1.0


    compute()
    ti.kernel_profiler_print()


The outputs would be:

::

    [ 22.73%] jit_evaluator_0_kernel_0_serial             min   0.001 ms   avg   0.001 ms   max   0.001 ms   total   0.000 s [      1x]
    [  0.00%] jit_evaluator_1_kernel_1_serial             min   0.000 ms   avg   0.000 ms   max   0.000 ms   total   0.000 s [      1x]
    [ 77.27%] compute_c4_0_kernel_2_serial                min   0.004 ms   avg   0.004 ms   max   0.004 ms   total   0.000 s [      1x]


ScopedProfiler
##############

``ScopedProfiler`` measures time spent on the **host tasks** hierarchically.

This profiler is automatically on.

To show its results, call ``ti.print_profile_info()``. For example:

.. code-block:: python

    import taichi as ti

    ti.init(arch=ti.cpu)
    var = ti.var(ti.f32, shape=1)


    @ti.kernel
    def compute():
        var[0] = 1.0
        print("Setting var[0] =", var[0])


    compute()
    ti.print_profile_info()


``ti.print_profile_info()`` prints profiling results in a hierarchical format.

.. Note::

    ``ScopedProfiler`` is a C++ class in the core of Taichi. It is not exposed to Python users.

