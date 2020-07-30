
Profiler
========

Taichi's profilers can help analyze the performance of your program.

Currently there are three profiling systems in Taichi:

- ``PythonProfiler``
- ``KernelProfiler``
- ``ScopedProfiler`` (for developers)



PythonProfiler
--------------

``PythonProfiler`` basically measures time spent between ``start()`` and ``stop()`` using
the Python-builtin function ``time.time()``.

Profiling APIs
**************

There are 3 ways to use this profiler:

1. ``ti.profiler.start()`` and ``ti.profiler.stop()`` are the most fundemental APIs:
   It will measure the time difference between ``start`` and ``stop``.
   Then save the result according to the given name. e.g.:

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


.. code-block:: none

      min   |   avg   |   max   |  num  |  total  |    name
     0.100s |  0.100s |  0.100s |    1x |  0.100s | B
    10.10ms | 10.10ms | 10.10ms |    1x | 10.10ms | A


2. ``with ti.profiler()``, this one makes our code cleaner and readable.
   Basically, it will automatically invoke ``stop`` when the indented block exited.
   For more details about the ``with`` syntax in Python,
   see `this tutorial <https://www.pythonforbeginners.com/files/with-statement-in-python>`_.

.. code-block:: python

    from time import sleep
    import taichi as ti
    ti.init()

    def do_something_A():
        sleep(0.01)

    def do_something_B():
        sleep(0.1)

    with ti.profiler('A'):
        do_something_A()
    # automatically invoke stop('A')

    with ti.profiler('B'):
        do_something_B()
    # automatically invoke stop('B')

    ti.profiler.print()


.. code-block:: none

      min   |   avg   |   max   |  num  |  total  |    name
     0.100s |  0.100s |  0.100s |    1x |  0.100s | B
    10.10ms | 10.10ms | 10.10ms |    1x | 10.10ms | A


3. ``@ti.profiler.timed``, this one is very intuitive when profiling kernels.
   It will measure the time spent in the function, i.e. ``start`` when entering the function,
   ``stop`` when leaving the function, and the record name is the function name.

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


.. code-block:: none

      min   |   avg   |   max   |  num  |  total  |    name
     0.100s |  0.100s |  0.100s |    1x |  0.100s | do_something_B
    10.10ms | 10.10ms | 10.10ms |    1x | 10.10ms | do_something_A


.. warning::

    When combining ``@ti.profiler.timed`` with other decorators like ``@ti.kernel``,
    then ``@ti.profiler.timed`` should be put **above** it, e.g.:

    .. code-block:: python

            @ti.profiler.timed
            @ti.kernel
            def substep():
                ...


Recording multiple entries
**************************

When a same **name** is used for multiple times, then they will be merged into one, e.g.:

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

    ti.profiler.start('A')
    do_something_B()
    ti.profiler.stop('A')

    ti.profiler.start('B')
    do_something_B()
    ti.profiler.stop('B')

    ti.profiler.print()

will obtain:

.. code-block:: none

      min   |   avg   |   max   |  num  |  total  |    name
    10.10ms | 55.12ms |  0.100s |    2x |  0.110s | A
     0.100s |  0.100s |  0.100s |    1x |  0.100s | B


- ``min`` is the minimum time in records.
- ``avg`` is the average time of records.
- ``max`` is the maximum time in records.
- ``num`` is the number of record entries.
- ``total`` is the total costed time of records.


Profiler options
****************

Due to Taichi's JIT mechanism, a kernel will be **compiled** on its first invocation.
So the first record will be extremely long compared to the following records since it
**involves both compile time and execution time**, e.g.:

.. code-block:: none

       min   |   avg   |   max   |  num  |  total  |    name
      2.37ms |  3.79ms |  1.615s | 1900x |  7.204s | substep

.. code-block:: none

       min   |   avg   |   max   |  num  |  total  |    name
      2.37ms |  2.95ms | 12.70ms | 1895x |  5.592s | substep


As you see, this make our result inaccurate, especially the ``max`` column.

To avoid this, you may specify a ``warmup`` option to ``ti.profiler``, e.g.:

.. code-block:: python

    @ti.profiler.timed(warmup=5)
    @ti.kernel
    def substep():
        ...


Set ``warmup=5`` for example, will **discard** the first 5 record entries.
I.e. discard the kernel compile time and possible TLB and cache misses on start up.


Check out `misc/mpm99_timed.py <https://github.com/taichi-dev/taichi/blob/master/misc/mpm99_timed.py>`_ for a summary example.


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
--------------

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
--------------

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
