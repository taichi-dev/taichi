
Profiler
========

Taichi's profiler can help you analyze the run-time cost of your program. There are two profiling systems in Taichi: ``ScopedProfiler`` and ``KernelProfiler``.

ScopedProfiler
##############

1. ``ScopedProfiler`` measures time spent on the **host tasks** hierarchically.

2. This profiler is automatically on. To show its results, call ``ti.print_profile_info()``. For example:

.. code-block:: python

    import taichi as ti

    ti.init(arch=ti.cpu)
    var = ti.field(ti.f32, shape=1)


    @ti.kernel
    def compute():
        var[0] = 1.0
        print("Setting var[0] =", var[0])


    compute()
    ti.print_profile_info()


``ti.print_profile_info()`` prints profiling results in a hierarchical format.

.. Note::

    ``ScopedProfiler`` is a C++ class in the core of Taichi. It is not exposed to Python users.

KernelProfiler
##############

1. ``KernelProfiler`` records the costs of Taichi kernels on devices. To enable this profiler, set ``kernel_profiler=True`` in ``ti.init``.

2. Call ``ti.kernel_profiler_print()`` to show the kernel profiling result. For example:

.. code-block:: python
    :emphasize-lines: 3, 13

    import taichi as ti

    ti.init(ti.cpu, kernel_profiler=True)
    var = ti.field(ti.f32, shape=1)


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
