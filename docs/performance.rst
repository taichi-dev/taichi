
Performance
===========

For-loop decorators
-------------------

In Taichi, as we all know, the outermost for-loop is parallelized.

However, there're some implementation details about **how it is parallelized**.

Taichi provides some API to modify these parameters which is usually invisible
from end-users.
This allows advanced users to manually fine tune the performance to be better.

For example, specifying a suitable ``ti.block_dim`` could yield almost 3x performance boost in `examples/mpm3d.py <https://github.com/taichi-dev/taichi/blob/master/examples/mpm3d.py>`_!

.. note::

   For performance profiling utilities, see :ref:`profiler`.

Hierarchical structure of GPU
*****************************

Taichi programs could be executed in parallel on GPU.
The level of structure of GPU is defined hierarchically.

From small to large, the computation units are:
**invocation** < **thread** < **block** < **grid**.

- **invocation**:
  Invocation is the **body of a for-loop**.
  Each invocation corresponding to a specific ``i`` value in for-loop.

- **thread**:
  Invocations are grouped into threads.
  Threads are the minimal unit that is parallelized.
  All invocations within a thread are executed in **serial**.
  We usually use 1 invocation per thread for maximizing parallel performance.

- **block**:
  Threads are grouped into blocks.
  All threads within a block are executed in **parallel**.
  Threads within the same block can share their **block local storage**.

- **grid**:
  Blocks are grouped into grids.
  Grid is the minimal unit that being **launched** from host.
  All blocks within a grid are executed in **parallel**.
  In Taichi, each **parallelized for-loop** is a grid.

For more details, please see `this tutorial by CUDA official <http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf>`_.

API reference
*************

We may **prepend** some decorator(s) to tweak the property of a for-loop, e.g.:

.. code-block:: python

    @ti.kernel
    def func():
        for i in range(8192):  # no decorator, use default settings
            ...

        ti.block_dim(128)      # change the property of next for-loop:
        for i in range(8192):  # will be parallelized with block_dim=128
            ...

        for i in range(8192):  # no decorator, use default settings
            ...


Here's the list of available decorators:


.. function:: ti.block_dim(n)

    :parameter n: (int) threads per block / block dimension

    Specify the **threads per block** of next parallelized for-loop.

    TODO: explain how-and-when to tune ``block_dim``.

    We often set ``block_dim`` to a smaller value when **intensive
    gather / scatter** is involved. E.g. the *MLS-MPM method*, especially
    when it comes to 3D.

    .. note::

        The argument ``n`` must be power-of-two for now.


.. function:: ti.thread_dim(n)

    :parameter n: (int) invocations per thread / thread dimension

    Specify the **invocations per thread** of next parallelized for-loop.

    Having a large ``thread_dim`` could be helpful when your program have
    reduction over a single variable.
    Usually the Taichi compiler will choose a optimal ``thread_dim`` when
    such reduction is detected.
    But you may fine tune it manually to get the best performance.

    .. note::

        This function is only available on OpenGL for now.
