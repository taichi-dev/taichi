.. _layout:

Tensor layout
=============

Tensors (:ref:`scalar_tensor`) can be *placed* in a specific shape and *layout*.
Defining a proper layout can be critical to performance, although in most cases, you probably don't have to worry about it.
In Taichi, layout is defined in a recursive manner. See :ref:`snode` for more details about how this works.

For example, this declares a 0-D tensor:

.. code-block:: python

    x = ti.var(ti.f32)
    ti.root.place(x)
    # or
    x = ti.var(ti.f32, shape=())

This declares a 1D tensor of size ``3``:

.. code-block:: python

    x = ti.var(ti.f32)
    ti.root.dense(ti.i, 3).place(x)
    # or
    x = ti.var(ti.f32, shape=3)

This declares a 2D tensor of shape ``(3, 4)``:

.. code-block:: python

    x = ti.var(ti.f32)
    ti.root.dense(ti.ij, (3, 4)).place(x)
    # or
    x = ti.var(ti.f32, shape=(3, 4))

You may wonder, why not just simply specify the tensor ``shape``? Why we need these fancy stuffs like ``ti.root`` and ``place``? Aren't they doing the same thing?
Good question, let go forward and figure out why.


Row-major versus column-major
-----------------------------

Let's start with the simplest layout.

Since address space are linear in most modern architectures, for 1D Taichi tensors, the address of ``i``-th element is simply ``i``.

To store a multi-dimensional tensor, however, it has to be flattened, in order to fit into the 1D address space.
For example, to store a 2D tensor of size ``(3, 2)``, there are two way to do this:

    1. The address of ``(i, j)``-th is ``base + i * 2 + j`` (row-major).

    2. The address of ``(i, j)``-th is ``base + j * 3 + i`` (column-major).

To specify which layout to use in Taichi:

.. code-block:: python

    ti.root.dense(ti.i, 3).dense(ti.j, 2).place(x)    # row-major (default)
    ti.root.dense(ti.j, 2).dense(ti.i, 3).place(y)    # column-major

Both ``x`` and ``y`` have the same shape of ``(3, 2)``, and they can be accessed in the same manner, where ``0 <= i < 3 && 0 <= j < 2``. They can be accessed in the same manner: ``x[i, j]`` and ``y[i, j]``.
However, they have a very different memory layouts:

.. code-block::

    #     address low ........................... address high
    # x:  x[0,0]   x[0,1]   x[0,2] | x[1,0]   x[1,1]   x[1,2]
    # y:  y[0,0]   y[1,0] | y[0,1]   y[1,1] | y[0,2]   y[1,2]

See? ``x`` first increases the first index, while ``y`` first increases the second index. When overflow, reset to zero and begin to increase the second index.
This is row-major versus column-major storage.

.. note::

    For those people from C/C++, here's what they looks like:

    .. code-block:: c

        int x[3][2];  // row-major
        int y[2][3];  // column-major

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                do_something ( x[i][j] );
                do_something ( y[j][i] );
            }
        }

AoS versus SoA
--------------

Tensors of same size can be placed together.

For example, this places two 1D tensors of size ``3`` (array of structure, AoS):

.. code-block:: python

    ti.root.dense(ti.i, 3).place(x, y)

Their memory layout:

.. code-block::

    #  address low ............. address high
    #  x[0]   y[0] | x[1]  y[1] | x[2]   y[2]

In contrast, this places two tensor placed separately (structure of array, SoA):

.. code-block:: python

    ti.root.dense(ti.i, 3).place(x)
    ti.root.dense(ti.i, 3).place(y)

Now, their memory layout:

.. code-block::

    #  address low ............. address high
    #  x[0]  x[1]   x[2] | y[0]   y[1]   y[2]


Normally, you don't have to worry about the performance nuances between different layouts, and should just define the simplest layout as a start.
However, locality sometimes have a significant impact on the performance, especially when the tensor is huge.

**To improve spatial locality of memory accesses (i.e. cache hit rate / cacheline utilization), it's sometimes helpful to minimize the address differences of elements that are accessed together.**
Take a simple 1D wave equation solver for example:

.. code-block:: python

    N = 200000
    pos = ti.var(ti.f32)
    vel = ti.var(ti.f32)
    ti.root.dense(ti.i, N).place(pos)
    ti.root.dense(ti.i, N).place(vel)

    @ti.kernel
    def step():
        pos[i] += vel[i] * dt
        vel[i] += -k * pos[i] * dt


Here, we placed ``pos`` and ``vel`` seperately. So the distance in address space between ``pos[i]`` and ``vel[i]`` is ``200000``. This will result in a poor spatial locality and lots of cache-misses, which damages the performance.
A better placement is to place them together:

.. code-block:: python

    ti.root.dense(ti.i, N).place(pos, vel)

Then ``vel[i]`` is placed right next to ``pos[i]``, this can increases the cache-hit rate and therefore increases performance.


Multi-shaping (WIP)
-------------------

.. code-block:: python

    ti.root.dense(ti.ij, (32, 32)).dense(ti.ij, (4, 4))


Advanced layout (WIP)
---------------------

Advanced layouts other than ``dense``: ``dynamic``, ``pointer``, ``bitmasked``, ``hash``.
TODO: complete documention here.
