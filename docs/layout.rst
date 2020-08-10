.. _layout:

Advanced dense layouts
======================

Fields (:ref:`scalar_tensor`) can be *placed* in a specific shape and *layout*.
Defining a proper layout can be critical to performance, especially for memory-bound applications. A carefully designed data layout can significantly improve cache/TLB-hit rates and cacheline utilization. Although when performance is not the first priority, you probably don't have to worry about it.

Taichi decouples algorithms from data layouts, and the Taichi compiler automatically optimizes data accesses on a specific data layout. These Taichi features allow programmers to quickly experiment with different data layouts and figure out the most efficient one on a specific task and computer architecture.

In Taichi, the layout is defined in a recursive manner. See :ref:`snode` for more details about how this works. We suggest starting with the default layout specification (simply by specifying ``shape`` when creating fields using ``ti.field/ti.Vector.field/ti.Matrix.field``), and then migrate to more advanced layouts using the ``ti.root.X`` syntax if necessary.


From ``shape`` to ``ti.root.X``
-------------------------------

For example, this declares a 0-D field:

.. code-block:: python

    x = ti.field(ti.f32)
    ti.root.place(x)
    # is equivalent to:
    x = ti.field(ti.f32, shape=())

This declares a 1D field of size ``3``:

.. code-block:: python

    x = ti.field(ti.f32)
    ti.root.dense(ti.i, 3).place(x)
    # is equivalent to:
    x = ti.field(ti.f32, shape=3)

This declares a 2D field of shape ``(3, 4)``:

.. code-block:: python

    x = ti.field(ti.f32)
    ti.root.dense(ti.ij, (3, 4)).place(x)
    # is equivalent to:
    x = ti.field(ti.f32, shape=(3, 4))

You may wonder, why not simply specify the ``shape`` of the field? Why bother using the more complex version?
Good question, let go forward and figure out why.


Row-major versus column-major
-----------------------------

Let's start with the simplest layout.

Since address spaces are linear in modern computers, for 1D Taichi fields, the address of the ``i``-th element is simply ``i``.

To store a multi-dimensional field, however, it has to be flattened, in order to fit into the 1D address space.
For example, to store a 2D field of size ``(3, 2)``, there are two ways to do this:

    1. The address of ``(i, j)``-th is ``base + i * 2 + j`` (row-major).

    2. The address of ``(i, j)``-th is ``base + j * 3 + i`` (column-major).

To specify which layout to use in Taichi:

.. code-block:: python

    ti.root.dense(ti.i, 3).dense(ti.j, 2).place(x)    # row-major (default)
    ti.root.dense(ti.j, 2).dense(ti.i, 3).place(y)    # column-major

Both ``x`` and ``y`` have the same shape of ``(3, 2)``, and they can be accessed in the same manner, where ``0 <= i < 3 && 0 <= j < 2``. They can be accessed in the same manner: ``x[i, j]`` and ``y[i, j]``.
However, they have a very different memory layouts:

.. code-block:: none

    #     address low ........................... address high
    # x:  x[0,0]   x[0,1]   x[0,2] | x[1,0]   x[1,1]   x[1,2]
    # y:  y[0,0]   y[1,0] | y[0,1]   y[1,1] | y[0,2]   y[1,2]

See? ``x`` first increases the first index (i.e. row-major), while ``y`` first increases the second index (i.e. column-major).

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


Array of Structures (AoS), Structure of Arrays (SoA)
----------------------------------------------------

Fields of same size can be placed together.

For example, this places two 1D fields of size ``3`` (array of structure, AoS):

.. code-block:: python

    ti.root.dense(ti.i, 3).place(x, y)

Their memory layout:

.. code-block:: none

    #  address low ............. address high
    #  x[0]   y[0] | x[1]  y[1] | x[2]   y[2]

In contrast, this places two field placed separately (structure of array, SoA):

.. code-block:: python

    ti.root.dense(ti.i, 3).place(x)
    ti.root.dense(ti.i, 3).place(y)

Now, their memory layout:

.. code-block:: none

    #  address low ............. address high
    #  x[0]  x[1]   x[2] | y[0]   y[1]   y[2]


Normally, you don't have to worry about the performance nuances between different layouts, and should just define the simplest layout as a start.
However, locality sometimes have a significant impact on the performance, especially when the field is huge.

**To improve spatial locality of memory accesses (i.e. cache hit rate / cacheline utilization), it's sometimes helpful to place the data elements within relatively close storage locations if they are often accessed together.**
Take a simple 1D wave equation solver for example:

.. code-block:: python

    N = 200000
    pos = ti.field(ti.f32)
    vel = ti.field(ti.f32)
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

Then ``vel[i]`` is placed right next to ``pos[i]``, this can increase the cache-hit rate and therefore increase the performance.


Flat layouts versus hierarchical layouts
----------------------------------------

By default, when allocating a ``ti.field``, it follows the simplest data layout.

.. code-block:: python

  val = ti.field(ti.f32, shape=(32, 64, 128))
  # C++ equivalent: float val[32][64][128]

However, at times this data layout can be suboptimal for certain types of computer graphics tasks.
For example, ``val[i, j, k]`` and ``val[i + 1, j, k]`` are very far away (``32 KB``) from each other, and leads to poor access locality under certain computation tasks. Specifically, in tasks such as texture trilinear interpolation, the two elements are not even within the same ``4KB`` pages, creating a huge cache/TLB pressure.

A better layout might be

.. code-block:: python

  val = ti.field(ti.f32)
  ti.root.dense(ti.ijk, (8, 16, 32)).dense(ti.ijk, (4, 4, 4)).place(val)

This organizes ``val`` in ``4x4x4`` blocks, so that with high probability ``val[i, j, k]`` and its neighbours are close to each other (i.e., in the same cacheline or memory page).


Struct-fors on advanced dense data layouts
------------------------------------------

Struct-fors on nested dense data structures will automatically follow their data order in memory. For example, if 2D scalar field ``A`` is stored in row-major order,

.. code-block:: python

  for i, j in A:
    A[i, j] += 1

will iterate over elements of ``A`` following row-major order. If ``A`` is column-major, then the iteration follows the column-major order.

If ``A`` is hierarchical, it will be iterated level by level. This maximizes the memory bandwidth utilization in most cases.

Struct-for loops on sparse fields follow the same philosophy, and will be discussed further in :ref:`sparse`.


Examples
--------

2D matrix, row-major

.. code-block:: python

  A = ti.field(ti.f32)
  ti.root.dense(ti.ij, (256, 256)).place(A)

2D matrix, column-major

.. code-block:: python

  A = ti.field(ti.f32)
  ti.root.dense(ti.ji, (256, 256)).place(A) # Note ti.ji instead of ti.ij

`8x8` blocked 2D array of size `1024x1024`

.. code-block:: python

  density = ti.field(ti.f32)
  ti.root.dense(ti.ij, (128, 128)).dense(ti.ij, (8, 8)).place(density)


3D Particle positions and velocities, AoS

.. code-block:: python

  pos = ti.Vector.field(3, dtype=ti.f32)
  vel = ti.Vector.field(3, dtype=ti.f32)
  ti.root.dense(ti.i, 1024).place(pos, vel)
  # equivalent to
  ti.root.dense(ti.i, 1024).place(pos(0), pos(1), pos(2), vel(0), vel(1), vel(2))

3D Particle positions and velocities, SoA

.. code-block:: python

  pos = ti.Vector.field(3, dtype=ti.f32)
  vel = ti.Vector.field(3, dtype=ti.f32)
  for i in range(3):
    ti.root.dense(ti.i, 1024).place(pos(i))
  for i in range(3):
    ti.root.dense(ti.i, 1024).place(vel(i))
