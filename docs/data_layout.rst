Advanced data layouts
===========================

Memory layout is key to performance, especially for memory-bound applications.
A carefully designed data layout can significantly improve cache/TLB-hit rates and cacheline utilization.

We suggested starting with the default layout specification (simply by specifying ``shape`` when creating tensors using ``ti.var/Vector/Matrix``),
and then migrate to more advanced layouts using the ``ti.root.X`` syntax.

Taichi decouples algorithms from data layouts, and the Taichi compiler automatically optimizes data accesses
on a specific data layout. These Taichi features allow programmers to quickly experiment with different data layouts
and figure out the most efficient one on a specific task and computer architecture.


The default data layout using ``shape``
-------------------------------------------------------

By default, when allocating a ``ti.var`` , it follows the most naive data layout

.. code-block:: python

  val = ti.var(ti.f32, shape=(32, 64, 128))
  # C++ equivalent: float val[32][64][128]

Or equivalently, the same data layout can be specified using advanced `data layout description`:

.. code-block:: python

  # Create the global tensor
  val = ti.var(ti.f32)
  # Specify the shape and layout
  ti.root.dense(ti.ijk, (32, 64, 128)).place(val)

However, oftentimes this data layout is suboptimal for computer graphics tasks.
For example, ``val[i, j, k]`` and ``val[i + 1, j, k]`` are very far away (``32 KB``) from each other,
and leads to poor access locality under certain computation tasks. Specifically,
in tasks such as texture trilinear interpolation, the two elements are not even within the same ``4KB`` pages,
creating a huge cache/TLB pressure.

Advanced data layout specification
--------------------------------------

A better layout might be

.. code-block:: python

  val = ti.var(ti.f32)
  ti.root.dense(ti.ijk, (8, 16, 32)).dense(ti.ijk, (4, 4, 4)).place(val)

This organizes ``val`` in ``4x4x4`` blocks, so that with high probability ``val[i, j, k]`` and its neighbours are close to each other (i.e., in the same cacheline or memory page).

Examples
-----------

2D matrix, row-major

.. code-block:: python

  A = ti.var(ti.f32)
  ti.root.dense(ti.ij, (256, 256)).place(A)

2D matrix, column-major

.. code-block:: python

  A = ti.var(ti.f32)
  ti.root.dense(ti.ji, (256, 256)).place(A) # Note ti.ji instead of ti.ij

`8x8` blocked 2D array of size `1024x1024`

.. code-block:: python

  density = ti.var(ti.f32)
  ti.root.dense(ti.ij, (128, 128)).dense(ti.ij, (8, 8)).place(density)


3D Particle positions and velocities, arrays-of-structures

.. code-block:: python

  pos = ti.Vector(3, dt=ti.f32)
  vel = ti.Vector(3, dt=ti.f32)
  ti.root.dense(ti.i, 1024).place(pos, vel)
  # equivalent to
  ti.root.dense(ti.i, 1024).place(pos(0), pos(1), pos(2), vel(0), vel(1), vel(2))

3D Particle positions and velocities, structures-of-arrays

.. code-block:: python

  pos = ti.Vector(3, dt=ti.f32)
  vel = ti.Vector(3, dt=ti.f32)
  for i in range(3):
    ti.root.dense(ti.i, 1024).place(pos(i))
  for i in range(3):
    ti.root.dense(ti.i, 1024).place(vel(i))
