.. _layout:

Tensor layout
=============

Tensors (:ref:`scalar_tensor`) can be *placed* in a specific shape and *layout*.
Having a good layout can be the key to performance.

In Taichi, placing a layout is treated in a recursive manner.

For example, this declares a 0-D tensor:

.. code-block:: python

    x = ti.var(ti.f32)
    ti.root.place(x)
    # or
    x = ti.var(ti.f32, shape=())

This declares a 1-D tensor of size ``3``:

.. code-block:: python

    x = ti.var(ti.f32)
    ti.root.dense(ti.i, 3).place(x)
    # or
    x = ti.var(ti.f32, shape=3)

This declares a 1-D tensor of shape ``(3, 4)``:

.. code-block:: python

    x = ti.var(ti.f32)
    ti.root.dense(ti.ij, (3, 4)).place(x)
    # or
    x = ti.var(ti.f32, shape=(3, 4))

Now, you may say, why not just simply specify ``shape=`` argument to specify tensor shape? Why we need these fancy stuffs like ``ti.root`` and ``place``? Wasn't them the same?
Good question, let go forward and figure out why.


Multi-dimentional layout
------------------------

As we all know, most mordern computers have a 1-D memory model.
So it's not a big deal to store 1-D tensors, the address of ``i``-th can simply be ``base + i``.

To store a multi-D tensor, however, it has to be reshaped into 1-D, to fit into the 1-D address space.
For example, to store a 2-D tensor of size ``(3, 2)``, there are two way to do this:

    1. The address of ``(i, j)``-th is ``base + i * 2 + j``.
    1. The address of ``(i, j)``-th is ``base + j * 3 + i``.

To specify which layout to use in Taichi:

.. code-block:: python

    ti.root.dense(ti.ij, (3, 2)).place(x)    # 1 (default)
    ti.root.dense(ti.ji, (2, 3)).place(y)    # 2

They can be accessed in the same manner: ``x[i, j]`` and ``y[i, j]``.
However, they have a very different memory layout:

.. code-block::
    #     address low ..................... address high
    # x:  x[0,0]  x[0,1]  x[0,2]  x[1,0]  x[1,1]  x[1,2]
    # y:  y[0,0]  y[1,0]  y[0,1]  y[1,1]  y[0,2]  y[1,2]

See? ``x`` first increases the first index, while ``y`` first increases the second index. When overflow, reset to zero and begin to increase the second index.

.. note::

    For those people from C/C++:

    .. code-block:: c

        int x[3][2];  // 1
        int y[2][3];  // 2

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                do_something ( x[i][j] );
                do_something ( y[j][i] );
            }
        }

Place together
--------------

Tensors of same size can be placed together.

For example, this places two 1-D tensor of size ``3``:

.. code-block:: python

    ti.root.dense(ti.i, 3).place(x, y)

Their memory layout:

.. code-block::

    #  address low ......... address high
    #  x[0]  y[0]  x[1]  y[1]  x[2]  y[2]

In contrast, this places two tensor placed seperately:

.. code-block:: python

    ti.root.dense(ti.i, 3).place(x)
    ti.root.dense(ti.i, 3).place(y)

Now, their memory layout:

.. code-block::

    #  address low ......... address high
    #  x[0]  x[1]  x[2]  y[0]  y[1]  y[2]


Impact on performance
---------------------

The difference in layout is usually ignored by ordinal users.
However, locality sometimes have significant impact on performance especially when your tensor is huge.
It's better to place two often-used-together elements as close as possible.

Let's take a simple 1-D wave equation solver as example:

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


Here, we placed ``pos`` and ``vel`` seperately. So the distance in address space between ``pos[i]`` and ``vel[i]`` is ``200000``. This will break locality and cause a huge overhead of cache-miss, which damages performance.
A better placement is to place them together:

.. code-block:: python

    ti.root.dense(ti.i, N).place(pos, vel)


Advanced layout (WIP)
---------------------

Advanced layouts other than ``dense``: ``dynamic``, ``pointer``, ``bitmasked``, ``hash``.
TODO: complete documention here.
