.. _offset:

Coordinate offsets
==================

- A Taichi field can be defined with **coordinate offsets**. The offsets will move field bounds so that field origins are no longer zero vectors. A typical use case is to support voxels with negative coordinates in physical simulations.
- For example, a matrix of ``32x64`` elements with coordinate offset ``(-16, 8)`` can be defined as the following:

.. code-block:: python

    a = ti.Matrix(2, 2, dtype=ti.f32, shape=(32, 64), offset=(-16, 8))

In this way, the field's indices are from ``(-16, 8)`` to ``(16, 72)`` (exclusive).

.. code-block:: python

    a[-16, 32]  # lower left corner
    a[16, 32]   # lower right corner
    a[-16, 64]  # upper left corner
    a[16, 64]   # upper right corner

.. note:: The dimensionality of field shapes should **be consistent** with that of the offset. Otherwise, a ``AssertionError`` will be raised.

.. code-block:: python

    a = ti.Matrix.field(2, 3, dtype=ti.f32, shape=(32,), offset=(-16, ))          # Works!
    b = ti.Vector.field(3, dtype=ti.f32, shape=(16, 32, 64), offset=(7, 3, -4))   # Works!
    c = ti.Matrix.field(2, 1, dtype=ti.f32, shape=None, offset=(32,))             # AssertionError
    d = ti.Matrix.field(3, 2, dtype=ti.f32, shape=(32, 32), offset=(-16, ))       # AssertionError
    e = ti.field(dtype=ti.i32, shape=16, offset=-16)                          # Works!
    f = ti.field(dtype=ti.i32, shape=None, offset=-16)                        # AssertionError
    g = ti.field(dtype=ti.i32, shape=(16, 32), offset=-16)                    # AssertionError
