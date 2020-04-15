Type system
===============================================

Supported types
---------------------------------------
Currently, supported basic types in Taichi are

- int8 ``ti.i8``
- int16 ``ti.i16``
- int32 ``ti.i32``
- int64 ``ti.i64``
- uint8 ``ti.u8``
- uint16 ``ti.u16``
- uint32 ``ti.u32``
- uint64 ``ti.u64``
- float32 ``ti.f32``
- float64 ``ti.f64``

.. note::
    Supported types on each backends:

    +------+-----------+-----------+---------+
    | type | CPU/CUDA  | OpenGL    | Metal   |
    +======+===========+===========+=========+
    | i8   |    OK     |   MISS    |   OK    |
    +------+-----------+-----------+---------+
    | i16  |    OK     |   MISS    |   OK    |
    +------+-----------+-----------+---------+
    | i32  |    OK     |    OK     |   OK    |
    +------+-----------+-----------+---------+
    | i64  |    OK     |   EXT     |  MISS   |
    +------+-----------+-----------+---------+
    | u8   |    OK     |   MISS    |   OK    |
    +------+-----------+-----------+---------+
    | u16  |    OK     |   MISS    |   OK    |
    +------+-----------+-----------+---------+
    | u32  |    OK     |   MISS    |   OK    |
    +------+-----------+-----------+---------+
    | u64  |    OK     |   MISS    |  MISS   |
    +------+-----------+-----------+---------+
    | f32  |    OK     |    OK     |   OK    |
    +------+-----------+-----------+---------+
    | f64  |    OK     |    OK     |  MISS   |
    +------+-----------+-----------+---------+

    (OK: supported, EXT: require extension, MISS: not supported)


Boolean types are represented using ``ti.i32``.

Binary operations on different types will give you a promoted type, following the C programming language, e.g.

- ``i32 + f32 = f32``
- ``f32 + f64 = f64``
- ``i32 + i64 = i64``


.. _default_precisions:

Default precisions
---------------------------------------

By default, numerical literals have 32-bit precisions.
For example, ``42`` has type ``ti.i32`` and ``3.14`` has type ``ti.f32``.
Default precisions can be specified when initializing Taichi:

.. code-block:: python

  ti.init(..., default_fp=ti.f32)
  ti.init(..., default_fp=ti.f64)

  ti.init(..., default_ip=ti.i32)
  ti.init(..., default_ip=ti.i64)


Type casts
---------------------------------------

Use ``ti.cast`` to type-cast scalar values.

.. code-block:: python

  a = 1.4
  b = ti.cast(a, ti.i32)
  c = ti.cast(b, ti.f32)

  # Equivalently, use ``int()`` and ``float()``
  #   to converting to default float-point/integer types
  b = int(a)
  c = float(b)

  # Element-wise casts in matrices
  mat = ti.Matrix([[3.0, 0.0], [0.3, 0.1]])
  mat_int = mat.cast(int)
  mat_int2 = mat.cast(ti.i32)
