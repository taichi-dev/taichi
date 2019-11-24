Type System
===============================================

Supported Types
---------------------------------------
Currently, supported basic types in Taichi are
  - int32 ``ti.i32``;
  - int64 ``ti.i64``;
  - float32 ``ti.f32``;
  - float64 ``ti.f64``.

Boolean types are represented using ``i32``.

Default Precision
---------------------------------------

By default, numerical literals have 32-bit precision.
For example, ``42`` has type ``ti.i32`` and ``3.14`` has type ``ti.f32``.
Default precisions can be altered using 

.. code-block:: python

  ti.get_runtime().set_default_fp(ti.f32)
  ti.get_runtime().set_default_fp(ti.f64)

  ti.get_runtime().set_default_ip(ti.i32)
  ti.get_runtime().set_default_ip(ti.i64)


Type Conversion
---------------------------------------

Use ``ti.cast`` to type cast scalar values.

# TODO: cast vector and matrices

.. code-block:: python

  a = 1.4
  b = ti.cast(a, ti.i32)
  c = ti.cast(b, ti.f32)

  # Equivalently, use ``int()`` and ``float()``
  #   to converting to default float-point/integer types
  b = int(a)
  c = float(b)
