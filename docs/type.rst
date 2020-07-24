Type system
===========

Taichi supports many kind of data types, the type name is recognized as
a *prefix character* + a *digital number*.

The *prefix character* can be one of:

- ``i`` for signed integers, e.g. 233, -666
- ``u`` for unsigned integers, e.g. 233, 666
- ``f`` for floating point numbers, e.g. 2.33, 1e-4

The *digital number* can be one of:

- ``8``
- ``16``
- ``32``
- ``64``

It represents how much **bits** is used in storing the data.
The larger the bit number, the higher precision is.

For example, the two most commonly used types:

- ``i32`` represents a 32-bit signed integer.
- ``f32`` represents a 32-bit floating pointer number.

Supported types
---------------

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
    Supported types on each backend:

    +------+-----------+-----------+---------+
    | type | CPU/CUDA  | OpenGL    | Metal   |
    +======+===========+===========+=========+
    | i8   |    OK     |   N/A     |   OK    |
    +------+-----------+-----------+---------+
    | i16  |    OK     |   N/A     |   OK    |
    +------+-----------+-----------+---------+
    | i32  |    OK     |    OK     |   OK    |
    +------+-----------+-----------+---------+
    | i64  |    OK     |   EXT     |  N/A    |
    +------+-----------+-----------+---------+
    | u8   |    OK     |   N/A     |   OK    |
    +------+-----------+-----------+---------+
    | u16  |    OK     |   N/A     |   OK    |
    +------+-----------+-----------+---------+
    | u32  |    OK     |   N/A     |   OK    |
    +------+-----------+-----------+---------+
    | u64  |    OK     |   N/A     |  N/A    |
    +------+-----------+-----------+---------+
    | f32  |    OK     |    OK     |   OK    |
    +------+-----------+-----------+---------+
    | f64  |    OK     |    OK     |  N/A    |
    +------+-----------+-----------+---------+

    (OK: supported, EXT: require extension, N/A: not available)


.. note::

    Boolean types are represented using ``ti.i32``.



Type promotion
--------------

Binary operations on different types will give you a promoted type, following the C programming language, e.g.:

- ``i32 + f32 = f32`` (integer + float = float)
- ``i32 + i64 = i64`` (less-bits + more-bits = lamore-bits)

Basically it will try to choose the least precise type to contain the result value.


.. _default_precisions:

Default precisions
------------------

By default, numerical literals have 32-bit precisions.
For example, ``42`` has type ``ti.i32`` and ``3.14`` has type ``ti.f32``.

Default integer and float-point precisions (``default_ip`` and ``default_fp``) can be specified when initializing Taichi:

.. code-block:: python

    ti.init(..., default_fp=ti.f32)
    ti.init(..., default_fp=ti.f64)

    ti.init(..., default_ip=ti.i32)
    ti.init(..., default_ip=ti.i64)


Type casts
----------

Implicit casts
****************

WARNING: The type of a variable is **determinated on it's initialization**.

When a *wide* variable is assigned with a *narrow* type, it will be
implicitly promoted to the *wide* type and no warning will be raised:

.. code-block:: python

    a = 1.7
    a = 1
    print(a)  # 1.0

When a *narrow* variable is assigned with a *wide* type, it will be
implicitly casted into the *narrow* type and Taichi will raise a warning:

.. code-block:: python

    a = 1
    a = 1.7
    print(a)  # 1

Explicit casts
**************

You may use ``ti.cast`` to explicitly cast scalar values between different types:

.. code-block:: python

    a = 1.7
    b = ti.cast(a, ti.i32)  # 1
    c = ti.cast(b, ti.f32)  # 1.0

Equivalently, use ``int()`` and ``float()`` to convert values to default float-point/integer types:

.. code-block:: python

    a = 1.7
    b = int(a)    # 1
    c = float(a)  # 1.0

Casting vector / matrix elements
********************************

Type cast applied to vectors / matrices are element-wise:

.. code-block:: python

    u = ti.Vector([2.3, 4.7])
    v = int(u)  # ti.Vector([2, 4])

Bit casting
***********

Use ``ti.bit_cast`` to bit-cast a value into another data type. The underlying bits will be preserved in this cast.
The new type must have the same width as the the old type.

.. code-block::

    For people from C++, ``ti.bit_cast`` is equivalent to ``reinterpret_cast``.
