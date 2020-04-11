.. _atomic:

Atomic operations
=================

In taichi, ``x[i] += 1`` is atomic but ``x[i] = x[i] + 1`` is not.

For example, to perform a reduction:
::
    @ti.kernel
    def sum():
        for i in x:
            result[None] += x[i]

Or use the function ``ti.atomic_add``, which is equivalent:
::
    @ti.kernel
    def sum():
        for i in x:
            ti.atomic_add(result[None], x[i])

See https://en.wikipedia.org/wiki/Fetch-and-add for more details.


.. note::
    Support of atomic operation on each backends:

    +------+-----------+-----------+---------+
    | type | LLVM      | OpenGL    | Metal   |
    +======+===========+===========+=========+
    | i32  |    OK     |    OK     |   OK    |
    +------+-----------+-----------+---------+
    | f32  |    OK     |    OK     |   OK    |
    +------+-----------+-----------+---------+
    | i64  |    OK     |   EXT     |  MISS   |
    +------+-----------+-----------+---------+
    | f64  |    OK     |   EXT     |  MISS   |
    +------+-----------+-----------+---------+

    (OK=supported, EXT=require extension, MISS=not supported)


Functions
---------

.. function:: ti.atomic_add(x, y)

    This is equivalent to ``x += y``.

    :parameter x: (Expr, lvalue) the LHS of addition, also where the result is saved
    :parameter y: (Expr) the RHS of addition
    :return: (Expr) the original value stored in ``x``

    e.g.:
    ::
        x = 3
        y = 4
        z = ti.atomic_add(x, y)
        # now x = 7, y = 4, z = 3


.. function:: ti.atomic_sub(x, y)

    This is equivalent to ``x -= y``.


.. function:: ti.atomic_max(x, y)

    e.g.:
    ::
        x = 3
        y = 4
        z = ti.atomic_max(x, y)
        # now x = 4, y = 4, z = 3


.. function:: ti.atomic_min(x, y)


.. function:: ti.atomic_or(x, y)

    This is equivalent to ``x |= y``.


.. function:: ti.atomic_and(x, y)

    This is equivalent to ``x &= y``.


.. function:: ti.atomic_xor(x, y)

    This is equivalent to ``x ^= y``.
