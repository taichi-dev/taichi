.. _atomic:

Atomic operations
=================

Sometimes we want to access a common variable in different GPU threads. For example, we want to sum up the values in tensor ``x``:
::
    @ti.kernel
    def sum():
        for i in x:
            result[None] = result[None] + x[i]

This will be translated into (GPU pseudo-code):
::
    read from ``result`` to register 1
    read from ``x[i]`` to register 2
    add the value of reg1 and reg2, save result to reg3
    write back the value in register 3 to ``result``

This code is executed by many parallel GPU threads for different ``i``.
What if thread 1 is doing add, while thread 2 has already write back the ``result``?
Then the value ``x[1]`` won't be added to ``result``, it losts.

Then we want to ensure that the ``result`` is unchanged until we have done ``write back`` in one thread. We may think about ``lock``, but it's too expensive for GPU.
The good news is, the GPU hardware provides a instruction called ``atomic_add`` to combine the three instructions above:
::
    read from ``result`` to register 1
    atomically add the value of reg1 to ``result``

If we can make use of this hardware mechanism, the order of execution will preserved, great!

To perform an atomic add in Taichi, we can use the Argumented Assignment syntax:
::
    @ti.kernel
    def sum():
        for i in x:
            result[None] += x[i]

Or use the function ``ti.atomic_add``, both of them are equivalent:
::
    @ti.kernel
    def sum():
        for i in x:
            ti.atomic_add(result[None], x[i])


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
