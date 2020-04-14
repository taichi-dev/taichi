.. _atomic:

Atomic operations
=================

In Taichi, augmented assignments (e.g., ``x[i] += 1``) are automatically `atomic <https://en.wikipedia.org/wiki/Fetch-and-add>`_.


.. warning::

    When accumulating to global variables in parallel, make sure you use atomic operations. For example, to compute the sum of all elements in ``x``,
    ::

        @ti.kernel
        def sum():
            for i in x:
                # Approach 1: OK
                total[None] += x[i]

                # Approach 2: OK
                ti.atomic_add(total[None], x[i])

                # Approach 3: Wrong result since the operation is not atomic.
                total[None] = total[None] + x[i]


.. note::
    When atomic operations are applied to local values, the Taichi compiler will try to demote these operations into their non-atomic correspondence.

Apart from augmented assignments, explicit atomic operations such as ``ti.atomic_add`` also do read-modify-write atomically.
These operations additionally return the **old value** of the first argument. Below is the full list of explicit atomic operations:

.. function:: ti.atomic_add(x, y)
.. function:: ti.atomic_sub(x, y)

    Atomically compute ``x + y``/``x - y`` and store the result to ``x``.

    :return: The old value of ``x``.

    For example,
    ::

        x = 3
        y = 4
        z = ti.atomic_add(x, y)
        # now z = 7, y = 4, z = 3


.. function:: ti.atomic_and(x, y)
.. function:: ti.atomic_or(x, y)
.. function:: ti.atomic_xor(x, y)

    Atomically compute ``x & y`` (bitwise and), ``x | y`` (bitwise or), ``x ^ y`` (bitwise xor) and store the result to ``x``.

    :return: The old value of ``x``.


.. note::

    Supported atomic operations on each backend:

    +----------+-----------+-----------+---------+
    | type     | CPU/CUDA  | OpenGL    | Metal   |
    +==========+===========+===========+=========+
    | ``i32``  |    OK     |    OK     |   OK    |
    +----------+-----------+-----------+---------+
    | ``f32``  |    OK     |    OK     |   OK    |
    +----------+-----------+-----------+---------+
    | ``i64``  |    OK     |   EXT     |  MISS   |
    +----------+-----------+-----------+---------+
    | ``f64``  |    OK     |   EXT     |  MISS   |
    +----------+-----------+-----------+---------+

    (OK: supported; EXT:r equire extension; MISS: not supported)
