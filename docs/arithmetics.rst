Scalar operations
=================

Operators
---------

Arithmetic operators
********************

- ``-a``
- ``a + b``
- ``a - b``
- ``a * b``
- ``a / b``
- ``a // b``
- ``a % b``
- ``a ** b``

.. note::

  The ``%`` operator in Taichi follows the Python style instead of C style, e.g.:

  .. code-block:: python

    # no matter Taichi-scope or Python-scope:
    print(2 % 3)   # 2
    print(-2 % 3)  # 1

  For C-style mod, please use ``ti.raw_mod``:

  .. code-block:: python

    print(ti.raw_mod(2, 3))   # 2
    print(ti.raw_mod(-2, 3))  # -2

.. note::

  Python 3 distinguishes ``/`` (true division) and ``//`` (floor division). For example, ``1.0 / 2.0 = 0.5``,
  ``1 / 2 = 0.5``, ``1 // 2 = 0``, ``4.2 // 2 = 2``. And Taichi follows the same design:

     - **true divisions** on integral types will first cast their operands to the default float point type.
     - **floor divisions** on float-point types will first cast their operands to the default integer type.

  To avoid such implicit casting, you can manually cast your operands to desired types, using ``ti.cast``.
  See :ref:`default_precisions` for more details on default numerical types.

Logic operators
***************

- ``~a``
- ``a == b``
- ``a != b``
- ``a > b``
- ``a < b``
- ``a >= b``
- ``a <= b``
- ``not a``
- ``a or b``
- ``a and b``
- ``a if cond else b``

Bitwise operators
*****************

- ``a & b``
- ``a ^ b``
- ``a | b``

Functions
---------

Trigonometric functions
***********************

.. function:: ti.sin(x)
.. function:: ti.cos(x)
.. function:: ti.tan(x)
.. function:: ti.asin(x)
.. function:: ti.acos(x)
.. function:: ti.atan2(y, x)
.. function:: ti.tanh(x)

Other arithmetic functions
**************************

.. function:: ti.sqrt(x)
.. function:: ti.rsqrt(x)

   A fast version for ``1 / ti.sqrt(x)``.

.. function:: ti.exp(x)
.. function:: ti.log(x)
.. function:: ti.floor(x)
.. function:: ti.ceil(x)

Casting types
*************

.. function:: ti.cast(x, dtype)

    See :ref:`type` for more details.

.. function:: int(x)

   A shortcut for ``ti.cast(x, int)``.

.. function:: float(x)

   A shortcut for ``ti.cast(x, float)``.

Builtin-alike functions
***********************

.. function:: abs(x)
.. function:: max(x, y, ...)
.. function:: min(x, y, ...)
.. function:: pow(x, y)

   Same as ``x ** y``.

Random number generator
***********************

.. function:: ti.random(dtype = float)

    Generates a uniform random float or integer number.

.. function:: ti.randn(dtype = None)

    Generates a random floating point number from the standard normal distribution.

Element-wise arithmetics for vectors and matrices
-------------------------------------------------

When these scalar functions are applied on :ref:`matrix` and :ref:`vector`, they are applied in an element-wise manner.
For example:

.. code-block:: python

    B = ti.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    C = ti.Matrix([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

    A = ti.sin(B)
    # is equivalent to
    for i in ti.static(range(2)):
        for j in ti.static(range(3)):
            A[i, j] = ti.sin(B[i, j])

    A = B ** 2
    # is equivalent to
    for i in ti.static(range(2)):
        for j in ti.static(range(3)):
            A[i, j] = B[i, j] ** 2

    A = B ** C
    # is equivalent to
    for i in ti.static(range(2)):
        for j in ti.static(range(3)):
            A[i, j] = B[i, j] ** C[i, j]

    A += 2
    # is equivalent to
    for i in ti.static(range(2)):
        for j in ti.static(range(3)):
            A[i, j] += 2

    A += B
    # is equivalent to
    for i in ti.static(range(2)):
        for j in ti.static(range(3)):
            A[i, j] += B[i, j]
