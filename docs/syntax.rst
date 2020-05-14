Syntax
======

Kernels
-------

Kernel arguments must be type-hinted. Kernels can have at most 8 parameters, e.g.,

.. code-block:: python

    @ti.kernel
    def print_xy(x: ti.i32, y: ti.f32):
        print(x + y)


A kernel can have a **scalar** return value. If a kernel has a return value, it must be type-hinted.
The return value will be automatically cast into the hinted type. e.g.,

.. code-block:: python

    @ti.kernel
    def add_xy(x: ti.f32, y: ti.f32) -> ti.i32:
        return x + y  # same as: ti.cast(x + y, ti.i32)

    res = add_xy(2.3, 1.1)
    print(res)  # 3, since return type is ti.i32


.. note::

    For now, we only support one scalar as return value. Returning ``ti.Matrix`` or ``ti.Vector`` is not supported. Python-style tuple return is not supported either. For example:

    .. code-block:: python

        @ti.kernel
        def bad_kernel() -> ti.Matrix:
            return ti.Matrix([[1, 0], [0, 1]])  # Error

        @ti.kernel
        def bad_kernel() -> (ti.i32, ti.f32):
            x = 1
            y = 0.5
            return x, y  # Error


We also support **template arguments** (see :ref:`template_metaprogramming`) and **external array arguments** (see :ref:`external`) in Taichi kernels.

.. warning::

   When using differentiable programming, there are a few more constraints on kernel structures. See the **Kernel Simplicity Rule** in :ref:`differentiable`.

   Also, please do not use kernel return values in differentiable programming, since the return value will not be tracked by automatic differentiation. Instead, store the result into a global variable (e.g. ``loss[None]``).

Functions
---------

Use ``@ti.func`` to decorate your Taichi functions. These functions are callable only in `Taichi`-scope. Do not call them in `Python`-scopes.

.. code-block:: python

   @ti.func
   def laplacian(t, i, j):
       return inv_dx2 * (
           -4 * p[t, i, j] + p[t, i, j - 1] + p[t, i, j + 1] + p[t, i + 1, j] +
           p[t, i - 1, j])

   @ti.kernel
   def fdtd(t: ti.i32):
       for i in range(n_grid): # Parallelized
           for j in range(n_grid): # Serial loops in each parallel threads
               laplacian_p = laplacian(t - 2, i, j)
               laplacian_q = laplacian(t - 1, i, j)
               p[t, i, j] = 2 * p[t - 1, i, j] + (
                   c * c * dt * dt + c * alpha * dt) * laplacian_q - p[
                              t - 2, i, j] - c * alpha * dt * laplacian_p


.. warning::

    Functions with multiple ``return`` statements are not supported for now. Use a **local** variable to store the results, so that you end up with only one ``return`` statement:

    .. code-block:: python

      # Bad function - two return statements
      @ti.func
      def safe_sqrt(x):
        if x >= 0:
          return ti.sqrt(x)
        else:
          return 0.0

      # Good function - single return statement
      @ti.func
      def safe_sqrt(x):
        rst = 0.0
        if x >= 0:
          rst = ti.sqrt(x)
        else:
          rst = 0.0
        return rst

.. warning::

    Currently, all functions are force-inlined. Therefore, no recursion is allowed.

.. note::

    Function arguments are passed by value.



Scalar arithmetics
------------------
Supported scalar functions:

.. function:: ti.sin(x)
.. function:: ti.cos(x)
.. function:: ti.asin(x)
.. function:: ti.acos(x)
.. function:: ti.atan2(x, y)
.. function:: ti.cast(x, data_type)
.. function:: ti.sqrt(x)
.. function:: ti.floor(x)
.. function:: ti.ceil(x)
.. function:: ti.inv(x)
.. function:: ti.tan(x)
.. function:: ti.tanh(x)
.. function:: ti.exp(x)
.. function:: ti.log(x)
.. function:: ti.random(data_type)
.. function:: abs(x)
.. function:: int(x)
.. function:: float(x)
.. function:: max(x, y)
.. function:: min(x, y)
.. function:: pow(x, y)

.. note::

  Python 3 distinguishes ``/`` (true division) and ``//`` (floor division). For example, ``1.0 / 2.0 = 0.5``,
  ``1 / 2 = 0.5``, ``1 // 2 = 0``, ``4.2 // 2 = 2``. Taichi follows this design:

     - **true divisions** on integral types will first cast their operands to the default float point type.
     - **floor divisions** on float-point types will first cast their operands to the default integer type.

  To avoid such implicit casting, you can manually cast your operands to desired types, using ``ti.cast``.
  See :ref:`default_precisions` for more details on default numerical types.

.. note::

    When these scalar functions are applied on :ref:`matrix` and :ref:`vector`, they are applied in an element-wise manner.
    For example:

    .. code-block:: python

        B = ti.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        A = ti.sin(B)
        # is equivalent to
        for i in ti.static(range(2)):
            for j in ti.static(range(3)):
                A[i, j] = ti.sin(B[i, j])


Debugging
---------

Debug your program with ``print(x)``. For example, if ``x`` is ``23``, then it prints

.. code-block:: none

    [debug] x = 23

in the console.

.. warning::

    This is not the same as the ``print`` in Python-scope. For now ``print`` in Taichi only takes **scalar numbers** as input. Strings, vectors and matrices are not supported. Please use ``print(v[0]); print(v[1])`` if you want to print a vector.
