Syntax
======

Taichi-scope vs Python-scope
----------------------------

Code decorated by ``@ti.kernel`` or ``@ti.func`` is in the **Taichi-scope**.

They are to be compiled and executed on CPU or GPU devices with high
parallelization performance, on the cost of less flexibility.

.. note::

    For people from CUDA, Taichi-scope = **device** side.


Code outside ``@ti.kernel`` or ``@ti.func`` is in the **Python-scope**.

They are not compiled by the Taichi compiler and have lower performance
but with a richer type system and better flexibility.

.. note::

    For people from CUDA, Python-scope = **host** side.


Kernels
-------

A Python function decorated by ``@ti.kernel`` is a **Taichi kernel**:

.. code-block:: python

   @ti.kernel
   def my_kernel():
       ...

   my_kernel()


Kernels should be called from **Python-scope**.

.. note::

    For people from CUDA, Taichi kernels = ``__global__`` functions.


Arguments
*********

Kernels can have at most 8 parameters so that you can pass values from
Python-scope to Taichi-scope easily.

Kernel arguments must be type-hinted:

.. code-block:: python

    @ti.kernel
    def my_kernel(x: ti.i32, y: ti.f32):
        print(x + y)

    my_kernel(2, 3.3)  # prints: 5.3

.. note::

    For now, we only support scalars as arguments. Specifying ``ti.Matrix`` or ``ti.Vector`` as argument is not supported. For example:

    .. code-block:: python

        @ti.kernel
        def bad_kernel(v: ti.Vector):
            ...

        @ti.kernel
        def good_kernel(vx: ti.f32, vy: ti.f32):
            v = ti.Vector([vx, vy])
            ...


Return value
************

A kernel may or may not have a **scalar** return value.
If it does, the type of return value must be hinted:

.. code-block:: python

    @ti.kernel
    def my_kernel() -> ti.f32:
        return 233.33

    print(my_kernel())  # 233.33


The return value will be automatically cast into the hinted type. e.g.,

.. code-block:: python

    @ti.kernel
    def add_xy() -> ti.i32:  # int32
        return 233.33

    print(my_kernel())  # 233, since return type is ti.i32


.. note::

    For now, a kernel can only have one scalar return value. Returning ``ti.Matrix`` or ``ti.Vector`` is not supported. Python-style tuple return is not supported either. For example:

    .. code-block:: python

        @ti.kernel
        def bad_kernel() -> ti.Matrix:
            return ti.Matrix([[1, 0], [0, 1]])  # Error

        @ti.kernel
        def bad_kernel() -> (ti.i32, ti.f32):
            x = 1
            y = 0.5
            return x, y  # Error


Advanced arguments
******************

We also support **template arguments** (see :ref:`template_metaprogramming`) and **external array arguments** (see :ref:`external`) in Taichi kernels.  Use ``ti.template()`` or ``ti.ext_arr()`` as their type-hints respectively.

.. note::

   When using differentiable programming, there are a few more constraints on kernel structures. See the **Kernel Simplicity Rule** in :ref:`differentiable`.

   Also, please do not use kernel return values in differentiable programming, since the return value will not be tracked by automatic differentiation. Instead, store the result into a global variable (e.g. ``loss[None]``).


Functions
---------

A Python function decorated by ``@ti.func`` is a **Taichi function**:

.. code-block:: python

    @ti.func
    def my_func():
        ...

    @ti.kernel
    def my_kernel():
        ...
        my_func()  # call functions from Taichi-scope
        ...

    my_kernel()    # call kernels from Python-scope


Taichi functions should be called from **Taichi-scope**.

.. note::

    For people from CUDA, Taichi functions = ``__device__`` functions.

.. note::

    Taichi functions can be nested.

.. warning::

    Currently, all functions are force-inlined. Therefore, no recursion is allowed.


Arguments and return values
***************************

Functions can have multiple arguments and return values.
Unlike kernels, arguments in functions don't need to be type-hinted:

.. code-block:: python

    @ti.func
    def my_add(x, y):
        return x + y


    @ti.kernel
    def my_kernel():
        ...
        ret = my_add(2, 3.3)
        print(ret)  # 5.3
        ...


Function arguments are passed by value. So changes made inside function scope
won't affect the outside value in the caller:

.. code-block:: python

    @ti.func
    def my_func(x):
        x = x + 1  # won't change the original value of x


    @ti.kernel
    def my_kernel():
        ...
        x = 233
        my_func(x)
        print(x)  # 233
        ...


Advanced arguments
******************

You may use ``ti.template()`` as type-hint to force arguments to be passed by
reference:

.. code-block:: python

    @ti.func
    def my_func(x: ti.template()):
        x = x + 1  # will change the original value of x


    @ti.kernel
    def my_kernel():
        ...
        x = 233
        my_func(x)
        print(x)  # 234
        ...


.. note::

    Unlike kernels, functions **do support vectors or matrices as arguments and return values**:

    .. code-block:: python

        @ti.func
        def sdf(u):  # functions support matrices and vectors as arguments. No type-hints needed.
            return u.norm() - 1

        @ti.kernel
        def render(d_x: ti.f32, d_y: ti.f32):  # kernels do not support vector/matrix arguments yet. We have to use a workaround.
            d = ti.Vector([d_x, d_y])
            p = ti.Vector([0.0, 0.0])
            t = sdf(p)
            p += d * t
            ...

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
        ret = 0.0
        if x >= 0:
          ret = ti.sqrt(x)
        else:
          ret = 0.0
        return ret


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
.. function:: ti.rsqrt(x)
.. function:: ti.floor(x)
.. function:: ti.ceil(x)
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
        C = ti.Matrix([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

        A = ti.sin(B)
        # is equivalent to
        for i in ti.static(range(2)):
            for j in ti.static(range(3)):
                A[i, j] = ti.sin(B[i, j])

        A = ti.pow(B, 2)
        # is equivalent to
        for i in ti.static(range(2)):
            for j in ti.static(range(3)):
                A[i, j] = ti.pow(B[i, j], 2)

        A = ti.pow(B, C)
        # is equivalent to
        for i in ti.static(range(2)):
            for j in ti.static(range(3)):
                A[i, j] = ti.pow(B[i, j], C[i, j])

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
