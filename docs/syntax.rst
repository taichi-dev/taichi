Syntax
==========================

Kernels
---------------------

Kernel arguments must be type-hinted. Kernels can have at most 8 parameters, e.g.,

.. code-block:: python

    @ti.kernel
    def print_xy(x: ti.i32, y: ti.f32):
      print(x + y)

    @ti.kernel
    def copy(x: ti.template(), y: ti.template()):
      for i in x:
        y[i] = x[i]


.. note::
    For differentiable programming kernels should better have either serial statements or a single parallel for-loop. If you don't use differentiable programming, feel free to ignore this tip.

    .. code-block:: python


        @ti.kernel
        def a_hard_kernel_to_auto_differentiate():
          sum = 0
          for i in x:
            sum += x[i]
          for i in y:
            y[i] = sum

        # instead, split it into multiple kernels to be nice to the Taichi autodiff compiler:

        @ti.kernel
        def reduce():
          for i in x:
            sum[None] += x[i]

        @ti.kernel
        def assign()
          for i in y:
            y[i] = sum[None]

        def main():
          with ti.Tape(loss):
            ...
            sum[None] = 0
            reduce()
            assign()
            ...


Functions
-----------------------------------------------

Use ``@ti.func`` to decorate your Taichi functions. These functions are callable only in `Taichi`-scope. Don't call them in `Python`-scope. All function calls are force-inlined, so no recursion supported.

.. code-block:: python

   @ti.func
   def laplacian(t, i, j):
     return inv_dx2 * (
         -4 * p[t, i, j] + p[t, i, j - 1] + p[t, i, j + 1] + p[t, i + 1, j] +
         p[t, i - 1, j])

   @ti.kernel
   def fdtd(t: ti.i32):
     for i in range(n_grid): # Parallelized over GPU threads
       for j in range(n_grid):
         laplacian_p = laplacian(t - 2, i, j)
         laplacian_q = laplacian(t - 1, i, j)
         p[t, i, j] = 2 * p[t - 1, i, j] + (
             c * c * dt * dt + c * alpha * dt) * laplacian_q - p[
                        t - 2, i, j] - c * alpha * dt * laplacian_p


.. warning::

    Functions with multiple ``return``'s are not supported for now. Use a **local** variable to store the results, so that you end up with only one ``return``:

    .. code-block:: python

      # Bad function - two return's
      @ti.func
      def safe_sqrt(x):
        if x >= 0:
          return ti.sqrt(x)
        else:
          return 0.0

      # Good function - single return
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


Data layout
-------------------
Non-power-of-two tensor dimensions are promoted into powers of two and thus these tensors will occupy more virtual address space.
For example, a tensor of size ``(18, 65)`` will be materialized as ``(32, 128)``.


Scalar arithmetics
-----------------------------------------
Supported scalar functions:

.. function:: ti.sin(x)
.. function:: ti.cos(x)
.. function:: ti.asin(x)
.. function:: ti.acos(x)
.. function:: ti.atan2(x, y)
.. function:: ti.cast(x, type)
.. function:: ti.sqr(x)
.. function:: ti.sqrt(x)
.. function:: ti.floor(x)
.. function:: ti.ceil(x)
.. function:: ti.inv(x)
.. function:: ti.tan(x)
.. function:: ti.tanh(x)
.. function:: ti.exp(x)
.. function:: ti.log(x)
.. function:: ti.random(type)
.. function:: abs(x)
.. function:: int(x)
.. function:: float(x)
.. function:: max(x, y)
.. function:: min(x, y)
.. function:: pow(x, y)

Note: when these scalar functions are applied on :ref:`matrix` and :ref:`vector`, it's applied element-wise, for example:

.. code-block:: python
   
    A = ti.sin(B)
    # is equalivant to (assuming B is a 3x2 matrix):
    for i in ti.static(range(3)):
        for j in ti.static(range(2)):
            A[i, j] = ti.sin(B[i, j])

.. note::

  Python 3 distinguishes ``/`` (true division) and ``//`` (floor division). For example, ``1.0 / 2.0 = 0.5``,
  ``1 / 2 = 0.5``, ``1 // 2 = 0``, ``4.2 // 2 = 2``. Taichi follows this design:

     - *true divisions* on integral types will first cast their operands to the default float point type.
     - *floor divisions* on float-point types will first cast their operands to the default integer type.

  To avoid such implicit casting, you can manually cast your operands to desired types, using ``ti.cast``.
  Read :ref:`default_precisions` for more details on default numerical types.

Debugging
-------------------------------------------

Debug your program with ``print(x)``. For example, if ``x`` is ``23``, then it shows:

.. code-block::

    [debug] x = 23

in the console.

.. note::

    For now ``print`` only takes scalar numbers as input. Strings, vectors and matrices are not supported. Please use ``print(v[0]); print(v[1])`` if you want to print a vector.


Why Python frontend
-----------------------------------

Embedding the language in ``python`` has the following advantages:

* Easy to learn. Taichi has a very similar syntax to Python.
* Easy to run. No ahead-of-time compilation is needed.
* This design allows people to reuse existing python infrastructure:

  * IDEs. A python IDE mostly works for Taichi with syntax highlighting, syntax checking, and autocomplete.
  * Package manager (pip). A developed Taichi application and be easily submitted to ``PyPI`` and others can easily set it up with ``pip``.
  * Existing packages. Interacting with other python components (e.g. ``matplotlib`` and ``numpy``) is just trivial.

* The built-in AST manipulation tools in ``python`` allow us to do magical things, as long as the kernel body can be parsed by the Python parser.

However, this design has drawbacks as well:

* Taichi kernels must parse-able by Python parsers. This means Taichi syntax cannot go beyond Python syntax.

  * For example, indexing is always needed when accessing elements in Taichi tensors, even if the tensor is 0D. Use ``x[None] = 123`` to set the value in ``x`` if ``x`` is 0D. This is because ``x = 123`` will set ``x`` itself (instead of its containing value) to be the constant ``123`` in python syntax, and, unfortunately, we cannot modify this behavior.

* Python has relatively low performance. This can cause a performance issue when initializing large Taichi tensors with pure python scripts. A Taichi kernel should be used to initialize a huge tensor.
