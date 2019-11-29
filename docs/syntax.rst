Syntax
==========================

Make sure you also check out the DiffTaichi paper (section "Language design" and "Appendix A") to learn more about the language.

Global tensors
--------------

* Every global variable is an N-dimensional tensor. Global `scalars` are treated as 0-D tensors.
* Global tensors are accessed using indices, e.g. ``x[i, j, k]`` if ``x`` is a 3D tensor. For 0-D tensor, access it as ``x[None]``.

  * Even when accessing 0-D tensor ``x``, use ``x[None] = 0`` instead of ``x = 0``. Please always use indexing to access entries in tensors.

* For a tensor ``F`` of element ``ti.Matrix``, make sure you first index the tensor dimensions, and then the matrix dimensions: ``F[i, j, k][0, 2]``. (Assuming ``F`` is a 3D tensor with ``ti.Matrix`` of size ``3x3`` as element)
* ``ti.Vector`` is simply an alias of ``ti.Matrix``.
* Tensors values are initially zero.
* Sparse tensors are initially inactive.

Defining your kernels
---------------------

Kernel arguments must be type hinted. Kernels can have at most 8 scalar parameters, e.g.,

.. code-block:: python

    @ti.kernel
    def print_xy(x: ti.i32, y: ti.f32):
      print(x + y)

* Restart the Taichi runtime system (clear memory, destroy all variables and kernels): ``ti.reset()``
* Right now kernels can have either statements or at most one for loop.

.. code-block:: python

    # Good kernels
    @ti.kernel
    def print(x: ti.i32, y: ti.f32):
      print(x + y)
      print(x * y)

    @ti.kernel
    def copy():
      for i in x:
        y[i] = x[i]

Bad kernels that won't compile correctly.
Compiler support coming soon. Please split them into two kernels for now.
For example:

.. code-block:: python

    # Bad kernel 1
    @ti.kernel
    def print():
      print(x[0])
      for i in x:
        y[i] = x[i]

    # Bad kernel 2
    @ti.kernel
    def print():
      for i in x:
        y[i] = x[i]
      for i in x:
        z[i] = x[i]

`Taichi`-scope (`ti.kernel`) v.s. `Python`-scope: everything decorated by `ti.kernel` is in `Taichi`-scope, which will be compiled by the Taichi compiler.

Functions
-----------------------------------------------

Use `@ti.func` to decorate your Taichi functions. These functions are callable only in `Taichi`-scope. Don't call them in `Python`-scope. All function calls are force-inlined, so no recursion supported.

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


Functions with multiple return values are not supported now. Use a local variable instead:

.. code-block:: python

  # Bad function
  @ti.func
  def safe_sqrt(x):
    if x >= 0:
      return ti.sqrt(x)
    else:
      return 0.0

  # Good function
  @ti.func
  def safe_sqrt(x):
    rst = 0.0
    if x >= 0:
      rst = ti.sqrt(x)
    else:
      rst = 0.0
    return rst


Data layout
-------------------
Non-power-of-two tensor dimensions are promoted into powers of two and thus these tensors will occupy more virtual address space.
For example, a tensor of size `(18, 65)` will be materialized as `(32, 128)`.


Scalar arithmetics
-----------------------------------------
Supported scalar functions:
* ``ti.sin(x)``
* ``ti.cos(x)``
* ``ti.cast(x, type)``
* ``ti.sqr(x)``
* ``ti.floor(x)``
* ``ti.inv(x)``
* ``ti.tan(x)``
* ``ti.tanh(x)``
* ``ti.exp(x)``
* ``ti.log(x)``
* ``ti.abs(x)``
* ``ti.random(type)``
* ``ti.max(a, b)`` Note: do not use native python ``max`` in Taichi kernels.
* ``ti.min(a, b)`` Note: do not use native python ``min`` in Taichi kernels.
* ``ti.length(dynamic_snode)``

Debugging
-------------------------------------------

Debug your program with `print(x)`.


Why Python frontend
-----------------------------------

Embedding the language in ``python`` has the following advantages:

* Easy to learn. Taichi has a very similar syntax to Python.
* Easy to run. No ahead-of-time compilation is needed.
* This design allows people to reuse existing python infrastructure:

  * IDEs. A python IDE simply works for TaichiLang, with syntax highlighting, checking, and autocomplete.
  * Package manager (pip). A developed Taichi application and be easily submitted to ``PyPI`` and others can easily set it up with ``pip``.
  * Existing packages. Interacting with other python components (e.g. ``matplotlib`` and ``numpy``) is just trivial.

* The built-in AST manipulation tools in ``python`` allow us to do magical things, as long as the kernel body can be parsed by the ``python`` parser.

However, this design decision has drawbacks as well:

* Taichi kernels must parse-able by Python parsers. This means Taichi syntax cannot go beyond Taichi syntax.

  * For example, indexing is always needed when accessing elements in Taichi tensors, even if the tensor is 0D. Use ``x[None] = 123`` to set the value in ``x`` if ``x`` is 0D. This is because ``x = 123`` will set ``x`` itself (instead of its containing value) to be the constant ``123`` in python syntax, and, unfortunately, we cannot modify this behavior.

* Python has relatively low performance. This can cause a performance issue when initializing large Taichi tensors with pure python scripts.
