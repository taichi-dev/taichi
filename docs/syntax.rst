Syntax
==========================

Make sure you also check out the DiffTaichi paper (section "Language design" and "Appendix A") to learn more about the language.

Global Tensors
--------------

* Every global variable is an N-dimensional tensor. Global scalars are treated as 0-D tensors.
* Global tensors are accessed using indices, e.g. ``x[i, j, k]`` if ``x`` is a 3D tensor. For 0-D tensor, access it as ``x[None]``.

  * If you access a 0-D tensor ``x`` using ``x = 0``\ , instead of ``x[None] = 0``\ , the handle ``x`` will be set to zero instead of the value in that tensor. This is a compromise to the native python semantics. So please always use indexing to access entries in tensors.

* For a tensor ``F`` of element ``ti.Matrix``\ , make sure you first index the tensor dimensions, and then the matrix dimensions: ``F[i, j, k][0, 2]``. (Assuming ``F`` is a 3D tensor with ``ti.Matrix`` of size ``3x3`` as element)
* ``ti.Vector`` is simply an alias of ``ti.Matrix``.
* Tensors values are initially zero.
* Sparse tensors are initially inactive.

Defining your kernels
---------------------


* Kernel arguments must be type hinted. Kernels can have at most 8 scalar parameters, e.g.
.. code-block:: python

    @ti.kernel
    def print_xy(x: ti.i32, y: ti.f32):
      print(x + y)

* Restart the Taichi runtime system (clear memory, desctory all variables and kernels) : ``ti.reset()``
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

Bad kernels that won't compile right now.
(split them into two kernels for now. Compiler support coming soon.)

.. code-block:: python

    @ti.kernel
    def print():
      print(x + y)
      for i in x:
        y[i] = x[i]

    @ti.kernel
    def print():
      for i in x:
        y[i] = x[i]
      for i in x:
        z[i] = x[i]

   - `Taichi`-scope (`ti.kernel`) v.s. `Python`-scope: everything decorated by `ti.kernel` is in `Taichi`-scope, which will be compiled by the Taichi compiler.

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


* Functions with multiple return values are not supported now. Use a local variable instead:
.. code-block:: python
  # Good function
  @ti.func
  def safe_sqrt(x):
  rst = 0.0
  if x >= 0:
   rst = ti.sqrt(x)
  else:
   rst = 0.0
  return rst

Bad function with two *return*\ s
*********************************

.. code-block:: python

  @ti.func
  def safe_sqrt(x):
    if x >= 0:
      return ti.sqrt(x)
    else:
      return 0.0


Data Layout
-------------------
Non-power-of-two tensor dimensions are promoted into powers of two. For example, a tensor of size `(18, 65)` will be materialized as `(32, 128)`. Be careful if you want to iterate over this structural node when it is dense: the loop variables will become iterate over the promoted large domain instead of the original compact domain. Use a range-for instead. For sparse structural nodes, this makes no difference.


Scalar Arithmetics
-----------------------------------------
- Supported scalar functions:

  * `ti.sin(x)`
  * `ti.cos(x)`
  * `ti.cast(x, type)`
  * `ti.sqr(x)`
  * `ti.floor(x)`
  * `ti.inv(x)`
  * `ti.tan(x)`
  * `ti.tanh(x)`
  * `ti.exp(x)`
  * `ti.log(x)`
  * `ti.abs(x)`
  * `ti.random(type)`
  * `ti.max(a, b)` Note: do not use native python `max` in Taichi kernels.
  * `ti.min(a, b)` Note: do not use native python `min` in Taichi kernels.
  * `ti.length(dynamic_snode)`

Debugging
-------------------------------------------

Debug your program with `print(x)`.

Performance Tips
-------------------------------------------

Avoid synchronization
When using GPU, an asynchronous task queue will be maintained. Whenever reading/writing global tensors, a synchronization will be invoked, which leads to idle cycles on CPU/GPU.
Make Use of GPU Shared Memory and L1-d$ `ti.cache_l1(x)` will enforce data loads related to `x` cached in L1-cache. `ti.cache_shared(x)` will allocate shared memory. TODO: add examples


Multi-Stage Programming
=======================================


* Use `ti.static` for compile-time branching (For those who come from C++17, this is `if constexpr <https://en.cppreference.com/w/cpp/language/if>`_.

.. code-block:: python


   enable_projection = True

   @ti.kernel
   def static():
     if ti.static(enable_projection): # No runtime overhead
       x[0] = 1


* Use ``ti.static`` for forced loop unrolling

.. code-block:: python


 @ti.kernel
 def g2p(f: ti.i32):
 for p in range(0, n_particles):
  base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
  fx = x[f, p] * inv_dx - ti.cast(base, real)
  w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0),
       0.5 * ti.sqr(fx - 0.5)]
  new_v = ti.Vector([0.0, 0.0])
  new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

  # Unrolled 9 iterations for higher performance
  for i in ti.static(range(3)):
    for j in ti.static(range(3)):
      dpos = ti.cast(ti.Vector([i, j]), real) - fx
      g_v = grid_v_out[base(0) + i, base(1) + j]
      weight = w[i](0) * w[j](1)
      new_v += weight * g_v
      new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx

  v[f + 1, p] = new_v
  x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
  C[f + 1, p] = new_C


When to use `ti.static`
-----------------------------------------



* Parameterize kernels with different global variables:

.. code-block:: python

  import taichi as ti

  x = ti.global_var(ti.f32)
  y = ti.global_var(ti.f32)
  z = ti.global_var(ti.f32)
  loss = ti.global_var(ti.f32)

  @ti.layout
  def tensors():
    ti.root.dense(ti.i, 16).place(x, y, z)
    ti.root.place(loss)
    ti.root.lazy_grad()


Why Python Frontend
-----------------------------------


Embedding the language in ``python`` has the following advantages:


* Easy to learn. Python itself is very easy to learn, so is PyTaichiLang.
* Easy to run. No ahead-of-time compilation is needed.
* It allows people to reuse existing python infrastructure:

  * IDEs. A python IDE simply works for TaichiLang, with syntax highlighting, checking, and autocomplete.
  * Package manager (pip). A developed Taichi application and be easily submitted to ``PyPI`` and others can easily set it up   with ``pip``.
  * Existing packages. Interacting with other python components is just trivial.

* The built-in AST manipulation tools in ``python`` allow us to do magical things, as long as the kernel body can be parsed by the ``python`` parser.

However, this design decision has drawbacks as well:

* Indexing is always needed when accessing elements in tensors, even if the tensor is 0D. Use ``x[None] = 123`` to set the value in ``x`` if ``x`` is 0D. This is because ``x = 123`` will set ``x`` itself (instead of its containing value) to be the constant ``123`` in python syntax, and unfortunately we cannot modify this behavior.
