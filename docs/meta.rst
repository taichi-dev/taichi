.. _meta:

Metaprogramming
=================================================

Taichi provides metaprogramming infrastructures. Metaprogramming can

* Unify the development of dimensionality-dependent code, such as 2D/3D physical simulations
* Improve run-time performance by from run-time costs to compile time
* Simplify the development of Taichi standard library

Taichi kernels are *lazily instantiated* and a lot of computation can happen at *compile-time*. Every kernel in Taichi is a template kernel, even if it has no template arguments.


.. _template_metaprogramming:

Template metaprogramming
------------------------

.. code-block:: python

    @ti.kernel
    def copy(x: ti.template(), y: ti.template()):
        for i in x:
            y[i] = x[i]


Dimensionality-independent programming using grouped indices
-------------------------------------------------------------

.. code-block:: python

    @ti.kernel
    def copy(x: ti.template(), y: ti.template()):
        for I in ti.grouped(y):
            x[I] = y[I]

    @ti.kernel
    def array_op(x: ti.template(), y: ti.template()):
        # If tensor x is 2D
        for I in ti.grouped(x): # I is a vector of size x.dim() and data type i32
            y[I + ti.Vector([0, 1])] = I[0] + I[1]
        # is equivalent to
        for i, j in x:
            y[i, j + 1] = i + j

Tensor size reflection
------------------------------------------

Sometimes it will be useful to get the dimensionality (``tensor.dim()``) and shape (``tensor.shape()``) of tensors.
These functions can be used in both Taichi kernels and python scripts.

.. code-block:: python

  @ti.func
  def print_tensor_size(x: ti.template()):
    print(x.dim())
    for i in ti.static(range(x.dim())):
      print(x.shape()[i])

For sparse tensors, the full domain shape will be returned.

Compile-time evaluations
------------------------------------------
Using compile-time evaluation will allow certain computation to happen when kernels are instantiated.
Such computation has no overhead at runtime.

* Use ``ti.static`` for compile-time branching (for those who come from C++17, this is `if constexpr <https://en.cppreference.com/w/cpp/language/if>`_.)

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


When to use for loops with ``ti.static``
-----------------------------------------

There are several reasons why ``ti.static`` for loops should be used.

 - Loop unrolling for performance.
 - Loop over vector/matrix elements. Indices into Taichi matrices must be a compile-time constant. Indexing into taichi tensors can be run-time variables. For example, if ``x`` is a 1-D tensor of 3D vector, accessed as ``x[tensor_index][matrix index]``. The first index can be variable, yet the second must be a constant.

For example, code for resetting this tensor of vectors should be

.. code-block:: python

   @ti.kernel
   def reset():
     for i in x:
       for j in ti.static(range(3)):
         # The inner loop must be unrolled since j is a vector index instead
         # of a global tensor index.
         x[i][j] = 0
