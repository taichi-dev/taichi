.. _meta:

Metaprogramming
===============

Taichi provides metaprogramming infrastructures. Metaprogramming can

* Unify the development of dimensionality-dependent code, such as 2D/3D physical simulations
* Improve run-time performance by from run-time costs to compile time
* Simplify the development of Taichi standard library

Taichi kernels are *lazily instantiated* and a lot of computation can happen at *compile-time*. Every kernel in Taichi is a template kernel, even if it has no template arguments.


.. _template_metaprogramming:

Template metaprogramming
------------------------

You may use ``ti.template()``
as a type hint to pass a field as an argument. For example:

.. code-block:: python

    @ti.kernel
    def copy(x: ti.template(), y: ti.template()):
        for i in x:
            y[i] = x[i]

    a = ti.field(ti.f32, 4)
    b = ti.field(ti.f32, 4)
    c = ti.field(ti.f32, 12)
    d = ti.field(ti.f32, 12)
    copy(a, b)
    copy(c, d)


As shown in the example above, template programming may enable us to reuse our
code and provide more flexibility.


Dimensionality-independent programming using grouped indices
------------------------------------------------------------

However, the ``copy`` template shown above is not perfect. For example, it can only be
used to copy 1D fields. What if we want to copy 2D fields? Do we have to write
another kernel?

.. code-block:: python

    @ti.kernel
    def copy2d(x: ti.template(), y: ti.template()):
        for i, j in x:
            y[i, j] = x[i, j]

Not necessary! Taichi provides ``ti.grouped`` syntax which enables you to pack
loop indices into a grouped vector to unify kernels of different dimensionalities.
For example:

.. code-block:: python

    @ti.kernel
    def copy(x: ti.template(), y: ti.template()):
        for I in ti.grouped(y):
            # I is a vector with same dimensionality with x and data type i32
            # If y is 0D, then I = ti.Vector([]), which is equivalent to `None` when used in x[I]
            # If y is 1D, then I = ti.Vector([i])
            # If y is 2D, then I = ti.Vector([i, j])
            # If y is 3D, then I = ti.Vector([i, j, k])
            # ...
            x[I] = y[I]

    @ti.kernel
    def array_op(x: ti.template(), y: ti.template()):
        # if field x is 2D:
        for I in ti.grouped(x): # I is simply a 2D vector with data type i32
            y[I + ti.Vector([0, 1])] = I[0] + I[1]

        # then it is equivalent to:
        for i, j in x:
            y[i, j + 1] = i + j


Field metadata
--------------

Sometimes it is useful to get the data type (``field.dtype``) and shape (``field.shape``) of fields.
These attributes can be accessed in both Taichi- and Python-scopes.

.. code-block:: python

  @ti.func
  def print_field_info(x: ti.template()):
    print('Field dimensionality is', len(x.shape))
    for i in ti.static(range(len(x.shape))):
      print('Size alone dimension', i, 'is', x.shape[i])
    ti.static_print('Field data type is', x.dtype)

See :ref:`scalar_tensor` for more details.

.. note::

    For sparse fields, the full domain shape will be returned.


Matrix & vector metadata
------------------------

Getting the number of matrix columns and rows will allow
you to write dimensionality-independent code. For example, this can be used to unify
2D and 3D physical simulators.

``matrix.m`` equals to the number of columns of a matrix, while ``matrix.n`` equals to
the number of rows of a matrix.
Since vectors are considered as matrices with one column, ``vector.n`` is simply
the dimensionality of the vector.

.. code-block:: python

  @ti.kernel
  def foo():
    matrix = ti.Matrix([[1, 2], [3, 4], [5, 6]])
    print(matrix.n)  # 2
    print(matrix.m)  # 3
    vector = ti.Vector([7, 8, 9])
    print(vector.n)  # 3
    print(vector.m)  # 1



Compile-time evaluations
------------------------

Using compile-time evaluation will allow certain computations to happen when kernels are being instantiated.
This saves the overhead of those computations at runtime.

* Use ``ti.static`` for compile-time branching (for those who come from C++17, this is `if constexpr <https://en.cppreference.com/w/cpp/language/if>`_.):

.. code-block:: python

   enable_projection = True

   @ti.kernel
   def static():
     if ti.static(enable_projection): # No runtime overhead
       x[0] = 1


* Use ``ti.static`` for forced loop unrolling:

.. code-block:: python

  @ti.kernel
  def func():
    for i in ti.static(range(4)):
        print(i)

    # is equivalent to:
    print(0)
    print(1)
    print(2)
    print(3)


When to use for loops with ``ti.static``
----------------------------------------

There are several reasons why ``ti.static`` for loops should be used.

 - Loop unrolling for performance.
 - Loop over vector/matrix elements. Indices into Taichi matrices must be a compile-time constant. Indexing into taichi fields can be run-time variables. For example, if you want to access a vector field ``x``, accessed as ``x[field_index][vector_component_index]``. The first index can be variable, yet the second must be a constant.

For example, code for resetting this vector fields should be

.. code-block:: python

   @ti.kernel
   def reset():
     for i in x:
       for j in ti.static(range(x.n)):
         # The inner loop must be unrolled since j is a vector index instead
         # of a global field index.
         x[i][j] = 0
