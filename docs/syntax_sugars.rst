Syntax sugars
==========================

Aliases
-------------------------------------------------------

Creating aliases for global variables and functions with cumbersome names can sometimes improve readability. In Taichi, this can be done by assigning kernel and function local variables with ``ti.static()``, which forces Taichi to use standard python pointer assignment.

For example, consider the simple kernel:

.. code-block:: python

  @ti.kernel
  def my_kernel():
    for i, j in field_a:
      field_b[i, j] = some_function(field_a[i, j])

The fields and function be aliased to new names with ``ti.static``:

.. code-block:: python

  @ti.kernel
  def my_kernel():
    a, b, fun = ti.static(field_a, field_b, some_function)
    for i,j in a:
      b[i,j] = fun(a[i,j])



Aliases can also be created for class members and methods, which can help prevent cluttering objective data-oriented programming code with ``self``.

For example, consider class kernel to compute the 2-D laplacian of some field:

.. code-block:: python

  @ti.kernel
  def compute_laplacian(self):
    for i, j in a:
      self.b[i, j] = (self.a[i + 1,j] - 2.0*self.a[i, j] + self.a[i-1, j])/(self.dx**2) \
                  + (self.a[i,j + 1] - 2.0*self.a[i, j] + self.a[i, j-1])/(self.dy**2)

Using ``ti.static()``, it can be simplified to:

.. code-block:: python

  @ti.kernel
  def compute_laplacian(self):
    a,b,dx,dy = ti.static(self.a,self.b,self.dx,self.dy)
    for i,j in a:
      b[i,j] = (a[i+1, j] - 2.0*a[i, j] + a[i-1, j])/(dx**2) \
             + (a[i, j+1] - 2.0*a[i, j] + a[i, j-1])/(dy**2)

.. note::

  ``ti.static`` can also be used in combination with ``if`` (compile-time branching) and ``for`` (compile-time unrolling). See :ref:`meta` for more details.

  Here, we are using it for *compile-time const values*, i.e. the **field/function handles** are constants at compile time.
