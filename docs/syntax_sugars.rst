Syntax sugars
==========================

Static assignment
-------------------------------------------------------

``ti.static()`` can also be useful for simplifying syntax in objective data-oriented taichi code. It can be used assign kernel (and func) local variables to global class member variables. The local variable then points to the same object as the global variable and can be used throughout the kernel instead.  

For example, consider a simple class kernel to compute the 2-D laplacian of some tensor ``a``:

.. code-block:: python

  @ti.kernel
  def compute_laplacian():
    for i,j in a:
      self.b[i,j] = (self.a[i+1,j] - 2.0*self.a[i,j] + self.a[i-1,j])/(self.dx**2) \
                  + (self.a[i,j+1] - 2.0*self.a[i,j] + self.a[i,j-1])/(self.dy**2)

Using ``ti.static()``, it can be simplified to:

.. code-block:: python

  @ti.kernel
  def compute_laplacian():
    a,b,dx,dy = ti.static(self.a,self.b,self.dx,self.dy)
    for i,j in a:
      b[i,j] = (a[i+1,j] - 2.0*a[i,j] + a[i-1,j])/(dx**2) \
             + (a[i,j+1] - 2.0*a[i,j] + a[i,j-1])/(dy**2)
