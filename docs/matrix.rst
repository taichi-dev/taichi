.. _matrix:

Matrices
========

- ``ti.Matrix`` is for small matrices (e.g. `3x3`) only. If you have `64x64` matrices, you should consider using a 2D tensor of scalars.
- ``ti.Vector`` is the same as ``ti.Matrix``, except that it has only one column.
- Differentiate element-wise product ``*`` and matrix product ``@``.
- ``ti.Vector(n, dt=ti.f32)`` or ``ti.Matrix(n, m, dt=ti.f32)`` to create tensors of vectors/matrices.
- ``ti.transposed(A)`` or simply ``A.T()``
- ``ti.inverse(A)``
- ``ti.Matrix.abs(A)``
- ``ti.tr(A)``
- ``ti.determinant(A, type)``
- ``ti.cross(a, b)``, where ``a`` and ``b`` are 3D vectors (i.e. ``3x1`` matrices)
- ``A.cast(type)``
- ``R, S = ti.polar_decompose(A, ti.f32)``
- ``U, sigma, V = ti.svd(A, ti.f32)`` (Note that ``sigma`` is a ``3x3`` diagonal matrix)

TODO: doc here better like Vector. WIP

A matrix in Taichi can have two forms:

  - as a temporary local variable. An ``n by m`` matrix consists of ``n * m`` scalar values.
  - as a an element of a global tensor. In this case, the tensor is an N-dimensional array of ``n by m`` matrices.

Declaration
-----------

As global tensors of matrices
+++++++++++++++++++++++++++++

.. function:: ti.Matrix(n, m, dt=type, shape=shape)

    :parameter n: (scalar) the number of rows in the matrix
    :parameter m: (scalar) the number of columns in the matrix
    :parameter type: (DataType) data type of the components
    :parameter shape: (scalar or tuple) shape the tensor of vectors, see :ref:`tensor`

    For example, this creates a 5x4 tensor of 3x3 matrices:
    ::

        # Python-scope
        a = ti.Matrix(3, 3, dt=ti.f32, shape=(5, 4))

.. note::

    In Python-scope, ``ti.var`` declares :ref:`scalar_tensor`, while ``ti.Matrix`` declares tensors of matrices.


As a temporary local variable
+++++++++++++++++++++++++++++

.. function:: ti.Matrix([x, y, ...])

    :parameter x: (scalar) the first component of the vector
    :parameter y: (scalar) the second component of the vector

    For example, this creates a 3x1 matrix with components (2, 3, 4):
    ::

        # Taichi-scope
        a = ti.Matrix([2, 3, 4])

.. note::

    this is equivalent to ti.Vector([x, y, ...])


.. function:: ti.Matrix([[x, y, ...], [z, w, ...], ...])

    :parameter x: (scalar) the first component of the first row
    :parameter y: (scalar) the second component of the first row
    :parameter z: (scalar) the first component of the second row
    :parameter w: (scalar) the second component of the second row

    For example, this creates a 2x2 matrix with components (2, 3) in the first row and (4, 5) in the second row:
    ::

        # Taichi-scope
        a = ti.Matrix([[2, 3], [4, 5])


.. function:: ti.Matrix(rows=[v0, v1, v2, ...])
.. function:: ti.Matrix(cols=[v0, v1, v2, ...])

    :parameter v0: (vector) vector of elements forming first row (or column)
    :parameter v1: (vector) vector of elements forming second row (or column)
    :parameter v2: (vector) vector of elements forming third row (or column)

    For example, this creates a 3x3 matrix by concactinating vectors into rows (or columns):
    ::

        # Taichi-scope
        v0 = ti.Vector([1.0, 2.0, 3.0])
        v1 = ti.Vector([4.0, 5.0, 6.0])
        v2 = ti.Vector([7.0, 8.0, 9.0])

        # to specify data in rows
        a = ti.Matrix(rows=[v0, v1, v2])

        # to specify data in columns instead
        a = ti.Matrix(cols=[v0, v1, v2])

        # lists can be used instead of vectors
        a = ti.Matrix(rows=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


Accessing components
--------------------

As global tensors of vectors
++++++++++++++++++++++++++++
.. attribute:: a[p, q, ...][i, j]

    :parameter a: (tensor of matrices) the tensor of matrices
    :parameter p: (scalar) index of the first tensor dimension
    :parameter q: (scalar) index of the second tensor dimension
    :parameter i: (scalar) row index of the matrix
    :parameter j: (scalar) column index of the matrix

    This extracts the first element in matrix ``a[6, 3]``:
    ::

        x = a[6, 3][0, 0]

        # or
        mat = a[6, 3]
        x = mat[0, 0]

.. note::

    **Always** use two pair of square brackets to access scalar elements from tensors of matrices.

     - The indices in the first pair of brackets locate the matrix inside the tensor of matrices;
     - The indices in the second pair of brackets locate the scalar element inside the matrix.

    For 0-D tensors of matrices, indices in the first pair of brackets should be ``[None]``.



As a temporary local variable
+++++++++++++++++++++++++++++

.. attribute:: a[i, j]

    :parameter a: (Matrix) the matrix
    :parameter i: (scalar) row index of the matrix
    :parameter j: (scalar) column index of the matrix

    For example, this extracts the element in row 0 column 1 of matrix ``a``:
    ::

        x = a[0, 1]

    This sets the element in row 1 column 3 of ``a`` to 4:
    ::

        a[1, 3] = 4

Methods
-------

TODO: WIP
