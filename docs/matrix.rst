.. _matrix:

Matrices
========

- ``ti.Matrix`` is for small matrices (e.g. `3x3`) only. If you have `64x64` matrices, you should consider using a 2D scalar field.
- ``ti.Vector`` is the same as ``ti.Matrix``, except that it has only one column.
- Differentiate element-wise product ``*`` and matrix product ``@``.
- ``ti.Vector.field(n, dtype=ti.f32)`` or ``ti.Matrix.field(n, m, dtype=ti.f32)`` to create vector/matrix fields.
- ``A.transpose()``
- ``R, S = ti.polar_decompose(A, ti.f32)``
- ``U, sigma, V = ti.svd(A, ti.f32)`` (Note that ``sigma`` is a ``3x3`` diagonal matrix)
- ``any(A)`` (Taichi-scope only)
- ``all(A)`` (Taichi-scope only)

TODO: doc here better like Vector. WIP

A matrix in Taichi can have two forms:

  - as a temporary local variable. An ``n by m`` matrix consists of ``n * m`` scalar values.
  - as a an element of a global field. In this case, the field is an N-dimensional array of ``n by m`` matrices.

Declaration
-----------

As global matrix fields
+++++++++++++++++++++++

.. function:: ti.Matrix.field(n, m, dtype, shape = None, offset = None)

    :parameter n: (scalar) the number of rows in the matrix
    :parameter m: (scalar) the number of columns in the matrix
    :parameter dtype: (DataType) data type of the components
    :parameter shape: (optional, scalar or tuple) shape of the matrix field, see :ref:`tensor`
    :parameter offset: (optional, scalar or tuple) see :ref:`offset`

    For example, this creates a 5x4 matrix field with each entry being a 3x3 matrix:
    ::

        # Python-scope
        a = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(5, 4))

.. note::

    In Python-scope, ``ti.field`` declares a :ref:`scalar_tensor`, while ``ti.Matrix.field`` declares a matrix field.


As a temporary local variable
+++++++++++++++++++++++++++++

.. function:: ti.Matrix([[x, y, ...], [z, w, ...], ...])

    :parameter x: (scalar) the first component of the first row
    :parameter y: (scalar) the second component of the first row
    :parameter z: (scalar) the first component of the second row
    :parameter w: (scalar) the second component of the second row

    For example, this creates a 2x3 matrix with components (2, 3, 4) in the first row and (5, 6, 7) in the second row:
    ::

        # Taichi-scope
        a = ti.Matrix([[2, 3, 4], [5, 6, 7]])


.. function:: ti.Matrix.rows([v0, v1, v2, ...])
.. function:: ti.Matrix.cols([v0, v1, v2, ...])

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
        a = ti.Matrix.rows([v0, v1, v2])

        # to specify data in columns instead
        a = ti.Matrix.cols([v0, v1, v2])

        # lists can be used instead of vectors
        a = ti.Matrix.rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


Accessing components
--------------------

As global matrix fields
+++++++++++++++++++++++
.. attribute:: a[p, q, ...][i, j]

    :parameter a: (ti.Matrix.field) the matrix field
    :parameter p: (scalar) index of the first field dimension
    :parameter q: (scalar) index of the second field dimension
    :parameter i: (scalar) row index of the matrix
    :parameter j: (scalar) column index of the matrix

    This extracts the first element in matrix ``a[6, 3]``:
    ::

        x = a[6, 3][0, 0]

        # or
        mat = a[6, 3]
        x = mat[0, 0]

.. note::

    **Always** use two pair of square brackets to access scalar elements from matrix fields.

     - The indices in the first pair of brackets locate the matrix inside the matrix fields;
     - The indices in the second pair of brackets locate the scalar element inside the matrix.

    For 0-D matrix fields, indices in the first pair of brackets should be ``[None]``.



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

.. function:: a.transpose()

    :parameter a: (ti.Matrix) the matrix
    :return: (ti.Matrix) the transposed matrix of ``a``.

    For example::

        a = ti.Matrix([[2, 3], [4, 5]])
        b = a.transpose()
        # Now b = ti.Matrix([[2, 4], [3, 5]])

    .. note::

        ``a.transpose()`` will not effect the data in ``a``, it just return the result.


.. function:: a.trace()

    :parameter a: (ti.Matrix) the matrix
    :return: (scalar) the trace of matrix ``a``.

    The return value can be computed as ``a[0, 0] + a[1, 1] + ...``.


.. function:: a.determinant()

    :parameter a: (ti.Matrix) the matrix
    :return: (scalar) the determinant of matrix ``a``.

    .. note::

        The matrix size of matrix must be 1x1, 2x2, 3x3 or 4x4 for now.

        This function only works in Taichi-scope for now.


.. function:: a.inverse()

    :parameter a: (ti.Matrix) the matrix
    :return: (ti.Matrix) the inverse of matrix ``a``.

    .. note::

        The matrix size of matrix must be 1x1, 2x2, 3x3 or 4x4 for now.

        This function only works in Taichi-scope for now.
