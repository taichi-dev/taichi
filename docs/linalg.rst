.. _linalg:

Linear Algebra
===============================================

Matrices
---------------------------------------
- ``ti.Matrix`` are for small matrices (e.g. `3x3`) only. If you have `64x64` matrices, you should consider using a 2D tensor of scalars.
- ``ti.Vector`` is the same as ``ti.Matrix``, except that it has only one column.
- Differentiate element-wise product ``*`` and matrix product ``@``.
- ``ti.Matrix.transposed(A)``
- ``ti.Matrix.inverse(A)``
- ``ti.Matrix.abs(A)``
- ``ti.Matrix.trace(A)``
- ``ti.Matrix.determinant(A, type)``
- ``ti.Matrix.cast(A, type)``

Vectors
---------------------------------------
Vectors are special matrices with only 1 column. In fact, ``ti.Vector`` is just an alias of ``ti.Matrix``.

- Dot product: ``a.dot(b)``, where `a` and `b` are vectors. ``ti.Matrix.transposed(a) @ b`` will give you a `matrix` of size 1x1, which is not a `scalar`.
- ``ti.Vector.outer_product(a, b)``