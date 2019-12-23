.. _linalg:

Linear algebra
===============================================

Matrices
---------------------------------------
- ``ti.Matrix`` is for small matrices (e.g. `3x3`) only. If you have `64x64` matrices, you should consider using a 2D tensor of scalars.
- ``ti.Vector`` is the same as ``ti.Matrix``, except that it has only one column.
- Differentiate element-wise product ``*`` and matrix product ``@``.
- ``ti.transposed(A)``
- ``ti.inverse(A)``
- ``ti.Matrix.abs(A)``
- ``ti.trace(A)``
- ``ti.determinant(A, type)``
- ``A.cast(type)``
- ``R, S = ti.polar_decompose(A, ti.f32)``
- ``U, sigma, V = ti.svd(A, ti.f32)`` (Note that ``sigma`` is a ``3x3`` diagonal matrix)


Vectors
---------------------------------------
Vectors are special matrices with only 1 column. In fact, ``ti.Vector`` is just an alias of ``ti.Matrix``.

- Dot product: ``a.dot(b)``, where ``a`` and ``b`` are vectors. ``ti.transposed(a) @ b`` will give you a ``matrix`` of size ``1x1``, which is not a `scalar`.
- Outer product: ``ti.outer_product(a, b)``
- l-2 norm: ``a.norm(eps = 0)``

    - returns ``sqrt(\sum_i(x_i ^ 2) + eps)``
    - Set ``eps = 1e-5`` for example, to safe guards the operator's gradient on zero vectors during differentiable programming.