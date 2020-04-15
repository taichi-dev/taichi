.. _linalg:

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

TODO: doc here better like Vector.
