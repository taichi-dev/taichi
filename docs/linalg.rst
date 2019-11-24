Linear Algebra
===============================================

Matrices
---------------------------------------
- ``ti.Matrix`` are for small matrices (e.g. `3x3`) only. If you have `64x64` matrices, you should consider using a 2D tensor of scalars.
- ``ti.Vector`` is the same as ``ti.Matrix``, except that it has only one column.
- Differentiate element-wise product ``*`` and matrix product ``@``.

Vectors
---------------------------------------

Vectors are special matrices.
