.. _tensor:

Fields and matrices
===================

Fields are global variables provided by Taichi. Fields can be either sparse or dense.
An element of a field can be either a scalar or a vector/matrix.

.. note::

    Matrices can be used as field elements, so you can have fields with each element being a matrix.

Scalar fields
-------------
* Every global variable is an N-dimensional field.

  - Global ``scalars`` are treated as 0-D scalar fields.

* Fields are always accessed using indices

   - E.g. ``x[i, j, k]`` if ``x`` is a 3D scalar field.
   - Even when accessing 0-D field ``x``, use ``x[None] = 0`` instead of ``x = 0``. Please **always** use indexing to access entries in fields.

* Field values are initially zero.
* Sparse fields are initially inactive.
* See :ref:`scalar_tensor` for more details.


Matrix fields
-------------
Field elements can also be matrices.

Suppose you have a ``128 x 64`` field called ``A``, each element containing a ``3 x 2`` matrix. To allocate a ``128 x 64`` matrix field which has a ``3 x 2`` matrix for each of its entry, use the statement ``A = ti.Matrix.field(3, 2, dtype=ti.f32, shape=(128, 64))``.

* If you want to get the matrix of grid node ``i, j``, please use ``mat = A[i, j]``. ``mat`` is simply a ``3 x 2`` matrix
* To get the element on the first row and second column of that matrix, use ``mat[0, 1]`` or ``A[i, j][0, 1]``.
* As you may have noticed, there are **two** indexing operators ``[]`` when you load an matrix element from a global matrix field: the first is for field indexing, the second for matrix indexing.
* ``ti.Vector`` is simply an alias of ``ti.Matrix``.
* See :ref:`matrix` for more on matrices.


Matrix size
-----------
For performance reasons matrix operations will be unrolled, therefore we suggest using only small matrices.
For example, ``2x1``, ``3x3``, ``4x4`` matrices are fine, yet ``32x6`` is probably too big as a matrix size.

.. warning::

  Due to the unrolling mechanisms, operating on large matrices (e.g. ``32x128``) can lead to very long compilation time and low performance.

If you have a dimension that is too large (e.g. ``64``), it's better to declare a field of size ``64``.
E.g., instead of declaring ``ti.Matrix.field(64, 32, dtype=ti.f32, shape=(3, 2))``, declare ``ti.Matrix.field(3, 2, dtype=ti.f32, shape=(64, 32))``.
Try to put large dimensions to fields instead of matrices.
