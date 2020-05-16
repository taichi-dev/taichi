.. _tensor:

Tensors and matrices
====================

Tensors are global variables provided by Taichi. Tensors can be either sparse or dense.
An element of a tensor can be either a scalar or a vector/matrix.

.. note::

    Although mathematically matrices are treated as 2D tensors, in Taichi, **tensor** and **matrix** are two completely different concepts.
    Matrices can be used as tensor elements, so you can have tensors with each element being a matrix.

Tensors of scalars
------------------
* Every global variable is an N-dimensional tensor.

  - Global ``scalars`` are treated as 0-D tensors of scalars.

* Tensors are always accessed using indices

   - E.g. ``x[i, j, k]`` if ``x`` is a scalar 3D tensor.
   - Even when accessing 0-D tensor ``x``, use ``x[None] = 0`` instead of ``x = 0``. Please **always** use indexing to access entries in tensors.

* Tensor values are initially zero.
* Sparse tensors are initially inactive.
* See :ref:`scalar_tensor` for more details.


Tensors of matrices
-------------------
Tensor elements can also be matrices.

Suppose you have a ``128 x 64`` tensor called ``A``, each element containing a ``3 x 2`` matrices. In this case you need to allocate a ``128 x 64`` tensor of ``3 x 2`` matrix, using the statement ``A = ti.Matrix(3, 2, dt=ti.f32, shape=(128, 64))``.

* If you want to get the matrix of grid node ``i, j``, please use ``mat = A[i, j]``. ``mat`` is simply a ``3 x 2`` matrix
* To get the element on the first row and second column of that matrix, use ``mat[0, 1]`` or ``A[i, j][0, 1]``.
* As you may have noticed, there are **two** indexing operators ``[]`` when you load an matrix element from a global tensor of matrices: the first is for tensor indexing, the second for matrix indexing.
* ``ti.Vector`` is simply an alias of ``ti.Matrix``.
* See :ref:`matrix` for more on matrices.


Matrix size
-----------
For performance reasons matrix operations will be unrolled, therefore we suggest using only small matrices.
For example, ``2x1``, ``3x3``, ``4x4`` matrices are fine, yet ``32x6`` is probably too big as a matrix size.

.. warning::

  Due to the unrolling mechanisms, operating on large matrices (e.g. ``32x128``) can lead to very long compilation time and low performance.

If you have a dimension that is too large (e.g. ``64``), it's better to declare a tensor of size ``64``.
E.g., instead of declaring ``ti.Matrix(64, 32, dt=ti.f32, shape=(3, 2))``, declare ``ti.Matrix(3, 2, dt=ti.f32, shape=(64, 32))``.
Try to put large dimensions to tensors instead of matrices.
