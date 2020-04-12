.. _linalg:

Linear Algebra
==============

Vectors
-------

A ``n``-D vector is composed of ``n`` variables. It's just a syntax sugar to abstract the linear algebra framework. There is no real difference in lower execution for now.


Creation
++++++++

As Tensor
:::::::::

.. function:: ti.Vector(n, dt=type, shape=shape)

    :parameter n: (scalar) how many compoments / dimensions of the vector
    :parameter type: (DataType) data type of the compoments
    :parameter shape: (scalar or tuple) shape of the tensor, see :ref:`tensor`

    This creates a 3x3 tensor of 3D vectors:
    ::
        # (python-scope)
        a = ti.Vector(3, dt=ti.f32, shape=(3, 3))

.. note::

    In python-scope, ``ti.var`` is a :ref:`scalar_tensor`, while ``ti.Vector`` is a tensor of vectors.
 

As Temporary Variable
:::::::::::::::::::::

.. function:: ti.Vector([x, y, ...])

    :parameter x: (scalar) the first compoment of vector
    :parameter y: (scalar) the second compoment of vector

    This creates a 3D vector with compoments initialized with (2, 3, 4):
    ::
        # (taichi-scope)
        a = ti.Vector([2, 3, 4])
 

Attributes
++++++++++

.. attribute:: a[i]

    :parameter a: (Vector) the vector
    :parameter i: (scalar) index of the compoment

    This extracts the first compoment of vector ``a``:
    ::
        x = a[0]

    This sets the second compoment of ``a`` to 4:
    ::
        a[1] = 4

    TODO: add descriptions about ``a(i, j)``


Methods
+++++++

.. function:: a.norm(eps = 0)

    :parameter a: (Vector)
    :parameter eps: (optional, scalar) usually be 0, see note below
    :return: (scalar) the magnitude / length / norm of vector

    e.g.:
    ::
        a = ti.Vector([3, 4])
        a.norm() # sqrt(3*3 + 4*4 + 0) = 5
    
    These two are equivalent:
    ::
        ti.sqrt(a.dot(a) + eps)
        a.norm(eps)

.. note::
    Set ``eps = 1e-5`` for example, to safe guards the operator's gradient on zero vectors during differentiable programming.


.. function:: a.dot(b)

    :parameter a: (Vector)
    :parameter b: (Vector)
    :return: (scalar) the dot product / inner product of ``a`` and ``b``

    e.g.:
    ::
        a = ti.Vector([1, 3])
        b = ti.Vector([2, 4])
        a.dot(b) # 1*2 + 3*4 = 14


.. function:: ti.cross(a, b)

    :parameter a: (Vector, 3D)
    :parameter b: (Vector, 3D)
    :return: (Vector, 3D) the cross product of ``a`` and ``b``

    We use right-handed coordinate system, e.g.:
    ::
        a = ti.Vector([1, 2, 3])
        b = ti.Vector([4, 5, 6])
        c = ti.cross(a, b) # [2*6 - 5*3, 4*3 - 1*6, 1*5 - 4*2]


.. function:: ti.outer_product(a, b)

    :parameter a: (Vector)
    :parameter b: (Vector)
    :return: (Matrix) the outer product of ``a`` and ``b``

    e.g.:
    ::
        a = ti.Vector([1, 2, 3])
        b = ti.Vector([4, 5, 6])
        c = ti.outer_product(a, b) # NOTE: c[i, j] = a[i] * b[j]
        # c = [[1*4, 1*5, 1*6], [2*4, 2*5, 2*6], [3*4, 3*5, 3*6]]

.. note::
    This is not the same as `ti.cross`. And thus ``a`` and ``b`` does not have to be 3D vectors.


.. function:: a.cast(dt)

    :parameter a: (Vector)
    :parameter dt: (DataType)
    :return: (Vector) vector with all compoments of ``a`` casted into type ``dt``

    e.g.:
    ::
        # (taichi-scope)
        a = ti.Vector([1.6, 2.3])
        a.cast(ti.i32) # [2, 3]

.. note::
    Vectors are special matrices with only 1 column. In fact, ``ti.Vector`` is just an alias of ``ti.Matrix``.



Matrices
--------

- ``ti.Matrix`` is for small matrices (e.g. `3x3`) only. If you have `64x64` matrices, you should consider using a 2D tensor of scalars.
- ``ti.Vector`` is the same as ``ti.Matrix``, except that it has only one column.
- Differentiate element-wise product ``*`` and matrix product ``@``.
- ``ti.Vector(n, dt=ti.f32)`` or ``ti.Matrix(n, m, dt=ti.f32)``.
- ``ti.transposed(A)`` or simply ``A.T()``
- ``ti.inverse(A)``
- ``ti.Matrix.abs(A)``
- ``ti.tr(A)``
- ``ti.determinant(A, type)``
- ``ti.cross(a, b)``, where ``a`` amd ``b`` are 3D vectors (i.e. ``3x1`` matrices)
- ``A.cast(type)``
- ``R, S = ti.polar_decompose(A, ti.f32)``
- ``U, sigma, V = ti.svd(A, ti.f32)`` (Note that ``sigma`` is a ``3x3`` diagonal matrix)

