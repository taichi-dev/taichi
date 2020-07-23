.. _vector:

Vectors
=======

A vector in Taichi can have two forms:

  - as a temporary local variable. An ``n`` component vector consists of ``n`` scalar values.
  - as an element of a global tensor. In this case, the tensor is an N-dimensional array of ``n`` component vectors.

In fact, ``Vector`` is simply an alias of ``Matrix``, just with ``m = 1``. See :ref:`matrix` and :ref:`tensor` for more details.

Declaration
-----------

As global tensors of vectors
++++++++++++++++++++++++++++

.. function:: ti.Vector.var(n, dt, shape = None, offset = None)

    :parameter n: (scalar) the number of components in the vector
    :parameter dt: (DataType) data type of the components
    :parameter shape: (optional, scalar or tuple) shape the tensor of vectors, see :ref:`tensor`
    :parameter offset: (optional, scalar or tuple) see :ref:`offset`

    For example, this creates a 5x4 tensor of 3 component vectors:
    ::

        # Python-scope
        a = ti.Vector.var(3, dt=ti.f32, shape=(5, 4))

.. note::

    In Python-scope, ``ti.var`` declares :ref:`scalar_tensor`, while ``ti.Vector`` declares tensors of vectors.


As a temporary local variable
+++++++++++++++++++++++++++++

.. function:: ti.Vector([x, y, ...])

    :parameter x: (scalar) the first component of the vector
    :parameter y: (scalar) the second component of the vector

    For example, this creates a 3D vector with components (2, 3, 4):
    ::

        # Taichi-scope
        a = ti.Vector([2, 3, 4])


Accessing components
--------------------

As global tensors of vectors
++++++++++++++++++++++++++++
.. attribute:: a[p, q, ...][i]

    :parameter a: (tensor of Vector) the vector
    :parameter p: (scalar) index of the first tensor dimension
    :parameter q: (scalar) index of the second tensor dimension
    :parameter i: (scalar) index of the vector component

    This extracts the first component of vector ``a[6, 3]``:
    ::

        x = a[6, 3][0]

        # or
        vec = a[6, 3]
        x = vec[0]

.. note::

    **Always** use two pairs of square brackets to access scalar elements from tensors of vectors.

     - The indices in the first pair of brackets locate the vector inside the tensor of vectors;
     - The indices in the second pair of brackets locate the scalar element inside the vector.

    For 0-D tensors of vectors, indices in the first pair of brackets should be ``[None]``.



As a temporary local variable
+++++++++++++++++++++++++++++

.. attribute:: a[i]

    :parameter a: (Vector) the vector
    :parameter i: (scalar) index of the component

    For example, this extracts the first component of vector ``a``:
    ::

        x = a[0]

    This sets the second component of ``a`` to 4:
    ::

        a[1] = 4

    TODO: add descriptions about ``a(i, j)``

Methods
-------

.. function:: a.norm(eps = 0)

    :parameter a: (Vector)
    :parameter eps: (optional, scalar) a safe-guard value for ``sqrt``, usually 0. See the note below.
    :return: (scalar) the magnitude / length / norm of vector

    For example,
    ::

        a = ti.Vector([3, 4])
        a.norm() # sqrt(3*3 + 4*4 + 0) = 5

    ``a.norm(eps)`` is equivalent to ``ti.sqrt(a.dot(a) + eps)``

.. note::
    Set ``eps = 1e-5`` for example, to safe guard the operator's gradient on zero vectors during differentiable programming.


.. function:: a.norm_sqr()

    :parameter a: (Vector)
    :return: (scalar) the square of the magnitude / length / norm of vector

    For example,
    ::

        a = ti.Vector([3, 4])
        a.norm_sqr() # 3*3 + 4*4 = 25

    ``a.norm_sqr()`` is equivalent to ``a.dot(a)``


.. function:: a.normalized()

    :parameter a: (Vector)
    :return: (Vector) the normalized / unit vector of ``a``

    For example,
    ::

        a = ti.Vector([3, 4])
        a.normalized() # [3 / 5, 4 / 5]

    ``a.normalized()`` is equivalent to ``a / a.norm()``.


.. function:: a.dot(b)

    :parameter a: (Vector)
    :parameter b: (Vector)
    :return: (scalar) the dot (inner) product of ``a`` and ``b``

    E.g.,
    ::

        a = ti.Vector([1, 3])
        b = ti.Vector([2, 4])
        a.dot(b) # 1*2 + 3*4 = 14


.. function:: a.cross(b)

    :parameter a: (Vector, 2 or 3 components)
    :parameter b: (Vector of the same size as a)
    :return: (scalar (for 2D inputs), or 3D Vector (for 3D inputs)) the cross product of ``a`` and ``b``

    We use a right-handed coordinate system. E.g.,
    ::

        a = ti.Vector([1, 2, 3])
        b = ti.Vector([4, 5, 6])
        c = ti.cross(a, b)
        # c = [2*6 - 5*3, 4*3 - 1*6, 1*5 - 4*2] = [-3, 6, -3]

        p = ti.Vector([1, 2])
        q = ti.Vector([4, 5])
        r = ti.cross(a, b)
        # r = 1*5 - 4*2 = -3


.. function:: a.outer_product(b)

    :parameter a: (Vector)
    :parameter b: (Vector)
    :return: (Matrix) the outer product of ``a`` and ``b``

    E.g.,
    ::

        a = ti.Vector([1, 2])
        b = ti.Vector([4, 5, 6])
        c = ti.outer_product(a, b) # NOTE: c[i, j] = a[i] * b[j]
        # c = [[1*4, 1*5, 1*6], [2*4, 2*5, 2*6]]

.. note::
    This have no common with ``ti.cross``. ``a`` and ``b`` do not have to be 3 or 2 component vectors.


.. function:: a.cast(dt)

    :parameter a: (Vector)
    :parameter dt: (DataType)
    :return: (Vector) vector with all components of ``a`` casted into type ``dt``

    E.g.,
    ::

        # Taichi-scope
        a = ti.Vector([1.6, 2.3])
        a.cast(ti.i32) # [2, 3]

.. note::
    Vectors are special matrices with only 1 column. In fact, ``ti.Vector`` is just an alias of ``ti.Matrix``.


Metadata
--------

.. attribute:: a.n

   :parameter a: (Vector or tensor of Vector)
   :return: (scalar) return the dimensionality of vector ``a``

    E.g.,
    ::

        # Taichi-scope
        a = ti.Vector([1, 2, 3])
        a.n  # 3

    ::
        # Python-scope
        a = ti.Vector.var(3, dt=ti.f32, shape=())
        a.n  # 3

TODO: add element wise operations docs
