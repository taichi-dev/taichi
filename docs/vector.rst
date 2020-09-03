.. _vector:

Vectors
=======

A vector in Taichi can have two forms:

  - as a temporary local variable. An ``n`` component vector consists of ``n`` scalar values.
  - as an element of a global field. In this case, the field is an N-dimensional array of ``n`` component vectors.

In fact, ``Vector`` is simply an alias of ``Matrix``, just with ``m = 1``. See :ref:`matrix` and :ref:`tensor` for more details.

Declaration
-----------

As global vector fields
+++++++++++++++++++++++

.. function:: ti.Vector.field(n, dtype, shape = None, offset = None)

    :parameter n: (scalar) the number of components in the vector
    :parameter dtype: (DataType) data type of the components
    :parameter shape: (optional, scalar or tuple) shape of the vector field, see :ref:`tensor`
    :parameter offset: (optional, scalar or tuple) see :ref:`offset`

    For example, this creates a 3-D vector field of the shape of ``5x4``:
    ::

        # Python-scope
        a = ti.Vector.field(3, dtype=ti.f32, shape=(5, 4))

.. note::

    In Python-scope, ``ti.field`` declares a scalar field (:ref:`scalar_tensor`), while ``ti.Vector.field`` declares a vector field.


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

As global vector fields
+++++++++++++++++++++++
.. attribute:: a[p, q, ...][i]

    :parameter a: (ti.Vector.field) the vector
    :parameter p: (scalar) index of the first field dimension
    :parameter q: (scalar) index of the second field dimension
    :parameter i: (scalar) index of the vector component

    This extracts the first component of vector ``a[6, 3]``:
    ::

        x = a[6, 3][0]

        # or
        vec = a[6, 3]
        x = vec[0]

.. note::

    **Always** use two pairs of square brackets to access scalar elements from vector fields.

     - The indices in the first pair of brackets locate the vector inside the vector fields;
     - The indices in the second pair of brackets locate the scalar element inside the vector.

    For 0-D vector fields, indices in the first pair of brackets should be ``[None]``.



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


XYZW vector component accessors
+++++++++++++++++++++++++++++++

We also provide four handy accessors for the first four vector components:

.. attribute:: a.x

   Same as ``a[0]``.

.. attribute:: a.y

   Same as ``a[1]``.

.. attribute:: a.z

   Same as ``a[2]``.

.. attribute:: a.w

   Same as ``a[3]``.

.. note::

   XYZW accessors can be used for both reading and writing::

      v = ti.Vector([2, 3, 4])
      print(v.x)  # 2
      print(v.y)  # 3
      print(v.z)  # 4
      v.y = 8
      print(v.y)  # 8

   XYZW accessors can be used in both Taichi-scope and Python-scope.

   XYZW accessors don't work for ``ti.Matrix``.

   For GLSL-alike shuffling accessors, consider using `taichi_glsl <https://taichi-glsl.readthedocs.io>`_::

        import taichi_glsl as tl

        v = tl.vec(2, 3, 4)
        print(v.xy)  # [2 3]
        print(v._xYzX_z)  # [0 2 -3 4 -2 0 4]

Methods
-------

.. function:: a.norm(eps = 0)

    :parameter a: (ti.Vector)
    :parameter eps: (optional, scalar) a safe-guard value for ``sqrt``, usually 0. See the note below.
    :return: (scalar) the magnitude / length / norm of vector

    For example,
    ::

        a = ti.Vector([3, 4])
        a.norm() # sqrt(3*3 + 4*4 + 0) = 5

    ``a.norm(eps)`` is equivalent to ``ti.sqrt(a.dot(a) + eps)``

.. note::
    To safeguard the operator's gradient on zero vectors during differentiable programming, set ``eps`` to a small, positive value such as ``1e-5``.


.. function:: a.norm_sqr()

    :parameter a: (ti.Vector)
    :return: (scalar) the square of the magnitude / length / norm of vector

    For example,
    ::

        a = ti.Vector([3, 4])
        a.norm_sqr() # 3*3 + 4*4 = 25

    ``a.norm_sqr()`` is equivalent to ``a.dot(a)``


.. function:: a.normalized()

    :parameter a: (ti.Vector)
    :return: (ti.Vector) the normalized / unit vector of ``a``

    For example,
    ::

        a = ti.Vector([3, 4])
        a.normalized() # [3 / 5, 4 / 5]

    ``a.normalized()`` is equivalent to ``a / a.norm()``.


.. function:: a.dot(b)

    :parameter a: (ti.Vector)
    :parameter b: (ti.Vector)
    :return: (scalar) the dot (inner) product of ``a`` and ``b``

    E.g.,
    ::

        a = ti.Vector([1, 3])
        b = ti.Vector([2, 4])
        a.dot(b) # 1*2 + 3*4 = 14


.. function:: a.cross(b)

    :parameter a: (ti.Vector, 2 or 3 components)
    :parameter b: (ti.Vector of the same size as a)
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

    :parameter a: (ti.Vector)
    :parameter b: (ti.Vector)
    :return: (ti.Matrix) the outer product of ``a`` and ``b``

    E.g.,
    ::

        a = ti.Vector([1, 2])
        b = ti.Vector([4, 5, 6])
        c = ti.outer_product(a, b) # NOTE: c[i, j] = a[i] * b[j]
        # c = [[1*4, 1*5, 1*6], [2*4, 2*5, 2*6]]

.. note::
    The outer product should not be confused with the cross product (``ti.cross``). For example, ``a`` and ``b`` do not have to be 2- or 3-component vectors for this function.


.. function:: a.cast(dt)

    :parameter a: (ti.Vector)
    :parameter dt: (DataType)
    :return: (ti.Vector) vector with all components of ``a`` casted into type ``dt``

    E.g.,
    ::

        # Taichi-scope
        a = ti.Vector([1.6, 2.3])
        a.cast(ti.i32) # [2, 3]

    See :ref:`type` for more details.

.. note::

    Vectors are special matrices with only 1 column. In fact, ``ti.Vector`` is just an alias of ``ti.Matrix``.


Metadata
--------

.. attribute:: a.n

   :parameter a: (ti.Vector or ti.Vector.field)
   :return: (scalar) return the dimensionality of vector ``a``

   E.g.,
   ::

        # Taichi-scope
        a = ti.Vector([1, 2, 3])
        a.n  # 3

   ::

        # Python-scope
        a = ti.Vector.field(3, dtype=ti.f32, shape=(4, 5))
        a.n  # 3

    See :ref:`meta` for more details.

.. note::

   When used as a global vector field, it will additionally contain all the
   metadata that a scalar field would have, E.g.::

        # Python-scope
        a = ti.Vector.field(3, dtype=ti.f32, shape=(4, 5))
        a.shape  # (4, 5)
        a.dtype  # ti.f32


Element-wise operations (WIP)
-----------------------------

TODO: add element wise operations docs
