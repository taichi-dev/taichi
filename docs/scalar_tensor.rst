.. _scalar_tensor:

Scalar fields
=============

**Taichi fields** are used to store data.

Field **elements** could be either scalar, vector or matrix (see :ref:`matrix`).
We'll only talk about **scalar fields** in this chapter.

Field **dimensions** could be arbitrary, including but not limited to 0D, 1D, 2D...

- 0D fields are simply scalars, they only contain one scalar value.
- 1D fields are simply 1D arrays for people from C/C++.
- 2D fields are used to save images, or matrices for people from Matlab.
- 3D fields can save the properties of a space volume, e.g., a temperature field.

Fields can have different shapes:

- The shape of a 0D field is always a **empty tuple**, e.g. ``shape=()``.
- The shape of a 1D field is **the length of array**, e.g. ``shape=512``.
- The shape of a 2D field is **the resolution of a image**, e.g. ``shape=(1024, 768)``.

Fields could be either dense or sparse, see ref:`sparse` for details on sparse
fields. We'll only talk about **dense fields** in this chapter.

.. note::

   We once used the term **tensor** instead of **field**, please caution about
   these old documentations using *tensor*. We will soon deprecate the term
   *tensor* in the future for clarity.


Declaration
-----------

.. function:: ti.field(dtype, shape = None, offset = None)

    :parameter dtype: (DataType) type of the field element
    :parameter shape: (optional, scalar or tuple) the shape of field
    :parameter offset: (optional, scalar or tuple) see :ref:`offset`

    For example, this creates a *dense* field with four ``int32`` as elements:
    ::

        x = ti.field(ti.i32, shape=4)

    This creates a 4x3 *dense* field with ``float32`` elements:
    ::

        x = ti.field(ti.f32, shape=(4, 3))

    If shape is ``()`` (empty tuple), then a 0-D field (scalar) is created:
    ::

        x = ti.field(ti.f32, shape=())

    Then access it by passing ``None`` as index:
    ::

        x[None] = 2

    If shape is **not provided** or ``None``, the user must manually ``place`` it afterwards:
    ::

        x = ti.field(ti.f32)
        ti.root.dense(ti.ij, (4, 3)).place(x)
        # equivalent to: x = ti.field(ti.f32, shape=(4, 3))

.. note::

    Not providing ``shape`` allows you to *place* the field in a layout other than the default *dense*, see :ref:`layout` for more details.


.. warning::

    All variables should be created and placed before any kernel invocation or any of them accessed from python-scope. For example:

    .. code-block:: python

        x = ti.field(ti.f32)
        x[None] = 1 # ERROR: x not placed!

    .. code-block:: python

        x = ti.field(ti.f32, shape=())
        @ti.kernel
        def func():
            x[None] = 1

        func()
        y = ti.field(ti.f32, shape=())
        # ERROR: cannot create fields after kernel invocation!

    .. code-block:: python

        x = ti.field(ti.f32, shape=())
        x[None] = 1
        y = ti.field(ti.f32, shape=())
        # ERROR: cannot create fields after any field accesses from the Python-scope!


Accessing components
--------------------

You can access an element of the Taichi field by an index or indices.

.. attribute:: a[p, q, ...]

    :parameter a: (ti.field) the sclar field
    :parameter p: (scalar) index of the first field dimension
    :parameter q: (scalar) index of the second field dimension
    :return: (scalar) the element at ``[p, q, ...]``

    This extracts the element value at index ``[3, 4]`` of field ``a``:
    ::

        x = a[3, 4]

    This sets the element value at index ``2`` of 1D field ``b`` to ``5``:
    ::

        b[2] = 5

    .. note ::

        In Python, x[(exp1, exp2, ..., expN)] is equivalent to x[exp1, exp2, ..., expN]; the latter is just syntactic sugar for the former.

    .. note ::

        The returned value can also be ``Vector`` / ``Matrix`` if ``a`` is a vector/matrix field, see :ref:`vector` for more details.


Meta data
---------


.. attribute:: a.shape

    :parameter a: (ti.field) the field
    :return: (tuple) the shape of field ``a``

    ::

        x = ti.field(ti.i32, (6, 5))
        x.shape  # (6, 5)

        y = ti.field(ti.i32, 6)
        y.shape  # (6,)

        z = ti.field(ti.i32, ())
        z.shape  # ()


.. attribute:: a.dtype

    :parameter a: (ti.field) the field
    :return: (DataType) the data type of ``a``

    ::

        x = ti.field(ti.i32, (2, 3))
        x.dtype  # ti.i32


.. function:: a.parent(n = 1)

    :parameter a: (ti.field) the field
    :parameter n: (optional, scalar) the number of parent steps, i.e. ``n=1`` for parent, ``n=2`` grandparent, etc.
    :return: (SNode) the parent of ``a``'s containing SNode

    ::

        x = ti.field(ti.i32)
        y = ti.field(ti.i32)
        blk1 = ti.root.dense(ti.ij, (6, 5))
        blk2 = blk1.dense(ti.ij, (3, 2))
        blk1.place(x)
        blk2.place(y)

        x.parent()   # blk1
        y.parent()   # blk2
        y.parent(2)  # blk1

    See :ref:`snode` for more details.
