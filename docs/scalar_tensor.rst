.. _scalar_tensor:

Scalar fields
=============

**Taichi fields** are used to store data.

Field **elements** could be either a scalar, a vector, or a matrix (see :ref:`matrix`).
In this paragraph, we will only talk about **scalar fields**, whose elements are simply scalars.

Fields can have up to eight **dimensions**.

- A 0D scalar field is simply a single scalar.
- A 1D scalar field is a 1D linear array.
- A 2D scalar field can be used to represent a 2D regular grid of values. For example, a gray-scale image.
- A 3D scalar field can be used for volumetric data.

Fields could be either dense or sparse, see ref:`sparse` for details on sparse
fields. We will only talk about **dense fields** in this paragraph.

.. note::

   We once used the term **tensor** instead of **field**. **Tensor** will no longer be used.


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
