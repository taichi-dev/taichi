.. _scalar_tensor:

Tensors of scalars
==================


Declaration
-----------

.. function:: ti.var(dt, shape = None)

    :parameter dt: (DataType) type of the tensor element
    :parameter shape: (optional, scalar or tuple) the shape of tensor

    For example, this creates a *dense* tensor with four ``int32`` as elements:
    ::

        x = ti.var(ti.i32, shape=4)

    This creates a 4x3 *dense* tensor with ``float32`` elements:
    ::

        x = ti.var(ti.f32, shape=(4, 3))

    If shape is ``()`` (empty tuple), then a 0-D tensor (scalar) is created:
    ::

        x = ti.var(ti.f32, shape=())

    Then access it by passing ``None`` as index:
    ::

        x[None] = 2

    If shape is **not provided** or ``None``, the user must manually ``place`` it afterwards:
    ::

        x = ti.var(ti.f32)
        ti.root.dense(ti.ij, (4, 3)).place(x)
        # equivalent to: x = ti.var(ti.f32, shape=(4, 3))

.. note::

    Not providing ``shape`` allows you to *place* the tensor in a layout other than the default *dense*, see :ref:`layout` for more details.


.. warning::

    All variables should be created and placed before any kernel invocation or any of them accessed from python-scope. For example:

    .. code-block:: python

        x = ti.var(ti.f32)
        x[None] = 1 # ERROR: x not placed!

    .. code-block:: python

        x = ti.var(ti.f32, shape=())
        @ti.kernel
        def func():
            x[None] = 1

        func()
        y = ti.var(ti.f32, shape=())
        # ERROR: cannot create tensor after kernel invocation!

    .. code-block:: python

        x = ti.var(ti.f32, shape=())
        x[None] = 1
        y = ti.var(ti.f32, shape=())
        # ERROR: cannot create tensor after any tensor accesses from the Python-scope!


Accessing components
--------------------

You can access an element of the Taichi tensor by an index or indices.

.. attribute:: a[p, q, ...]

    :parameter a: (Tensor) the tensor of scalars
    :parameter p: (scalar) index of the first tensor dimension
    :parameter q: (scalar) index of the second tensor dimension
    :return: (scalar) the element at ``[p, q, ...]``

    This extracts the element value at index ``[3, 4]`` of tensor ``a``:
    ::

        x = a[3, 4]

    This sets the element value at index ``2`` of 1D tensor ``b`` to ``5``:
    ::

        a[2] = 2

    .. note ::

        In Python, x[(exp1, exp2, ..., expN)] is equivalent to x[exp1, exp2, ..., expN]; the latter is just syntactic sugar for the former.

    .. note ::

        The returned value can also be ``Vector`` / ``Matrix`` if ``a`` is a tensor of vector / matrix, see :ref:`vector` for more details.


Meta data
---------

.. function:: a.dim()

    :parameter a: (Tensor) the tensor
    :return: (scalar) the length of ``a``

    ::

        x = ti.var(ti.i32, (6, 5))
        x.dim()  # 2

        y = ti.var(ti.i32, 6)
        y.dim()  # 1

        z = ti.var(ti.i32, ())
        z.dim()  # 0


.. function:: a.shape()

    :parameter a: (Tensor) the tensor
    :return: (tuple) the shape of tensor ``a``

    ::

        x = ti.var(ti.i32, (6, 5))
        x.shape()  # (6, 5)

        y = ti.var(ti.i32, 6)
        y.shape()  # (6,)

        z = ti.var(ti.i32, ())
        z.shape()  # ()


.. function:: a.data_type()

    :parameter a: (Tensor) the tensor
    :return: (DataType) the data type of ``a``

    ::

        x = ti.var(ti.i32, (2, 3))
        x.data_type()  # ti.i32


.. function:: a.parent(n = 1)

    :parameter a: (Tensor) the tensor
    :parameter n: (optional, scalar) the number of parent steps, i.e. ``n=1`` for parent, ``n=2`` grandparent, etc...
    :return: (SNode) the parent of ``a``'s containing SNode

    ::
        x = ti.var(ti.i32)
        y = ti.var(ti.i32)
        blk1 = ti.root.dense(ti.ij, (6, 5))
        blk2 = blk1.dense(ti.ij, (3, 2))
        blk1.place(x)
        blk2.place(y)

        x.parent()   # blk1
        y.parent()   # blk2
        y.parent(2)  # blk1

    See :ref:`snode` for more details.
