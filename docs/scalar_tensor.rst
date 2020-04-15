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

    Not providing ``shape`` allows you to *place* the tensor as *sparse* tensors, see :ref:`sparse` for more details.


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


Attribute
---------

TODO: WIP
