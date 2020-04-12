.. _scalar_tensor:

Tensor of scalars
=================


Creation
--------

.. function:: ti.var(dt, shape = None)

    :parameter dt: (DataType) type of the tensor element
    :parameter shape: (optional, scalar or tuple) the shape of tensor

    This creates a *dense* tensor with four ``int32`` as elements:
    ::
        x = ti.var(ti.i32, shape=4)

    This creates a 4x3 *dense* tensor called ``x`` with ``float32`` elements:
    ::
        x = ti.var(ti.f32, shape=(4, 3))

    If shape is ``()`` (empty tuple), then a scalar is created:
    ::
        x = ti.var(ti.f32, shape=())

    Then access it by passing ``None`` as index:
    ::
        x[None] = 2

    If shape is **not provided** or ``None``, then you must manually *place* it afterwards:
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
        x[None] = 233 # ERROR: x not placed!

    .. code-block:: python

        x = ti.var(ti.f32, shape=())
        @ti.kernel
        def func():
            x[None] = 233

        func()
        y = ti.var(ti.f32, shape=())
        # ERROR: cannot create tensor after kernel invocated!

    .. code-block:: python

        x = ti.var(ti.f32, shape=())
        x[None] = 233
        y = ti.var(ti.f32, shape=())
        # ERROR: cannot create tensor after accessed from python-scope!
