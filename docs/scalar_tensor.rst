.. _scalar_tensor:

Tensor of scalars
=================


Creation
--------

.. function:: ti.var(dt, shape = None)

    :parameter dt: (DataType) type of the tensor element
    :parameter shape: (optional, scalar or tuple) the shape of tensor

    This creates a 4x3 tensor called ``x`` with ``float32`` elements in GPU memory:
    ::
        x = ti.var(ti.f32, shape=(4, 3))

    If shape is ``()``, then a scalar is created:
    ::
        x = ti.var(ti.f32, shape=())

    If ``shape`` is not provided then you must *place* the variable later yourself:
    ::
        x = ti.var(ti.f32)
        ti.root.dense(ti.ij, (4, 3)).place(x)

    The about two are equivalent.


.. note::

    All variables should be created and *placed* before any kernel invocation or any of them accessed from python-scope

For example:

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
