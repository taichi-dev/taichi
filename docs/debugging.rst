Debugging
=========

Debugging a massive-parallelized CPU/GPU program is not easy, so Taichi provides some
builtin utilities that hopefully could help you debug your Taichi program.

Runtime ``print`` in kernel
---------------------------

Debug your program with ``print()`` in Taichi-scope. For example:

.. code-block:: python

    @ti.kernel
    def inside_taichi_scope():
        x = 233
        print('hello', x)
        #=> hello 233

        print('hello', x * 2 + 200)
        #=> hello 666

        print('hello', x, sep='')
        #=> hello233

        print('hello', x, sep='', end='')
        print('world', x, sep='')
        #=> hello233world233

        m = ti.Matrix([[2, 3, 4], [5, 6, 7]])
        print('m =', m)
        #=> m = [[2, 3, 4], [5, 6, 7]]

        v = ti.Vector([3, 4])
        print('v =', v)
        #=> v = [3, 4]

For now, Taichi-scope ``print`` support string, scalar, vector, matrix expressions as argument.
``print`` in Taichi-scope maybe a little different from the ones in Python-scope, see below.

.. warning::

    For the **CPU, CUDA and Metal backend**, ``print`` will not work in Graphical Python Shells
    including IDLE and Jupyter notebook, since they print the result to console, instead of GUI.
    Taichi developers are trying to solve this now. Use **OpenGL backend** if you wish to
    use ``print`` in IDLE / Jupyter.

.. note::

    For the **OpenGL and CUDA backend**, the printed result won't shows up until ``ti.sync()``:

    .. code-block:: python

        import taichi as ti
        ti.init(arch=ti.cuda)

        @ti.kernel
        def kern():
            print('inside kernel')

        print('before kernel')
        kern()
        print('after kernel')
        ti.sync()
        print('after sync')

    obtains:

    .. code-block:: none

        before kernel
        after kernel
        inside kernel
        after

    Also note that host access or program end will also implicitly invoke for ``ti.sync()``.


Compile-time ``ti.static_print``
--------------------------------

Sometimes it's also useful to print Python-scope objects / constants like data type, SNode.
So, similar to ``ti.static`` we provide ``ti.static_print`` to print compile-time constants.
It behave as same as Python-scope ``print`` does, just being embbed into Taichi kernel.

.. code-block:: python

    x = ti.var(ti.f32, (2, 3))

    @ti.kernel
    def inside_taichi_scope():
        ti.static_print(x.shape)
        # => (2, 3)
        ti.static_print(x.data_type())
        # => DataType.float32
        for i in range(4):
                ti.static_print(i.data_type())
                # => DataType.int32
                # will only print once

Unlike ``print``, ``ti.static_print`` will only print the expression once at compile-time. And
therefore has no runtime cost.


Runtime ``assert`` in kernel
----------------------------

We may use ``assert`` statement in Taichi-scope. When assertion condition failed, a
``RuntimeError`` will be raised to indicate error.

To make ``assert`` work, first make sure you are using the **CPU backend**.
For performance reason, ``assert`` is only work when ``debug`` mode is on, For example:

.. code-block:: python

    ti.init(arch=ti.cpu, debug=True)

    x = ti.var(ti.f32, 128)

    @ti.kernel
    def do_sqrt_all():
        for i in x:
            assert x[i] >= 0
            x[i] = ti.sqrt(x)


When your debugging work is done, set ``debug=False``. Now ``assert`` will be simply ignored
therefore no runtime overhead, making your program easy to debug without losing performance.


Compile-time ``ti.static_assert``
---------------------------------

Like ``ti.static_print``, we also provide a static version for ``assert``:
``ti.static_assert``, it can be useful to make assertion on data type / dimention / shape.
It works no matter wheater ``debug=True`` is specified. When assertion failure, it will
raise ``AssertionError`` as a Python-scope ``assert`` does.

For example:

.. code-block:: python

    @ti.func
    def is_odd(x: ti.template()):
        ti.static_assert(x.data_type() == ti.i32, "is_odd() is only supported for i32")
        return x % 2 == 1
