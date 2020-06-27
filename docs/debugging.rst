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


Tips for debugging
------------------

Debugging a Taichi program can be hard even with the builtin tools above.
Taichi developers are currently devoting themselves in improving error messages and warnings
to help user find potential BUGs in their programs.

Here we collected some common BUGs that one might encounter with a Taichi program:

Static typing system
++++++++++++++++++++

Taichi pertend that it's a dynamical-typed language like Python, but it's actually a
statically-typed language which will be translated into high performance CPU/GPU instructions.

So the code behavior in Taichi-scope is actually very different from Python-scope!

Type of a variable is simply **determined at its first initialization and never changes later**.

Although static-type provides better performance and simplicity, but may leads to BUGs if
users not distinguished Taichi-scope from Python-scope, e.g.:

.. code-block:: python

    @ti.kernel
    def buggy():
        ret = 0  # 0 is a integer, so `ret` is typed as int32
        for i in range(3):
            ret += 0.1 * i  # i32 += f32, the result is still stored in int32!
        print(ret)  # will shows 0

    buggy()

The codes above shows a common BUG due to the limitation of the static-type system.
The Taichi compiler should shows a warning like:

.. code-block:: none

    [W 06/27/20 21:43:51.853] [type_check.cpp:visit@66] [$19] Atomic add (float32 to int32) may lose precision.

This means that it can not store a float32 result to int32.
The solution is to type ``ret`` as float32 at the first place:

.. code-block:: python

    @ti.kernel
    def not_buggy():
        ret = 0.0  # 0 is a floating point number, so `ret` is typed as float32
        for i in range(3):
            ret += 0.1 * i  # f32 += f32, OK!!
        print(ret)  # will shows 0.6

    not_buggy()


`@archibate <https://github.com/archibate>`_'s personal suggestion to prevent issues like this:

* Recall the ``float ret = 0;`` in C/C++, always use ``ret = float(0)`` on **initialization**,
  and ``ret = int(0)`` for integers. So that you are always clear of what type every variable.

Advanced Optimization
+++++++++++++++++++++

Taichi has a advanced optimization engine to make your Taichi kernel to be as fast as it could.
But like the ``gcc -O3`` does, sometimes advanced optimization can leads to BUGs as it tried
too hard, including runtime errors like:

```RuntimeError: [verify.cpp:basic_verify@40] stmt 8 cannot have operand 7.```

You may use ``ti.core.toggle_advance_optimization(False)`` to turn off advanced
optimization and see if the issue still exists:

.. code-block:: python

    import taichi as ti

    ti.init()
    ti.core.toggle_advance_optimization()

    ...

If that fixed the issue, please report this BUG on `GitHub <https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md>`_ to help us improve, if you would like to.
