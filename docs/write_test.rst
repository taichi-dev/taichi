Workflow for writing a Python test
----------------------------------

Normally we write functional tests in Python.

- We use `pytest <https://github.com/pytest-dev/pytest>`_ for our Python test infrastructure.
- Python tests should be added to ``tests/python/test_xxx.py``.

For example, you've just added a utility function ``ti.log10``.
Now you want to write a **test**, to test if it functions properly.

Adding a new test case
**********************

Look into ``tests/python``, see if there's already a file suit for your test.
If not, feel free to create a new file for it :)
So in this case let's create a new file ``tests/python/test_logarithm.py`` for simplicity.

Add a function, the function name **must** be started with ``test_`` so that ``pytest`` could find it. e.g:

.. code-block:: python

    import taichi as ti

    def test_log10():
        pass

Add some simple code make use of our ``ti.log10`` to make sure it works well.
Hint: You may pass/return values to/from Taichi-scope using 0-D fields, i.e. ``r[None]``.

.. code-block:: python

    import taichi as ti

    def test_log10():
        ti.init(arch=ti.cpu)

        r = ti.var(ti.f32, ())

        @ti.kernel
        def foo():
            r[None] = ti.log10(r[None])

        r[None] = 100
        foo()
        assert r[None] == 2

Execute ``ti test logarithm``, and the functions starting with ``test_`` in ``tests/python/test_logarithm.py`` will be executed.

Testing against multiple backends
*********************************

The above method is not good enough, for example, ``ti.init(arch=ti.cpu)``, means that it will only test on the CPU backend.
So do we have to write many tests ``test_log10_cpu``, ``test_log10_cuda``, ... with only the first line different? No worries,
we provide a useful decorator ``@ti.test``:

.. code-block:: python

    import taichi as ti

    # will test against both CPU and CUDA backends
    @ti.test(ti.cpu, ti.cuda)
    def test_log10():
        r = ti.var(ti.f32, ())

        @ti.kernel
        def foo():
            r[None] = ti.log10(r[None])

        r[None] = 100
        foo()
        assert r[None] == 2

And you may test against **all backends** by simply not specifying the argument:

.. code-block:: python

    import taichi as ti

    # will test against all backends available on your end
    @ti.test()
    def test_log10():
        r = ti.var(ti.f32, ())

        @ti.kernel
        def foo():
            r[None] = ti.log10(r[None])

        r[None] = 100
        foo()
        assert r[None] == 2

Cool! Right? But that's still not good enough.

Using ``ti.approx`` for comparison with tolerance
*************************************************

Sometimes the math percison could be poor on some backends like OpenGL, e.g. ``ti.log10(100)``
may return ``2.001`` or ``1.999`` in this case.

To cope with this behavior, we provide ``ti.approx`` which can tolerate such errors on different
backends, for example ``2.001 == ti.approx(2)`` will return ``True`` on the OpenGL backend.

.. code-block:: python

    import taichi as ti

    # will test against all backends available on your end
    @ti.test()
    def test_log10():
        r = ti.var(ti.f32, ())

        @ti.kernel
        def foo():
            r[None] = ti.log10(r[None])

        r[None] = 100
        foo()
        assert r[None] == ti.approx(2)

.. warning::

    Simply using ``pytest.approx`` won't work well here, since it's tolerance won't vary among
    different Taichi backends. It'll be likely to fail on the OpenGL backend.

    ``ti.approx`` also do treatments on boolean types, e.g.: ``2 == ti.approx(True)``.

Great on improving stability! But the test is still not good enough, yet.

Parametrize test inputs
***********************

For example, ``r[None] = 100``, means that it will only test the case of ``ti.log10(100)``.
What if ``ti.log10(10)``? ``ti.log10(1)``?

We may test against different input values using the ``@pytest.mark.parametrize`` decorator:

.. code-block:: python

    import taichi as ti
    import pytest
    import math

    @pytest.mark.parametrize('x', [1, 10, 100])
    @ti.test()
    def test_log10(x):
        r = ti.var(ti.f32, ())

        @ti.kernel
        def foo():
            r[None] = ti.log10(r[None])

        r[None] = x
        foo()
        assert r[None] == math.log10(x)

Use a comma-separated list for multiple input values:

.. code-block:: python

    import taichi as ti
    import pytest
    import math

    @pytest.mark.parametrize('x,y', [(1, 2), (1, 3), (2, 1)])
    @ti.test()
    def test_atan2(x, y):
        r = ti.var(ti.f32, ())
        s = ti.var(ti.f32, ())

        @ti.kernel
        def foo():
            r[None] = ti.atan2(r[None])

        r[None] = x
        s[None] = y
        foo()
        assert r[None] == math.atan2(x, y)

Use two separate ``parametrize`` to test **all combinations** of input arguments:

.. code-block:: python

    import taichi as ti
    import pytest
    import math

    @pytest.mark.parametrize('x', [1, 2])
    @pytest.mark.parametrize('y', [1, 2])
    # same as:  .parametrize('x,y', [(1, 1), (1, 2), (2, 1), (2, 2)])
    @ti.test()
    def test_atan2(x, y):
        r = ti.var(ti.f32, ())
        s = ti.var(ti.f32, ())

        @ti.kernel
        def foo():
            r[None] = ti.atan2(r[None])

        r[None] = x
        s[None] = y
        foo()
        assert r[None] == math.atan2(x, y)

Specifying ``ti.init`` configurations
*************************************

You may specify keyword arguments to ``ti.init()`` in ``ti.test()``, e.g.:

.. code-block:: python

    @ti.test(ti.cpu, debug=True, log_level=ti.TRACE)
    def test_debugging_utils():
        # ... (some tests have to be done in debug mode)

is the same as:

.. code-block:: python

    def test_debugging_utils():
        ti.init(arch=ti.cpu, debug=True, log_level=ti.TRACE)
        # ... (some tests have to be done in debug mode)

Exclude some backends from test
*******************************

Sometimes some backends are not capable of specific tests, we have to exclude them from test:

.. code-block:: python

    # Run this test on all backends except for OpenGL
    @ti.test(excludes=[ti.opengl])
    def test_sparse_field():
        # ... (some tests that requires sparse feature which is not supported by OpenGL)

You may also use the ``extensions`` keyword to exclude backends without specific feature:

.. code-block:: python

    # Run this test on all backends except for OpenGL
    @ti.test(extensions=[ti.extension.sparse])
    def test_sparse_field():
        # ... (some tests that requires sparse feature which is not supported by OpenGL)
