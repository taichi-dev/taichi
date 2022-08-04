---
sidebar_position: 3
---

# Write a Python test

Normally we write functional tests in Python.

- We use [pytest](https://github.com/pytest-dev/pytest) for our Python
  test infrastructure.
- Python tests should be added to `tests/python/test_xxx.py`.

For example, you've just added a utility function `ti.log10`. Now you
want to write a **test** to ensure that it functions properly.

## Add a new test case

Look into `tests/python`, see if there is already a file suitable for your
test. If not, create a new file for it. In this case,
let's create a new file `tests/python/test_logarithm.py` for
simplicity.

Add a function, the function name **must** start with `test_` so
that `pytest` could find it. e.g:

```python {3}
import taichi as ti

def test_log10():
    pass
```

Add some tests that make use of `ti.log10` to ensure it works
well. Hint: You may pass/return values to/from Taichi-scope using 0-D
fields, i.e. `r[None]`.

```python
import taichi as ti

def test_log10():
    ti.init(arch=ti.cpu)

    r = ti.field(ti.f32, ())

    @ti.kernel
    def foo():
        r[None] = ti.log10(r[None])

    r[None] = 100
    foo()
    assert r[None] == 2
```

Execute `python tests/run_tests.py logarithm`, and the functions starting with `test_` in
`tests/python/test_logarithm.py` will be executed.

## Test against multiple backends

The line `ti.init(arch=ti.cpu)` in the test above means that it will only test on the CPU backend. In order to test against multiple backends, please use the `@ti.test` decorator, as illustrated below:

```python
import taichi as ti

# will test against both CPU and CUDA backends
@ti.test(arch=[ti.cpu, ti.cuda])
def test_log10():
    r = ti.field(ti.f32, ())

    @ti.kernel
    def foo():
        r[None] = ti.log10(r[None])

    r[None] = 100
    foo()
    assert r[None] == 2
```

In this case, Taichi will execute this test case on both `ti.cpu` and `ti.cuda` backends.

And you may test against all available backends (depends on your system and building environment) by simply not specifying the
argument:

```python
import taichi as ti

# will test against all backends available on your end
@ti.test()
def test_log10():
    r = ti.field(ti.f32, ())

    @ti.kernel
    def foo():
        r[None] = ti.log10(r[None])

    r[None] = 100
    foo()
    assert r[None] == 2
```

## Use `ti.approx` for comparison with tolerance

Sometimes the precision of math operations could be limited on certain backends such as OpenGL,
e.g., `ti.log10(100)` may return `2.000001` or `1.999999` in this case.

Adding tolerance with `ti.approx` can be helpful to mitigate
such errors on different backends, for example `2.001 == ti.approx(2)`
will return `True` on the OpenGL backend.

```python
import taichi as ti

# will test against all backends available on your end
@ti.test()
def test_log10():
    r = ti.field(ti.f32, ())

    @ti.kernel
    def foo():
        r[None] = ti.log10(r[None])

    r[None] = 100
    foo()
    assert r[None] == ti.approx(2)
```

:::caution
Simply using `pytest.approx` won't work well here, since it's
tolerance won't vary among different Taichi backends. It'll likely
fail on the OpenGL backend.

`ti.approx` also correctly treats boolean types, e.g.:
`2 == ti.approx(True)`.
:::

## Parametrize test inputs

In the test above, `r[None] = 100` means that it will only test that `ti.log10` works correctly for the input `100`. In order to test against different input values, you may use the `@pytest.mark.parametrize` decorator:

```python {5}
import taichi as ti
import pytest
import math

@pytest.mark.parametrize('x', [1, 10, 100])
@ti.test()
def test_log10(x):
    r = ti.field(ti.f32, ())

    @ti.kernel
    def foo():
        r[None] = ti.log10(r[None])

    r[None] = x
    foo()
    assert r[None] == math.log10(x)
```

Use a comma-separated list for multiple input values:

```python
import taichi as ti
import pytest
import math

@pytest.mark.parametrize('x,y', [(1, 2), (1, 3), (2, 1)])
@ti.test()
def test_atan2(x, y):
    r = ti.field(ti.f32, ())
    s = ti.field(ti.f32, ())

    @ti.kernel
    def foo():
        r[None] = ti.atan2(r[None])

    r[None] = x
    s[None] = y
    foo()
    assert r[None] == math.atan2(x, y)
```

Use two separate `parametrize` to test **all combinations** of input
arguments:

```python {5-6}
import taichi as ti
import pytest
import math

@pytest.mark.parametrize('x', [1, 2])
@pytest.mark.parametrize('y', [1, 2])
# same as:  .parametrize('x,y', [(1, 1), (1, 2), (2, 1), (2, 2)])
@ti.test()
def test_atan2(x, y):
    r = ti.field(ti.f32, ())
    s = ti.field(ti.f32, ())

    @ti.kernel
    def foo():
        r[None] = ti.atan2(r[None])

    r[None] = x
    s[None] = y
    foo()
    assert r[None] == math.atan2(x, y)
```

## Specify `ti.init` configurations

You may specify keyword arguments to `ti.init()` in `ti.test()`, e.g.:

```python {1}
@ti.test(ti.cpu, debug=True, log_level=ti.TRACE)
def test_debugging_utils():
    # ... (some tests have to be done in debug mode)
```

is the same as:

```python {2}
def test_debugging_utils():
    ti.init(arch=ti.cpu, debug=True, log_level=ti.TRACE)
    # ... (some tests have to be done in debug mode)
```

## Exclude some backends from test

Some backends are not capable of executing certain tests, you may have to
exclude them from the test in order to move forward:

```python
# Run this test on all backends except for OpenGL
@ti.test(exclude=[ti.opengl])
def test_sparse_field():
    # ... (some tests that requires sparse feature which is not supported by OpenGL)
```

You may also use the `extensions` keyword to exclude backends without
a specific feature:

```python
# Run this test on all backends except for OpenGL
@ti.test(extensions=[ti.extension.sparse])
def test_sparse_field():
    # ... (some tests that requires sparse feature which is not supported by OpenGL)
```

## Require extension in tests

If a test case depends on some extensions, you should add an argument `require` in the `@ti.test` decorator.

```python {1}
@ti.test(require=ti.extension.sparse)
def test_struct_for_pointer_block():
    n = 16
    block_size = 8

    f = ti.field(dtype=ti.f32)

    block = ti.root.pointer(ti.ijk, n // block_size)
    block.dense(ti.ijk, block_size).place(f)

    f[0, 2, 3] = 1

    @ti.kernel
    def count() -> int:
        tot = 0
        for I in ti.grouped(block):
            tot += 1
        return tot

    assert count() == 1
```

Now, Taichi supports the following extensions:

| Name          | Extension details                                             |
| ------------- | ------------------------------------------------------------- |
| sparse        | Sparse data structures                                        |
| quant_basic   | Basic operations in quantization                              |
| quant         | Full quantization functionalities                             |
| data64        | 64-bit data and arithmetic                                   |
| adstack       | For keeping the history of mutable local variables in autodiff|
| bls           | Block-local storage                                           |
| assertion     | Run-time asserts in Taichi kernels                            |
| extfunc       | Support inserting external function calls or backend source   |
| packed        | Packed mode: shapes will not be padded to powers of two       |
| dynamic_index | Dynamic index support for tensors                             |
