---
sidebar_position: 2
---

# Debugging

Debugging a parallel program is not easy, so Taichi provides builtin
utilities that could hopefully help you debug your Taichi program.

## Run-time `print` in kernels

```python
print(arg1, ..., sep='', end='\n')
```

Debug your program with `print()` in Taichi-scope. For example:

```python {1}
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
```

For now, Taichi-scope `print` supports string, scalar, vector, and
matrix expressions as arguments. `print` in Taichi-scope may be a little
different from `print` in Python-scope. Please see details below.

:::caution
For the **CPU and CUDA backends**, `print` will not work in Graphical
Python Shells including IDLE and Jupyter notebook. This is because these
backends print the outputs to the console instead of the GUI. Use the
**OpenGL or Metal backend** if you wish to use `print` in IDLE /
Jupyter.
:::

:::caution

For the **CUDA backend**, the printed result will not show up until
`ti.sync()` is called:

```python
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
```

results in:

```
before kernel
after kernel
inside kernel
after sync
```

Note that host access or program end will also implicitly invoke
`ti.sync()`.
:::

:::note
Note that `print` in Taichi-scope can only receive **comma-separated
parameters**. Neither f-string nor formatted string should be used. For
example:

```python {9-11}
import taichi as ti
ti.init(arch=ti.cpu)
a = ti.field(ti.f32, 4)


@ti.kernel
def foo():
    a[0] = 1.0
    print('a[0] = ', a[0]) # right
    print(f'a[0] = {a[0]}') # wrong, f-string is not supported
    print("a[0] = %f" % a[0]) # wrong, formatted string is not supported

foo()
```

:::

## Compile-time `ti.static_print`

Sometimes it is useful to print Python-scope objects and constants like
data types or SNodes in Taichi-scope. So, similar to `ti.static`, Taichi
provides `ti.static_print` to print compile-time constants, which is similar
to Python-scope `print`:

```python
x = ti.field(ti.f32, (2, 3))
y = 1

@ti.kernel
def inside_taichi_scope():
    ti.static_print(y)
    # => 1
    ti.static_print(x.shape)
    # => (2, 3)
    ti.static_print(x.dtype)
    # => DataType.float32
    for i in range(4):
            ti.static_print(i.dtype)
            # => DataType.int32
            # will only print once
```

Unlike `print`, `ti.static_print` will only print the expression once at
compile-time, and therefore it has no runtime cost.

## Serial execution

The automatic parallelization feature of Taichi may lead to
nondeterministic behaviors. For debugging purposes, it may be useful to
serialize program execution to get repeatable results and to diagnose
data races. When running your Taichi program on CPUs, you can initialize
Taichi to use a single thread with `cpu_max_num_threads=1`, so that the
whole program becomes serial and deterministic. For example,

```
ti.init(arch=ti.cpu, cpu_max_num_threads=1)
```

If you program works well in serial but not in parallel, check
parallelization-related issues such as data races.

## Runtime `assert` in kernel

Programmers may use `assert` statements in Taichi-scope. When the
assertion condition failed, a `RuntimeError` will be raised to indicate
the error.

:::note
`assert` is currently supported on the CPU, CUDA, and Metal backends.
:::

For performance reason, `assert` only works when `debug` mode
is on. For example:

```python
ti.init(arch=ti.cpu, debug=True)

x = ti.field(ti.f32, 128)

@ti.kernel
def do_sqrt_all():
    for i in x:
        assert x[i] >= 0
        x[i] = ti.sqrt(x)
```

When you are done with debugging, simply set `debug=False`. Now `assert`
will be ignored and there will be no runtime overhead.

## Compile-time `ti.static_assert`

```python
ti.static_assert(cond, msg=None)
```

Like `ti.static_print`, Taichi also provides a static version of `assert`:
`ti.static_assert`. It can be useful to make assertions on data types,
dimensionality, and shapes. It works whether `debug=True` is specified
or not. When an assertion fails, it will raise an `AssertionError`, just
like a Python-scope `assert`.

For example:

```python
@ti.func
def copy(dst: ti.template(), src: ti.template()):
    ti.static_assert(dst.shape == src.shape, "copy() needs src and dst fields to be same shape")
    for I in ti.grouped(src):
        dst[I] = src[I]
    return x % 2 == 1
```

## Pretty Taichi-scope traceback

Sometimes the Python stack tracebacks resulted from **Taichi-scope** errors
could be too complicated to read. For example:

```python
import taichi as ti
ti.init()

@ti.func
def func3():
    ti.static_assert(1 + 1 == 3)

@ti.func
def func2():
    func3()

@ti.func
def func1():
    func2()

@ti.kernel
def func0():
    func1()

func0()
```

The above snippet would result in an `AssertionError`:

```
Traceback (most recent call last):
  File "misc/demo_excepthook.py", line 20, in <module>
    func0()
  File "/root/taichi/python/taichi/lang/kernel.py", line 559, in wrapped
    return primal(*args, **kwargs)
  File "/root/taichi/python/taichi/lang/kernel.py", line 488, in __call__
    self.materialize(key=key, args=args, arg_features=arg_features)
  File "/root/taichi/python/taichi/lang/kernel.py", line 367, in materialize
    taichi_kernel = taichi_kernel.define(taichi_ast_generator)
  File "/root/taichi/python/taichi/lang/kernel.py", line 364, in taichi_ast_generator
    compiled()
  File "misc/demo_excepthook.py", line 18, in func0
    func1()
  File "/root/taichi/python/taichi/lang/kernel.py", line 39, in decorated
    return fun.__call__(*args)
  File "/root/taichi/python/taichi/lang/kernel.py", line 79, in __call__
    ret = self.compiled(*args)
  File "misc/demo_excepthook.py", line 14, in func1
    func2()
  File "/root/taichi/python/taichi/lang/kernel.py", line 39, in decorated
    return fun.__call__(*args)
  File "/root/taichi/python/taichi/lang/kernel.py", line 79, in __call__
    ret = self.compiled(*args)
  File "misc/demo_excepthook.py", line 10, in func2
    func3()
  File "/root/taichi/python/taichi/lang/kernel.py", line 39, in decorated
    return fun.__call__(*args)
  File "/root/taichi/python/taichi/lang/kernel.py", line 79, in __call__
    ret = self.compiled(*args)
  File "misc/demo_excepthook.py", line 6, in func3
    ti.static_assert(1 + 1 == 3)
  File "/root/taichi/python/taichi/lang/error.py", line 14, in wrapped
    return foo(*args, **kwargs)
  File "/root/taichi/python/taichi/lang/impl.py", line 252, in static_assert
    assert cond
AssertionError
```

Many of the stack frames are the Taichi compiler implementation details, which
could be too noisy to read. You could choose to elide them by using
`ti.init(excepthook=True)`, which _hooks_ on the exception handler, and makes
the stack traceback from Taichi-scope more intuitive:

```python {2}
import taichi as ti
ti.init(excepthook=True)
...
```

which makes the result look like:

```python
========== Taichi Stack Traceback ==========
In <module>() at misc/demo_excepthook.py:21:
--------------------------------------------
@ti.kernel
def func0():
    func1()

func0()  <--
--------------------------------------------
In func0() at misc/demo_excepthook.py:19:
--------------------------------------------
    func2()

@ti.kernel
def func0():
    func1()  <--

func0()
--------------------------------------------
In func1() at misc/demo_excepthook.py:15:
--------------------------------------------
    func3()

@ti.func
def func1():
    func2()  <--

@ti.kernel
--------------------------------------------
In func2() at misc/demo_excepthook.py:11:
--------------------------------------------
    ti.static_assert(1 + 1 == 3)

@ti.func
def func2():
    func3()  <--

@ti.func
--------------------------------------------
In func3() at misc/demo_excepthook.py:7:
--------------------------------------------
ti.enable_excepthook()

@ti.func
def func3():
    ti.static_assert(1 + 1 == 3)  <--

@ti.func
--------------------------------------------
AssertionError
```

:::note
For IPython / Jupyter notebook users, the IPython stack traceback hook
will be overriden by the Taichi one when `ti.enable_excepthook()` is called.
:::

## Debugging Tips

Debugging a Taichi program can be hard even with the above builtin tools.
Here we showcase some common bugs that one may encounter in a
Taichi program.

### Static type system

Python code in Taichi-scope is translated into a statically typed
language for high performance. This means code in Taichi-scope can have
a different behavior compared with that in Python-scope, especially when
it comes to types.

The type of a variable is simply **determined at its initialization and
never changes later**.

Although Taichi's static type system provides better performance, it
may lead to bugs if programmers used the wrong types. For
example:

```python
@ti.kernel
def buggy():
    ret = 0  # 0 is an integer, so `ret` is typed as int32
    for i in range(3):
        ret += 0.1 * i  # i32 += f32, the result is still stored in int32!
    print(ret)  # will show 0

buggy()
```

The code above shows a common bug due to misuse of the Taichi's static type system,
the Taichi compiler should show a warning like:

```
[W 06/27/20 21:43:51.853] [type_check.cpp:visit@66] [$19] Atomic add (float32 to int32) may lose precision.
```

This means that Taichi cannot store a `float32` result precisely to
`int32`. The solution is to initialize `ret` as a float-point value:

```python
@ti.kernel
def not_buggy():
    ret = 0.0  # 0 is a floating point number, so `ret` is typed as float32
    for i in range(3):
        ret += 0.1 * i  # f32 += f32. OK!
    print(ret)  # will show 0.6

not_buggy()
```

### Advanced Optimization

By default, Taichi runs a handful of advanced IR optimizations to make your
Taichi kernels as performant as possible. Unfortunately, advanced
optimization may occasionally lead to compilation errors, such as the following:

`RuntimeError: [verify.cpp:basic_verify@40] stmt 8 cannot have operand 7.`

You can turn off the advanced optimizations with
`ti.init(advanced_optimization=False)` and see if it makes a difference. If
the issue persists, please feel free to report this bug on
[GitHub](https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md).
