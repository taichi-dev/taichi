---
sidebar_position: 1
---

# Debugging

Debugging a parallel program is not easy, so Taichi provides built-in
utilities that could hopefully help you debug your Taichi program.

## Runtime `print` in Taichi-scope

```python
print(arg1, ..., sep='', end='\n')
```

Debug your program with `print()` in Taichi-scope. For example:

```python {1}
@ti.kernel
def inside_taichi_scope():
    x = 256
    print('hello', x)
    #=> hello 256

    print('hello', x * 2 + 200)
    #=> hello 712

    print('hello', x, sep='')
    #=> hello256

    print('hello', x, sep='', end='')
    print('world', x, sep='')
    #=> hello256world256

    m = ti.Matrix([[2, 3, 4], [5, 6, 7]])
    print('m =', m)
    #=> m = [[2, 3, 4], [5, 6, 7]]

    v = ti.Vector([3, 4])
    print('v =', v)
    #=> v = [3, 4]

    ray = ti.Struct({
    	"ori": ti.Vector([0.0, 0.0, 0.0]),
    	"dir": ti.Vector([0.0, 0.0, 1.0]),
    	"len": 1.0
		})
    # print(ray)
    # Print a struct directly in Taichi-scope has not been supported yet
    # Instead, use:
    print('ray.ori =', ray.ori, ', ray.dir =', ray.dir, ', ray.len =', ray.len)
    #=> ray.ori = [0.0, 0.0, 0.0], ray.dir = [0.0, 0.0, 1.0], ray.len = 1.0
```

For now, Taichi-scope `print` supports string, scalar, vector, and
matrix expressions as arguments. `print` in Taichi-scope may be a little
different from `print` in Python-scope. Please see the details below.

:::note
`print` in Taichi-scope is currently supported on the CPU, CUDA, OpenGL, and Metal backends.
:::

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
non-deterministic behaviors. For debugging purposes, it may be useful to
serialize program execution to get repeatable results and to diagnose
data races. When running your Taichi program on CPUs, you can initialize
Taichi to use a single thread with `cpu_max_num_threads=1`, so that the
whole program becomes serial and deterministic. For example,

```
ti.init(arch=ti.cpu, cpu_max_num_threads=1)
```

If your program works well in serial but not in parallel, check
parallelization-related issues such as data races.

## Runtime `assert` in Taichi-scope

Programmers may use `assert` statements in Taichi-scope. When the
assertion condition fails, a `TaichiAssertionError` will be raised to indicate
the error.

:::note
`assert` is currently supported on the CPU, CUDA, and Metal backends.
:::

For performance reasons, `assert` only works when `debug` mode
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

Taichi reports traceback messages when encountered errors in **Taichi-scope**. For example:

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

The above snippet would trigger a long and scaring `AssertionError`:

```
Traceback (most recent call last):
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 23, in __call__
    return method(ctx, node)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer.py", line 342, in build_Call
    node.ptr = node.func.ptr(*args, **keywords)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/impl.py", line 471, in static_assert
    assert cond
AssertionError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 23, in __call__
    return method(ctx, node)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer.py", line 360, in build_Call
    node.ptr = node.func.ptr(*args, **keywords)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/kernel_impl.py", line 59, in decorated
    return fun.__call__(*args)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/kernel_impl.py", line 178, in __call__
    ret = transform_tree(tree, ctx)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/transform.py", line 8, in transform_tree
    ASTTransformer()(ctx, tree)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 26, in __call__
    raise e
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 23, in __call__
    return method(ctx, node)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer.py", line 488, in build_Module
    build_stmt(ctx, stmt)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 26, in __call__
    raise e
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 23, in __call__
    return method(ctx, node)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer.py", line 451, in build_FunctionDef
    build_stmts(ctx, node.body)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer.py", line 1086, in build_stmts
    build_stmt(ctx, stmt)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 26, in __call__
    raise e
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 23, in __call__
    return method(ctx, node)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer.py", line 964, in build_Expr
    build_stmt(ctx, node.value)
  File "/Users/lanhaidong/taichi/taichi/python/taichi/lang/ast/ast_transformer_utils.py", line 32, in __call__
    raise TaichiCompilationError(msg)
taichi.lang.exception.TaichiCompilationError: On line 10 of file "misc/demo_traceback.py":
    ti.static_assert(1 + 1 == 3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError:

...
```
The error message can be verbose and scary. However, many stack frames reveal
Taichi compiler implementation details, which are too noisy for debugging.
In current version, you could choose to suppress the level of traceback messages by setting `sys.tracebacklimit`,
which makes the stack traceback from Taichi-scope more intuitive:

```python {2}
import taichi as ti
import sys
sys.tracebacklimit=0
...
```

which makes the result look like:

```python
AssertionError

During handling of the above exception, another exception occurred:

taichi.lang.exception.TaichiCompilationError: On line 10 of file "misc/demo_traceback.py":
    ti.static_assert(1 + 1 == 3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError:

...
```

Moreover, when filing an issue, please always unset the `sys.tracebacklimit` value and paste full traceback messages.

## Debugging Tips

Debugging a Taichi program can be hard even with the above built-in tools.
Here we showcase some common bugs that one may encounter in a
Taichi program.

### Static type system

Python code in Taichi-scope is translated into a statically typed
language for high performance. This means code in Taichi-scope can have
a different behavior compared with that in Python-scope, especially when
it comes to types.

The type of a variable is **determined at its initialization and
never changes later**.

Although Taichi's static type system provides better performance, it
may lead to bugs if programmers use the wrong types. For
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

The code above shows a common bug due to misuse of Taichi's static type system.
The Taichi compiler should show a warning like:

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
