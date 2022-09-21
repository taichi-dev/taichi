---
sidebar_position: 1
---

# Debugging

Debugging a parallel program can be tricky. To help you debug your Taichi program more easily, Taichi provides a few built-in utilities, including:

- `print` in the Taichi scope, which checks the value of a variable,
- serialization of your program or a specified parallel for loop,
- the debug mode activated by `ti.init(debug=True)`, which can detect out-of-bound array accesses,
- static or non-static `assert`, which verifies an assertion condition at compile time or runtime respectively,
- `sys.tracebacklimit`, which produces a more concise traceback

## Runtime `print` in Taichi scope

```python
print(arg1, ..., sep='', end='\n')
```

Debug your program with `print()` in the Taichi scope. For example:

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

For now, `print`, when placed in the Taichi scope, can take string literal, scalar, vector, and matrix expressions as arguments. It behaves differently depending on the scope where it is called, as detailed below.

### Applicable backends

`print` in the Taichi scope is currently supported on the CPU, CUDA, and Vulkan backends only.

Note that `print` does not work in Graphical Python Shells, including IDLE and Jupyter Notebook. This is because these backends print the outputs to the console instead of GUI.

### Comma-separated strings only

Strings passed to `print` in the Taichi scope *must* be comma-separated strings. Neither f-strings nor formatted strings can be recognized. For example:

```python {9-11}
import taichi as ti
ti.init(arch=ti.cpu)
a = ti.field(ti.f32, 4)


@ti.kernel
def foo():
    a[0] = 1.0
    print('a[0] = ', a[0]) # right
    print(f'a[0] = {a[0]}') # wrong: f-strings are not supported
    print("a[0] = %f" % a[0]) # wrong: formatted strings are not supported

foo()
```

## Compile-time `ti.static_print`

It can be useful to print Python objects and their properties like data types or SNodes in the Taichi scope. Based on `ti.static` (see [Metaprogramming](../advanced/meta.md)), Taichi provides `ti.static_print` to print compile-time constants in the Taichi scope:

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

`ti.static_print` behaves similarly with `print` in the Python scope. The difference is that unlike `print`, `ti.static_print` only prints the expression once at compile time, thus incurring no runtime cost.

## Serial execution

Taichi's automatic parallelization mechanism may lead to non-deterministic behaviors. For debugging purposes, serializing program execution may be useful to get repeatable results and diagnose data races. You can serialize either an entire Taichi program or a specific for loop.

### Serialize entire Taichi program

If you choose CPU as the backend, you can serialize the program with `cpu_max_num_threads=1` when initiating Taichi, so that the whole program runs on a single thread and becomes deterministic. For example:

```python
ti.init(arch=ti.cpu, cpu_max_num_threads=1)
```

If your program works well in serial but fails in parallel, check parallelization-related issues, such as data races.

### Serialize the for loop at the outermost scope

By default, Taichi automatically parallelizes the for loop at the outermost scope. But some scenarios require serial exeution. In this case, you can prevent automatic parallelization with `ti.loop_config(serialize=True)`:

```python
import taichi as ti

ti.init(arch=ti.cpu)
n = 1024
val = ti.field(dtype=ti.i32, shape=n)

val.fill(1)

@ti.kernel
def prefix_sum():
    ti.loop_config(serialize=True) # serialize the for loop
    for i in range(1, n):
        val[i] += val[i - 1]

prefix_sum()
print(val)
```

:::note

1. `ti.loop_config(serialize=True)` decorates the outermost for loop that immediately follows it.
2. `ti.loop_config` works only for the *range-for* loop at the outermost scope.
3. Inner for loops are serialized by default.

:::

## Out-of-bound array access

The array access violation issue is common, but a program would usually proceed without raising a warning, only to end up with a wrong result. Even if a segmentation fault were triggered, it would be hard to debug.

Taichi makes out-of-bound array accesses readily detectable with an auto-debugging mode. You can activate the mode by setting `debug=True` when initiating Taichi. For example:

```python
import taichi as ti
ti.init(arch=ti.cpu, debug=True)
f = ti.field(dtype=ti.i32, shape=(32, 32))
@ti.kernel
def test() -> ti.i32:
    return f[0, 73]

print(test())
```

The code snippet above would raise a `TaichiAssertionError` indicating that you are trying to access a field with improper indices.

:::note
Automatic bound checks are supported on the CPU and CUDA beckends only.

After `debug=Ture` is turned on, your program performance may worsen.
:::

## Runtime `assert` in Taichi scope

You can use `assert` statements in the Taichi scope to verify the assertion conditions. If an assertion fails, the program will halt and throw a `TaichiAssertionError`.

:::note
`assert` is currently supported on the CPU, CUDA, and Metal backends.
:::

Make sure you activate the `debug` mode before using `assert` statements in the Taichi scope:

```python
ti.init(arch=ti.cpu, debug=True)

x = ti.field(ti.f32, 128)

@ti.kernel
def do_sqrt_all():
    for i in x:
        assert x[i] >= 0
        x[i] = ti.sqrt(x[i])
```

When you are done with debugging, set `debug=False`, and then the program will ignore the subsequent `assert` statements and avoid additional runtime overhead.

## Compile-time `ti.static_assert`

```python
ti.static_assert(cond, msg=None)
```

Like `ti.static_print`, Taichi also provides a static version of `assert`: `ti.static_assert`, which comes handy to assert data types, dimensionality, and shapes. It works regardless whether `debug=True` is enabled or not. A false statement triggers an `AssertionError`, just as `assert` in the Python scope does.

For example:

```python
@ti.func
def copy(dst: ti.template(), src: ti.template()):
    ti.static_assert(dst.shape == src.shape, "copy() needs src and dst fields to be same shape")
    for I in ti.grouped(src):
        dst[I] = src[I]
```

## More concise traceback in Taichi scope

Taichi reports a traceback when an error occurs in the **Taichi scope**. For example:

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

The above snippet would trigger an `AssertionError`, with a lenthy and overwhelming traceback message:

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
taichi.lang.exception.TaichiCompilationError: File "misc/demo_traceback.py", line 10:
    ti.static_assert(1 + 1 == 3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError:

...
```

It takes time to read through the message. In addition, many stack frames reveal implementation details, which are irrelevant to debugging.

To relieve your burden, Taichi allows you to access a more concise and intuitive version of traceback messages via `sys.tracebacklimit`:

```python {2}
import taichi as ti
import sys
sys.tracebacklimit=0
...
```

You will get the following information:

```python
AssertionError

During handling of the above exception, another exception occurred:

taichi.lang.exception.TaichiCompilationError: File "misc/demo_traceback.py", line 10:
    ti.static_assert(1 + 1 == 3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError:

...
```

However, always unset `sys.tracebacklimit` and submit the full traceback messages when filing an issue with us.

## Debugging tips

The above built-in tools cannot guarantee a smooth debugging experience, though. Here, we conclude some common bugs that one may encounter in a
Taichi program.

### Static type system

Taichi translates Python code into a statically typed language for high performance. Therefore, code in the Taichi scope may behave differently from native Python code, especially when it comes to variable types.

In the Taichi scope, the type of a variable is **determined upon initialization and never changes later**.

Although Taichi's static typing system provides better performance, it may make your program more error-prone if you fail to specify the correct types. For example:

```python
@ti.kernel
def buggy():
    ret = 0  # 0 is an integer, so `ret` is typed as int32
    for i in range(3):
        ret += 0.1 * i  # i32 += f32, the result is still stored in int32!
    print(ret)  # will show 0

buggy()
```

The code above results in an unexpected result due to a misuse of Taichi's static typing system. The Taichi compiler should show a warning:

```
[W 06/27/20 21:43:51.853] [type_check.cpp:visit@66] [$19] Atomic add (float32 to int32) may lose precision.
```

This means that Taichi cannot convert a `float32` result to `int32` without precision loss. The solution is to initialize `ret` as a floating-point value:

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

Taichi runs a handful of advanced IR optimizations by default to make your Taichi kernels as performant as possible. However, advanced optimizations may occasionally lead to compilation errors, such as the following:

`RuntimeError: [verify.cpp:basic_verify@40] stmt 8 cannot have operand 7.`

You can turn off the advanced optimizations by setting `ti.init(advanced_optimization=False)` and see if it makes a difference. If the issue persists, please feel free to report it on [GitHub](https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md).
