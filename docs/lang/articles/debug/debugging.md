---
sidebar_position: 1
---

# Debugging

Taichi provides the following mechanisms to facilitate parallel programming debugging:

- `print` in the Taichi scope checks the value of a variable.
- Serialization of your program or a specific parallel for loop.
- Activated with `ti.init(debug=True)`, debug mode detects out-of-bound array accesses.
- Static or non-static `assert` verifies an assertion condition at compile time or runtime respectively.
- `sys.tracebacklimit` produces a conciser traceback.

## Runtime `print` in Taichi scope


You can call `print()` in the Taichi scope to debug your program:

```python
print(*args, sep='', end='\n')
```

When passed into a runtime `print()` in the Taichi scope, `args` can take string literal, scalar, vector, and matrix expressions.


For example:

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


### Applicable backends

`print` in the Taichi scope is supported on the CPU, CUDA, and Vulkan backends only.

:::note
To enable printing on vulkan backend, please
- make sure validation layer is installed via [vulkan sdk](https://vulkan.lunarg.com/sdk/home).
- turn on debug mode by `ti.init(debug=True)`.

Note printing is not supported on macOS vulkan backend.
:::

:::note
`print` does not work in Graphical Python Shells, such as IDLE and Jupyter Notebook. This is because these backends print outputs to the console, not to the GUI.
:::

### Comma-separated strings only

Strings passed to `print` in the Taichi scope *must* be comma-separated. Neither f-strings nor formatted strings can be recognized. For example:

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

It can be useful to print Python objects and their properties like data types or SNodes in the Taichi scope. Similar to `ti.static`, which makes the compiler evaluate an argument at compile time (see the [Metaprogramming](../advanced/meta.md) for more information), `ti.static_print` prints compile-time constants in the Taichi scope:

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
        # Only print once
```

`ti.static_print` behaves similarly to `print` in the Python scope. The difference is that, unlike `print`, `ti.static_print` prints the expression only once at compile time, thus incurring no runtime cost.

## Serial execution

Taichi's automatic parallelization mechanism may lead to non-deterministic behaviors because the threads are executed in random order. For debugging purposes, serializing program execution may be useful for getting repeatable results or diagnosing data races. You can serialize either the entire Taichi program or a specific for loop.

### Serialize an entire Taichi program

If you choose CPU as the backend, you can set `cpu_max_num_thread=1` when initializing Taichi to serialize the program. Then the program runs on a single thread and its behavior becomes deterministic. For example:

```python
ti.init(arch=ti.cpu, cpu_max_num_threads=1)
```

If your program works well in serial but fails in parallel, check if there are parallelization-related issues, such as data races.

### Serialize a specified parallel for loop

By default, Taichi automatically parallelizes the for loops at the outermost scope in a Taichi kernel. But some scenarios require serial execution. In this case, you can prevent automatic parallelization with `ti.loop_config(serialize=True)`. Note that only the outermost for loop that immediately follows this line is serialized. For example:

```python
import taichi as ti

ti.init(arch=ti.cpu)
n = 1024
val = ti.field(dtype=ti.i32, shape=n)

val.fill(1)

@ti.kernel
def prefix_sum():
    ti.loop_config(serialize=True) # Serializes the next for loop
    for i in range(1, n):
        val[i] += val[i - 1]

    for i in range(1, n):  # Parallel for loop
	    val[i] += val[i - 1]

prefix_sum()
print(val)
```

:::note

- `ti.loop_config` works only for the *range-for* loop at the outermost scope.

:::

## Out-of-bound array access

The array index out of bounds error is common. But Taichi turns off bounds checking by default and proceeds without raising a warning. Therefore, a program with such an error may end up with a wrong result or even trigger segmentation faults, which makes debugging hard.

Taichi detects array index out of bound errors in debug mode. You can activate this mode by setting `debug=True` in the `ti.init()` call:

```python
import taichi as ti
ti.init(arch=ti.cpu, debug=True)
f = ti.field(dtype=ti.i32, shape=(32, 32))
@ti.kernel
def test() -> ti.i32:
    return f[0, 73]

print(test())
```

The code snippet above raises a `TaichiAssertionError` because you are trying to access elements from a field of shape (32, 32) with indices `[0, 73]`.

:::note
Automatic bound checks are supported on the CPU and CUDA beckends only.

Your program performance may worsen if you set `debug=True`.
:::

## Runtime `assert` in Taichi scope

You can use `assert` statements in the Taichi scope to verify the assertion conditions. If an assertion fails, the program throws a `TaichiAssertionError`.

:::note
`assert` is currently supported on the CPU, CUDA, and Metal backends.
:::

Ensure that you activate `debug` mode before using `assert` statements in the Taichi scope:

```python
import taichi as ti
ti.init(arch=ti.cpu, debug=True)

x = ti.field(ti.f32, 128)
x.fill(-1)

@ti.kernel
def do_sqrt_all():
    for i in x:
        assert x[i] >= 0, f"The {i}-th element cannot be negative"
        x[i] = ti.sqrt(x[i])

do_sqrt_all()
```

When you are done with debugging, set `debug=False`. Then, the program ignores all `assert` statements in the Taichi scope, which can avoid additional runtime overhead.

## Compile-time `ti.static_assert`

```python
ti.static_assert(cond, msg=None)
```

Like `ti.static_print`, Taichi also provides a static version of `assert`: `ti.static_assert`, which is useful to make assertions on data types, dimensionality, and shapes. It works regardless of whether `debug=True` is set or not. A false statement of `ti.static_assert` triggers an `AssertionError`, just as a false `assert` statement in the Python scope does.

For example:

```python
@ti.func
def copy(dst: ti.template(), src: ti.template()):
    ti.static_assert(dst.shape == src.shape, "copy() needs src and dst fields to be same shape")
    for I in ti.grouped(src):
        dst[I] = src[I]
```

## Conciser tracebacks in Taichi scope

Taichi reports the traceback of an error in the **Taichi scope**. For example:

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

The above snippet triggers an `AssertionError`, with a lengthy traceback message:

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

Taichi allows you to access a conciser and more intuitive version of traceback messages via `sys.tracebacklimit`:

```python {2}
import taichi as ti
import sys
sys.tracebacklimit=0
...
```

The traceback contains the following information only:

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

In the Taichi scope, the type of a variable is *determined upon initialization and never changes afterwards*.

Although Taichi's static typing system delivers a better performance, it may lead to unexpected results if you fail to specify the correct types. For example:

```python
@ti.kernel
def buggy():
    ret = 0  # 0 is an integer, so `ret` is typed as int32
    for i in range(3):
        ret += 0.1 * i  # i32 += f32, the result is still stored in int32!
    print(ret)  # will show 0

buggy()
```

The code above leads to an unexpected result due to a misuse of Taichi's static typing system. The Taichi compiler shows a warning:

```
[W 06/27/20 21:43:51.853] [type_check.cpp:visit@66] [$19] Atomic add (float32 to int32) may lose precision.
```

This means that a precision loss occurs when Taichi converts a `float32` result to `int32`. The solution is to initialize `ret` as a floating-point value:

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

By default, Taichi runs a number of advanced IR optimizations to maximize the performance of your Taichi kernels. However, advanced optimizations may occasionally lead to compilation errors, such as:

`RuntimeError: [verify.cpp:basic_verify@40] stmt 8 cannot have operand 7.`

You can use the `ti.init(advanced_optimization=False)` setting to turn off advanced optimizations and see if it makes a difference. If this issue persists, feel free to report it on [GitHub](https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md).
