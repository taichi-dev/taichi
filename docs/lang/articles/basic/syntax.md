---
sidebar_position: 1
---

# Kernels and functions

## Taichi-scope vs Python-scope

Code decorated by `@ti.kernel` or `@ti.func` is in the **Taichi-scope**.

They will be compiled by the Taichi compiler and executed on CPU or GPU devices in parallel with high performance.

:::note
For people from CUDA, Taichi-scope = **device** side.
:::

Code outside `@ti.kernel` or `@ti.func` is in the **Python-scope**.

It is native Python code and will not be compiled by the Taichi compiler.

:::note
For people from CUDA, Python-scope = **host** side.
:::

## Kernels

A Python function decorated by `@ti.kernel` is a **Taichi kernel**:

```python {1}
@ti.kernel
def my_kernel():
    ...

my_kernel()
```

Kernels should be called from **Python-scope**. Nested kernels are not supported.

:::note
For people from CUDA, Taichi kernels are similar to `__global__` functions.
:::

### Arguments

Kernels can have multiple arguments, which support passing values from Python-scope to Taichi-scope conveniently.

:::note
For kernels executed on OpenGL and CC backends, the number of arguments is limited to 8.
:::

Kernel arguments must be type hinted:

```python {2}
@ti.kernel
def my_kernel(x: ti.i32, y: ti.f32):
    print(x + y)

my_kernel(24, 3.2)  # prints: 27.2
```

:::note

For now, Taichi supports scalars as kernel arguments. Specifying `ti.Matrix` or
`ti.Vector` as an argument is not supported yet:

```python {2,7}
@ti.kernel
def valid_kernel(vx: ti.f32, vy: ti.f32):
    v = ti.Vector([vx, vy])
    ...

@ti.kernel
def error_kernel(v: ti.Vector): # Error: Invalid type annotation
    ...
```

:::

### Return value

It is optional for a kernel to have a return value. If specified, it must be a type hinted **scalar** value:

```python {2}
@ti.kernel
def my_kernel() -> ti.f32:
    return 128.32

print(my_kernel())  # 128.32
```

In addition, the return value will be automatically cast into the hinted type:

```python {2-3,5}
@ti.kernel
def my_kernel() -> ti.i32:  # int32
    return 128.32

print(my_kernel())  # 128, cast into ti.i32
```

:::note

For now, a kernel can only have one scalar return value. Returning
`ti.Matrix`, `ti.Vector` or Python-style tuple is not supported:

```python {3,9}

@ti.kernel
def valid_kernel() -> ti.f32:
    return 128.0  # Return 128.0

@ti.kernel
def error_kernel() -> ti.Matrix:
    return ti.Matrix([[1, 0], [0, 1]])  # Compilation error

@ti.kernel
def error_kernel() -> (ti.i32, ti.f32):
    x = 1
    y = 0.5
    return x, y  # Compilation error
```

:::

### Advanced arguments

Taichi also supports **template arguments** (see
[Template metaprogramming](../advanced/meta.md#template-metaprogramming)) and **external
array arguments** (see [Interacting with external arrays](./external.md)) in
Taichi kernels. Use `ti.template()` or `ti.ext_arr()` as their
type-hints respectively.

:::note

For differentiable programming related features, there are a few more constraints
on kernel structures. See the [**Kernel Simplicity Rule**](../advanced/differentiable_programming.md#kernel-simplicity-rule).

Besides, please do not specify a return value for kernels in differentiable
programming, since the return value will not be tracked by automatic
differentiation. Instead, it is recommended to store the result into a global variable (e.g.
`loss[None]`).
:::

## Functions

A Python function decorated by `@ti.func` is a **Taichi function**:

```python {8,11}
@ti.func
def my_func():
    ...

@ti.kernel
def my_kernel():
    ...
    my_func()  # call functions from Taichi-scope
    ...

my_kernel()    # call kernels from Python-scope
```

Taichi functions can only be called from **Taichi-scope**.

:::note
For people from CUDA, Taichi functions are similar to `__device__` functions.
:::

:::note
Taichi functions can be nested.
:::

:::caution
Currently, all functions are force-inlined. Therefore, no recursion is
allowed.
:::

### Arguments and return values

Functions can have multiple arguments and return values. Unlike kernels,
arguments in functions are not required to be type-hinted:

```python
@ti.func
def my_add(x, y):
    return x + y


@ti.kernel
def my_kernel():
    ...
    ret = my_add(24, 3.2)
    print(ret)  # 27.2
    ...
```

Function arguments are passed by value. So changes made inside the function
scope won't affect the original value in the caller:

```python {3,9,11}
@ti.func
def my_func(x):
    x = x + 1  # won't change the original value of x


@ti.kernel
def my_kernel():
    ...
    x = 24
    my_func(x)
    print(x)  # 24
    ...
```

### Advanced arguments

By using `ti.template()` as a type hint, arguments are forced to be passed by reference:

```python {3,9,11}
@ti.func
def my_func(x: ti.template()):
    x = x + 1  # This line will change the original value of x


@ti.kernel
def my_kernel():
    ...
    x = 24
    my_func(x)
    print(x)  # 25
    ...
```

:::note

Unlike kernels, functions **do support vectors or matrices as arguments
and return values**:

```python {2,6}
@ti.func
def sdf(u):  # functions support matrices and vectors as arguments. No type-hints needed.
    return u.norm() - 1

@ti.kernel
def render(d_x: ti.f32, d_y: ti.f32):  # Kernels do not support vector/matrix arguments yet.
    d = ti.Vector([d_x, d_y])
    p = ti.Vector([0.0, 0.0])
    t = sdf(p)
    p += d * t
    ...
```

:::

:::caution

Functions with multiple `return` statements are not supported for now.
It is recommended to use a **local** variable to store the results, so that only one `return` statement is needed:

```python {1,5,7,9,17}
# Error function - two return statements
@ti.func
def safe_sqrt(x):
  if x >= 0:
    return ti.sqrt(x)
  else:
    return 0.0

# Valid function - single return statement
@ti.func
def safe_sqrt(x):
  ret = 0.0
  if x >= 0:
    ret = ti.sqrt(x)
  else:
    ret = 0.0
  return ret
```

:::
