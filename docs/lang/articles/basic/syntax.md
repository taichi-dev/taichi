---
sidebar_position: 1
---

# Kernels and functions

Taichi has two types of functions: Taichi kernel, and Taichi functions.

Scope inside Taichi kernels and Taichi functions is called Taichi scope, and scope outside it is called Python scope.

A Taichi kernel is the entrypoint of a Taichi program, and it is similar to `__global__` function in CUDA. It can only be called inside Python scope.

A Taichi function can only be called inside Taichi scope, and it is similar to `__device__` function in CUDA.

Major differences of Taichi kernels and Taichi functions are listed in the table below.

| | Taichi kernels | Taichi functions |
| :--- | :---: | :---: |
| Can be called in | Python scope | Taichi scope |
| Argument type annotation | Mandatory | Recommended |
| Return type annotation | Mandatory| Recommended |
| Return value | Scalar/Vector/Matrix | Arbitrary |
| Number of total elements in arguments | Up to 64 | Unlimited |
| Number of return values in a return statement | 1 | Unlimited |
| Number of elements in return value | Up to 30 | Unlimited |




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
:::

Kernel arguments must be type hinted:

```python {2}
@ti.kernel
def my_kernel(x: ti.i32, y: ti.f32):
    print(x + y)

my_kernel(24, 3.2)  # prints: 27.2
```

:::note
Taichi supports scalars, `ti.Matrix` and`ti.Vector` as kernel arguments. The total number of elements in kernel arguments must not exceed 64.
The element number of a scalar argument is 1, and the element number of a `ti.Matrix` or`ti.Vector` is the number of elements inside it.

```python {2,7,11}
@ti.kernel
def valid_scalar_argument(vx: ti.f32, vy: ti.f32):
    v = ti.Vector([vx, vy])
    ...

@ti.kernel
def valid_matrix_argument(u: ti.i32, v: ti.types.matrix(2, 2, ti.i32)):  # OK: has 5 elements in total
    ...

@ti.kernel
def error_too_many_arguments(u: ti.i32, v: ti.i64, w: ti.types.matrix(7, 9, ti.i64)):  # Error: has 65 elements in total
    ...
```

:::

### Return value

It is optional for a kernel to have a return value. If specified, it must be a type hinted **scalar/vector/matrix** value:

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

For now, a kernel can only have one return value, and the number of elements in the return value must not exceed 30.

```python {3,9}

@ti.kernel
def valid_scalar_return() -> ti.f32:
    return 128.0  # Return 128.0

@ti.kernel
def valid_matrix_return() -> ti.types.matrix(2, 2, ti.i32):
    return ti.Matrix([[1, 0], [0, 1]])

@ti.kernel
def error_multiple_return() -> (ti.i32, ti.f32):
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
Currently, all functions are force-inlined. Therefore, no runtime recursion is allowed.

Compile-time recursion is an advanced metaprogramming feature for experienced programmers. See [Metaprogramming](/lang/articles/advanced/meta#compile-time-recursion-of-tifunc) for more information.
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
