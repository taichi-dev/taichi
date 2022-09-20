---
sidebar_position: 1
---

# Kernels and functions

Embedded in Python, Taichi resembles Python in language syntax. To differentiate Taichi code from native Python code, we use the two decorators `@ti.kernel` and `@ti.func`:

+ Functions decorated with `@ti.kernel` are called Taichi kernels (or kernels for short). They serve as the entry points where Taichi begins to take over the tasks, and they *must* be called directly by Python code.
+ Functions decorated with `@ti.func` are called Taichi functions. They serve as the building blocks of kernels and can *only* be called by kernels or other Taichi functions.

Let's see an example:

```python
import taichi as ti
ti.init(arch=ti.cpu)

@ti.func
def inv_square(x):  # a Taichi function
    return 1.0 / (x * x)

@ti.kernel
def partial_sum(n: int) -> float:  # a kernel
    total = 0.0
    for i in range(1, n + 1):
        total += inv_square(n)
    return total
```

In the code above, `inv_square()` is a Taichi function because it is decorated by `@ti.func`, while `partial_sum()` is a kernel because it is decorated by `@ti.kernel`. The Taichi function (former) is called by the kernel (latter).

You may have noticed that the argument and the return value in the **kernel** `partial_sum()` are both type-hinted, but those in the **Taichi function** `inv_square()` are not. Here comes an important difference between Python and Taichi. In native Python code, type hinting is recommended but not mandatory. But Taichi makes it *compulsory* that kernels must take type-hinted arguments and return type-hinted values. The only exception where you can leave out a type hint in a kernel is that the kernel does not have an argument or a `return` statement.

It is worth your attention that Taichi will raise a syntax error if you try to call `inv_square()` directly from the native Python code (i.e., out of the Taichi scope).
For example:

```python
import taichi as ti
ti.init(arch=ti.cpu)

@ti.func
def inv_square(x):
    return 1.0 / (x * x)

print(inv_square(1.0))  # Syntax error!
```

The Taichi function should have fallen in the Taichi scope, a concept as opposed to the "Python scope".

:::tip IMPORTANT

We give the following definitions:

1. The code inside a kernel or a Taichi function is in the **Taichi scope**. The code in the Taichi scope is compiled by Taichi's runtime and executed in parallel on CPU or GPU devices for high-performance computation.

   The Taichi scope corresponds to the device side in CUDA.

2. Code outside of the Taichi scope is in the **Python scope**. The code in the Python scope is native Python and executed by Python's virtual machine, not by Taichi's runtime.

   The Python scope corresponds to the host side in CUDA.

:::

We should not confuse kernels with Taichi functions. Though they belong to the Taichi scope, the syntax rules applied to them are not exactly the same. We now dive into their usages and the roles they play in detail.


## Kernel

As the smallest execution unit in Taichi, a kernel is the entry point from which Taichi's runtime takes control. You call a kernel the same way you call a Python function and can switch back and forth between Taichi's runtime and Python's virtual machine.

For example, you can call the kernel `partial_sum()` as defined in the above section from inside a Python function:

```python
def main():
    print(partial_sum(100))
    print(partial_sum(1000))

main()
```

You can define multiple kernels in your program. They are mutually *independent* of each other and are compiled and executed in the same order as they are first called. The compiled kernels are stored in the cache to save the launch overhead for subsequent calls.

:::caution WARNING

You must *not* call a kernel from inside another kernel or from inside a Taichi function. You can only call a kernel directly or from inside a native Python function. In other words, you can only call a kernel from inside the Python scope.

:::


### Arguments


A kernel can take multiple arguments. However, you *cannot* pass any arbitrary Python object to a kernel because Python objects can be highly dynamic and may hold data unrecognized by Taichi's compiler.

The argument types accepted by kernels are scalars, `ti.Matrix/ti.Vector` (In Taichi, vectors are essentially matrices), `ti.types.ndarray()` and `ti.template()`. You can easily pass data from the Python scope to the Taichi scope.

It should be noted that scalars and `ti.Matrix` are passed by value, while `ti.types.ndarray()` and `ti.template()` are passed by reference. In the latter case, any modification to the arguments in the called function also affects the original values.

In the following example, the arguments `x` and `y` are passed to `my_kernel` by value:

```python
@ti.kernel
def my_kernel(x: int, y: float):
    print(x + y)

my_kernel(1, 1.0)  # prints 2.0
```

Using `ti.types.ndarray()` as the type hint, you can pass a NumPy's `ndarray` or a PyTorch's `tensor` to a kernel. Taichi recognizes the shape and data type of such a data structure and allows you to access these attributes in a kernel. For example:

```python
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

@ti.kernel
def my_kernel(x: ti.types.ndarray(), y: ti.types.ndarray()):
    for i in range(x.shape[0]):
        x[i] += y[i]

my_kernel(x, y)
print(x)  # prints [5, 7, 9]
```

`x` is modified by `my_kernel()` because it is passed by reference.

:::note

We skip `ti.template()` here and leave it for a more advanced topic: Meta-programming. Refer to [Metaprogramming](../advanced/meta.md#template-metaprogramming) for more information.

:::


### Return value

A kernel can have *at most* one return value, which can be a scalar, `ti.Matrix`, or `ti.Vector`. Follow these rules when defining the return value of a kernel:

- Type hint the return value of a kernel.
- Ensure that you have *at most* one return value in a kernel.
- Ensure that you have *at most* one return statement in a kernel.
- Ensure that the number of elements in the return value does not exceed 30.

Let's see an exmaple:

```python
vec2 = ti.math.vec2

@ti.kernel
def test(x: float, y: float) -> vec2: # Return value must be type hinted
    # Return x, y  # Compilation error: Only one return value is allowed
    return vec2(x, y)  # OK!
```

In addition, the return value is automatically cast into the hinted type:

```python
@ti.kernel
def my_kernel() -> ti.i32:  # int32
    return 128.32

print(my_kernel())  # 128, the return value is cast into ti.i32
```


#### At most one return statement in a kernel

```python
@ti.kernel
def test_sign(x: float) -> float:
    if x >= 0:
        return 1.0
    else:
        return -1.0
    # Error: multiple return statements
```

As a workaround, you can save the result in a local variable and return it at the end:

```python
@ti.kernel
def test_sign(x: float) -> float:
    sign = 1.0
    if x < 0:
        sign = -1.0
    return sign
    # One return statement works fine
```


### Global variables are compile-time constants

A kernel treats global variables as compile-time constants. This means that it takes in the current values of the global variables at the time it is compiled and that it does not track changes to them afterwards. Then, if the value of a global variable is updated between two calls of the same kernel, the second call does not use the updated value.

Let's take a look at the following example, where the global variable `a` is updated after the first call of `kernel_1`.

- Because `kernel_1` does not track changes to `a` after it is compiled, the second call of `kernel_1` still prints `1`.
- Because `kernerl_2` is compiled after `a` is updated, it takes in the current value of `a` and prints `2`.

```python
import taichi as ti
ti.init()

a = 1

@ti.kernel
def kernel_1():
    print(a)


@ti.kernel
def kernel_2():
    print(a)

kernel_1()  # 1
a = 2
kernel_1()  # 1
kernel_2()  # 2
```

## Taichi function

Taichi functions are the building blocks of a kernel. **You must call a Taichi function from inside a kernel or from inside another Taichi function**. All Taichi functions are force-inlined. Therefore, no runtime recursion is allowed.

Let's see an example:

```python
# a normal python function
def foo_py():
    print("I'm a python function")

@ti.func
def foo_1():
    print("I'm a taichi function called by another taichi function")

@ti.func
def foo_2():
    print("I'm a taichi function called by a kernel")
    foo_1()

@ti.kernel
def foo_kernel():
    print("I'm a kernel calling a taichi function")
    foo_2()

foo_py()
#foo_func() # You cannot call a Taichi function from within the Python scope
foo_kernel()
```

### Arguments

A Taichi function can have multiple arguments, supporting scalar, `ti.Matrix/ti.Vector`, `ti.types.ndarray()`, `ti.template()`, `ti.field` and `ti.Struct` as argument types. Note that the restrictions applied to a kernel's arguments do not apply here:

- You are *not* required (but it is still recommended) to type hint arguments.
- You can have *an unlimited* number of elements in the arguments.


### Return values

The return values of a Taichi function can be scalars, `ti.Matrix`, `ti.Vector`, `ti.Struct`, or others. Note that:

- Unlike a kernel, a Taichi function can have multiple return values.
- You do not need (but it is still recommended) to type hint the return values of a Taichi function.

However, you still *cannot* have more than one `return` statement in a Taichi function.


###

## A recap: Taichi kernel vs. Taichi function

|                                                       | **Kernel**                          | **Taichi Function**                            |
| ----------------------------------------------------- | ----------------------------------- | ---------------------------------------------- |
| Call scope                                            | Python scope                        | Taichi scope                                   |
| Type hint arguments                                   | Required                            | Recommended                                    |
| Type hint return values                               | Required                            | Recommended                                    |
| Return type                                           | Scalar/`ti.Vector`/`ti.Matrix`      | Scalar/`ti.Vector`/`ti.Matrix`/`ti.Struct`/... |
| Maximum number of elements in arguments               | <ul><li>32 (for OpenGL)</li><li>64 (for others)</li></ul> | Unlimited                                      |
| Maximum number of return values in a return statement | 1                                   | Unlimited                                      |


## Key terms

#### Backend

In the computer world, the term *backend* may have different meanings based on the context, and generally refers to any part of a software program that users do not directly engage with. In the context of Taichi, backend is the place where your code is being executed, for example `cpu`, `opengl`, `cuda`, and `vulkan`.

#### Compile-time recursion

Compile-time recursion is a technique of meta-programming. The recursion is handled by Taichi's compiler and expanded and compiled into a serial function without recursion. The recursion conditions must be constant during compile time, and the depth of the recursion must be a constant.

#### Force inline

Force inline means that the users cannot choose whether to inline a function or not. The function will always be expanded into the caller by the compiler.

#### Metaprogramming

Metaprogramming generally refers to the manipulation of programs with programs. In the context of Taichi, it means generating actual-running programs with compile-time computations. In many cases, this allows developers to minimize the number of code lines to express a solution.

#### Runtime recursion

Runtime recursion is the kind of recursion that happens at runtime. The compiler does not expand the recursion, and it is compiled into a function that calls itself recursively. The recursion conditions are evaluated at runtime, and the depth does not need to be a constant number.

#### Type hint

Type hinting is a formal solution to statically indicate the type of value within your code.

## FAQ

#### Can I call a kernel from within a Taichi function?

No. Keep in mind that a kernel is the smallest unit for Taichi's runtime execution. You cannot call a kernel from within a Taichi function (in the Taichi scope). You can only call a kernel from the Python scope.
