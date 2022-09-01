---
sidebar_position: 1
---

# Kernels and functions

Taichi is embedded in Python, the code you have written for Taichi is also valid Python code, hence we need a way to differentiate the code for Taichi from the code for Python. We use the two decorators `@ti.kernel` and `@ti.func` for this:

+ Functions decorated with `@ti.kernel` are called kernels.
+ Functions decorated with `@ti.func` are called Taichi functions.

Let's see an example:

```python
import taichi as ti
ti.init(arch=ti.cpu)

@ti.func
def inv_square(x):
    return 1.0 / (x * x)

@ti.kernel
def partial_sum(n: int) -> float:  # sum 1/i**2 from 1 to n
    total = 0.0
    for i in range(1, n + 1):
        total += inv_square(n)
    return total
```

In the above code, the function `inv_square` is a Taichi function since it's decorated by `@ti.func`, while the function `partial_sum` is a kernel since it's decorated by `@ti.kernel`. The Taichi function `inv_square` is called by the kernel `partial_sum`.

You may have noticed that the argument and return in the kernel `partial_sum` are both type hinted, while those in the Taichi function `inv_square` are not. In native Python, type hinting is a suggested but not mandatory syntax, but in the Taichi language *this is a mandatory syntax*: you must add type hints for arguments and returns of a kernel. When there are no arguments (or returns) in a kernel, the corresponding type hinting can be omitted.

Another point worth mentioning is, if you try to call `inv_square` out of the Taichi scope, Taichi will raise a syntax error. For example:

```python
import taichi as ti
ti.init(arch=ti.cpu)

@ti.func
def inv_square(x):
    return 1.0 / (x * x)

print(inv_square(1.0))  # syntax error!
```

Here by "out of the Taichi scope" we mean the scope that is not inside a kernel nor a Taichi function.

As you have seen, there are a few differences between kernels and Taichi functions. We now explain the roles these two kinds of functions play in a program and their differences in more detail.


## Kernel

A kernel is the entry point from which Taichi's runtime takes control and the smallest unit for runtime execution. You can define multiple kernels in your program, and these kernels are *independent* from each other. You call a kernel the same way you call a Python function, and you are allowed to switch back and forth between Taichi's runtime and Python's virtual machine.

For example you can call our kernel function `partial_sum` in the above section from a Python function:

```python
def main():
    print(partial_sum(100))
    print(partial_sum(1000))

main()
```

When there are more than one kernel in a program, Taichi will compile and execute them in the order they are called. The compiled kernels are stored in the cache so that subsequent calls won't invoke re-compilations.

:::caution WARNING

You must *not* call a kernel from inside another kernel or from inside a Taichi function. You can only call a kernel directly or from inside a native Python function.

:::


### Arguments


A kernel can take multiple arguments, but unlike in native Python, you cannot pass an arbitrary Python object to a kernel. This is because Python objects can be highly dynamic and hold data and resources only known to the Python interpreter.

Kernels support scalar, `ti.Matrix/ti.Vector` (In Taichi vectors are essentially matrices), `ti.types.ndarray()` and `ti.template()` as argument types. This allows you to pass data from the Python scope to the Taichi scope.

Arguments of type scalar or `ti.Matrix` are passed by value, while arguments of type `ti.types.ndarray()` and `ti.template()` are passed by reference, modifying to the arguments will also affect the original values.

In the following example, `x, y` are passed to `my_kernel` by values:

```python
@ti.kernel
def my_kernel(x: int, y: float):
    print(x + y)

my_kernel(1, 1.0)  # prints 4.0
```

You can also pass a NumPy's `ndarray` or a Pytorch's `tensor` as an argument to a kernel using `ti.types.ndarray()` as the type hint, Taichi knows its shape and data type and use these attributes. For example:

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

The array `x` is modified by `my_kernel` since it's passed by reference.

:::note

We skip the discussion of `ti.template()` as type hints here, it's related to meta-programming and is a bit advanced topic for this stage, see [here](../advanced/meta.md#template-metaprogramming) for more infomation.

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
def test(x: float, y: float) -> vec2: # return value must be type hinted
    #return x, y  # compilation error! only one return value is allowed
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
#foo_func() # You cannot call a taichi function from within the python scope
foo_kernel()
```

### Arguments

A Taichi function can have multiple arguments, supporting scalar, `ti.Matrix/ti.Vector`, `ti.types.ndarray()`, `ti.template()`, `ti.field` and `ti.Struct` as argument types. Note that the restrictions applied to a kernel's arguments do not apply here:

- You are *not* required (but it is still recommended) to type hint arguments.
- You can have *an unlimited* number of elements in the arguments.


### Return values

The return values of a Taichi function can be scalar, `ti.Matrix`, `ti.Vector`, `ti.Struct`, and more. Note that:

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
| Maximum number of elements in arguments               | <ul><li>8 (for OpenGL)</li><li>64 (for others)</li></ul> | Unlimited                                      |
| Maximum number of return values in a return statement | 1                                   | Unlimited                                      |
| Maximum number of elements in return values           | 30                                  | Unlimited                                      |

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
