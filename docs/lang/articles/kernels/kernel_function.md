---
sidebar_position: 1
---

# Kernels and Functions

Taichi and Python share a similar syntax, but they are not identical. To distinguish Taichi code from native Python code, we utilize two decorators, `@ti.kernel` and `@ti.func`:

- Functions decorated with `@ti.kernel` are known as *Taichi kernels* or simply *kernels*. These functions are the entry points where Taichi's runtime takes over the tasks, and they *must* be directly invoked by Python code. You can use native Python to prepare tasks, such as reading data from disk and pre-processing, before calling the kernel to offload computation-intensive tasks to Taichi.
- Functions decorated with `@ti.func` are known as *Taichi functions*. These functions are building blocks of kernels and can only be invoked by another Taichi function or a kernel. Like normal Python functions, you can divide your tasks into multiple Taichi functions to enhance readability and reuse them across different kernels.

In the following example, `inv_square()` is decorated with `@ti.func` and is a Taichi function. `partial_sum()` is decorated with `@ti.kernel` and is a kernel. The former (`inv_square()`) is called by the latter (`partial_sum()`). The arguments and return value in `partial_sum()` are type hinted, while those in the Taichi function `inv_square()` are not.

```python
import taichi as ti
ti.init(arch=ti.cpu)

@ti.func
def inv_square(x):  # A Taichi function
    return 1.0 / (x * x)

@ti.kernel
def partial_sum(n: int) -> float:  # A kernel
    total = 0.0
    for i in range(1, n + 1):
        total += inv_square(n)
    return total

partial_sum(1000)
```

Here comes a significant difference between Python and Taichi - *type hinting*:

- Type hinting in Python is recommended, but not compulsory.
- Taichi mandates that the arguments and return value of a kernel are type hinted, unless it has neither an argument nor a return statement.


:::caution WARNING

Calling a Taichi function from within the native Python code (the Python scope) results in a syntax error raised by Taichi. For example:

```python skip-ci:NotRunnable
import taichi as ti
ti.init(arch=ti.cpu)

@ti.func
def inv_square(x):
    return 1.0 / (x * x)

print(inv_square(1.0))  # Syntax error
```

You must call Taichi functions from within the Taichi scope, a concept as opposed to the *Python scope*.
:::

Let's introduce two important concepts: *Taichi scope* and *Python scope*.

- The code inside a kernel or a Taichi function is part of the *Taichi scope*. Taichi's runtime compiles and executes this code in parallel on multi-core CPU or GPU devices for high-performance computation. The Taichi scope corresponds to the device side in CUDA.

- Code outside of the Taichi scope belongs to the *Python scope*. This code is written in native Python and executed by Python's virtual machine, not by Taichi's runtime. The Python scope corresponds to the host side in CUDA.

It is important to distinguish between kernels and Taichi functions as they have slightly different syntax. The following sections explain their respective usages.

## Kernel

A kernel is the basic unit of execution in Taichi, and serves as the entry point for Taichi's runtime, which takes over from Python's virtual machine. Kernels are called in the same way as Python functions, and allow for switching between Taichi's runtime and Python's virtual machine.

For instance, the `partial_sum()` kernel can be called from within a Python function:

```python skip-ci:ToyDemo
@ti.kernel
def partial_sum(n: int) -> float:
    ...

def main():
    print(partial_sum(100))
    print(partial_sum(1000))

main()
```

Multiple kernels can be defined in a single Taichi program. These kernels are *independent* of each other, and are compiled and executed in the same order in which they are *first* called. The compiled kernels are cached to reduce the launch overhead for subsequent calls.

:::caution WARNING

Kernels in Taichi can be called either directly or from inside a native Python function. However, calling a kernel from inside another kernel or from inside a Taichi function is not allowed. In other words, kernels can only be called from the Python scope.

:::


### Arguments


A kernel can accept multiple arguments. However, it's important to note that you can't pass arbitrary Python objects to a kernel. This is because Python objects can be dynamic and may contain data that the Taichi compiler cannot recognize.

The kernel can accept various argument types, including scalars, `ti.types.matrix()`, `ti.types.vector()`, `ti.types.struct()`, `ti.types.ndarray()`, and `ti.template()`. These argument types make it easy to pass data from the Python scope to the Taichi scope. You can find the supported types in the `ti.types` module. For more information on this, see the [Type System](../type_system/type.md).

Scalars, `ti.types.matrix()`, `ti.types.vector()`, and `ti.types.struct()` are passed by value, which means that the kernel receives a copy of the argument. However, `ti.types.ndarray()` and `ti.template()` are passed by reference, which means that any changes made to the argument inside the kernel will affect the original value as well.

Note that we won't cover `ti.template()` here as it is a more advanced topic and is discussed in [Metaprogramming](../advanced/meta.md#template-metaprogramming).

Here is an example of passing arguments `x` and `y` to `my_kernel()` by value:

```python
@ti.kernel
def my_kernel(x: int, y: float):
    print(x + y)

my_kernel(1, 1.0)  # Prints 2.0
```

Here is another example of passing a nested struct argument with a matrix to a kernel by value, in which we created a struct type `transform_type` that contains two members: a rotation matrix `R` and a translation vector `T`. We then created another struct type `pos_type` that has `transform_type` as its member and passed an instance of `pos_type` to a kernel.

```python
transform_type = ti.types.struct(R=ti.math.mat3, T=ti.math.vec3)
pos_type = ti.types.struct(x=ti.math.vec3, trans=transform_type)

@ti.kernel
def kernel_with_nested_struct_arg(p: pos_type) -> ti.math.vec3:
    return p.trans.R @ p.x + p.trans.T

trans = transform_type(ti.math.mat3(1), [1, 1, 1])
p = pos_type(x=[1, 1, 1], trans=trans)
print(kernel_with_nested_struct_arg(p))  # [4., 4., 4.]
```

You can use `ti.types.ndarray()` as a type hint to pass a `ndarray` from NumPy or a `tensor` from PyTorch to a kernel. Taichi recognizes the shape and data type of these data structures, which allows you to access their attributes in a kernel.

In the example below, `x` is updated after `my_kernel()` is called since it is passed by reference:


```python {9,10,11}
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

@ti.kernel
def my_kernel(x: ti.types.ndarray(), y: ti.types.ndarray()):
    # Taichi recognizes the shape of the array x and allows you to access it in a kernel
    for i in range(x.shape[0]):
        x[i] += y[i]

my_kernel(x, y)
print(x)  # Prints [5, 7, 9]
```


### Return value

In Taichi, a kernel is allowed to have a maximum of one return value, which could either be a scalar, `ti.types.matrix()`, or `ti.types.vector()`.
Moreover, in the LLVM-based backends (CPU and CUDA backends), a return value could also be a `ti.types.struct()`.

Here is an example of a kernel that returns a ti.Struct:

```python
s0 = ti.types.struct(a=ti.math.vec3, b=ti.i16)
s1 = ti.types.struct(a=ti.f32, b=s0)

@ti.kernel
def foo() -> s1:
    return s1(a=1, b=s0(a=ti.math.vec3(100, 0.2, 3), b=1))

print(foo())  # {'a': 1.0, 'b': {'a': [100.0, 0.2, 3.0], 'b': 1}}
```

When defining the return value of a kernel in Taichi, it is important to follow these rules:

- Use type hint to specify the return value of a kernel.
- Make sure that you have at most one return value in a kernel.
- Make sure that you have at most one return statement in a kernel.
- If the return value is a vector or matrix, please ensure that it contains no more than 32 elements. In case it contains more than 32 elements, the kernel will still compile, but a warning will be raised.

#### At most one return value

In this code snippet, the `test()` kernel cannot have more than one return value:


```python skip-ci:ToyDemo
vec2 = ti.math.vec2

@ti.kernel
def test(x: float, y: float) -> vec2: # Return value must be type hinted
    # Return x, y  # Compilation error: Only one return value is allowed
    return vec2(x, y)  # Fine
```

#### Automatic type cast

In the following code snippet, the return value is automatically cast into the hinted type:

```python skip-ci:ToyDemo
@ti.kernel
def my_kernel() -> ti.i32:  # int32
    return 128.32
# The return value is cast into the hinted type ti.i32
print(my_kernel())  # 128
```

#### At most one return statement

In this code snippet, Taichi raises an error because the kernel `test_sign()` has more than one return statement:

```python skip-ci:ToyDemo
@ti.kernel
def test_sign(x: float) -> float:
    if x >= 0:
        return 1.0
    else:
        return -1.0
    # Error: multiple return statements
```

As a workaround, you can save the result in a local variable and return it at the end:

```python skip-ci:ToyDemo
@ti.kernel
def test_sign(x: float) -> float:
    sign = 1.0
    if x < 0:
        sign = -1.0
    return sign
    # One return statement works fine
```

### Global variables are compile-time constants

In Taichi, a kernel treats global variables as compile-time constants. This means that it takes in the current values of the global variables at the time it is compiled and does not track changes to them afterwards. Consider the following example:

```python skip-ci:ToyDemo
import taichi as ti
ti.init()

a = 1

@ti.kernel
def kernel_1():
    print(a)

@ti.kernel
def kernel_2():
    print(a)

kernel_1()  # Prints 1
a = 2
kernel_1()  # Prints 1
kernel_2()  # Prints 2
```

Here, `kernel_1` and `kernel_2` both access the global variable `a`. The first call to `kernel_1` prints 1, which is the value of `a` at the time the kernel was compiled. When `a` is updated to 2, the second call to `kernel_1` still prints 1 because the kernel does not track changes to a after it is compiled.

On the other hand, `kernel_2` is compiled after `a` is updated, so it takes in the current value of `a` and prints 2.

## Taichi function

Taichi functions are fundamental units of a kernel and can only be called from within a kernel or another Taichi function.

In the code snippet below, Taichi will raise an error because the function `foo_1()` is called from the Python scope, not the Taichi scope:

```python
# A normal Python function
def foo_py():
    print("This is a Python function.")

@ti.func
def foo_1():
    print("This is a Taichi function to be called by another Taichi function, foo_2().")

@ti.func
def foo_2():
    print("This is a Taichi function to be called by a kernel.")
    foo_1()

@ti.kernel
def foo_kernel():
    print("This is a kernel calling a Taichi function, foo_2().")
    foo_2()

foo_py()
# foo_1() # You cannot call a Taichi function from the Python scope
foo_kernel()
```

:::caution WARNING

All Taichi functions are force-inlined. This means that if you call a Taichi function from another Taichi function, the calling function is fully expanded, or inlined, into the called function at compile time. This process continues until there are no more function calls to inline, resulting in a single, large function. This means that runtime recursion is *not allowed* in Taichi, because it would cause an infinite expansion of the function call stack at compile time.

:::

### Arguments

A Taichi function can accept multiple arguments, which may include scalar, `ti.types.matrix()`, `ti.types.vector()`, `ti.types.struct()`, `ti.types.ndarray()`, `ti.field()`, and `ti.template()` types. Note that some of the restrictions on kernel arguments do not apply to Taichi functions:

- It is not strictly required to type hint the function arguments (but it is still recommended).
- You can pass an unlimited number of elements in the function arguments.


### Return values

Return values of a Taichi function can be scalars, `ti.types.matrix()`, `ti.types.vector()`, `ti.types.struct()`, or other types. Note the following:

- Unlike a kernel, a Taichi function can have multiple return values.
- It is *not* required (but recommended) to type hint the return values of a Taichi function.
- A Taichi function *cannot* have more than one `return` statement.

## A recap: Taichi kernel vs. Taichi function

|                                                       | **Kernel**                                                                                                        | **Taichi Function**                                          |
| ----------------------------------------------------- |-------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------ |
| Call scope                                            | Python scope                                                                                                      | Taichi scope                                                 |
| Type hint arguments                                   | Mandatory                                                                                                         | Recommended                                                  |
| Type hint return values                               | Mandatory                                                                                                         | Recommended                                                  |
| Return type                                           | <ul><li>Scalar</li><li>`ti.types.matrix()`</li><li>`ti.types.vector()`</li><li>`ti.types.struct()`(Only on LLVM-based backends)</li></ul> | <ul><li>Scalar</li><li>`ti.types.matrix()`</li><li>`ti.types.vector()`</li><li>`ti.types.struct()`</li><li>...</li></ul> |
| Maximum number of elements in arguments               | <ul><li>32 (OpenGL)</li><li>64 (otherwise)</li></ul>                                                              | Unlimited                                                    |
| Maximum number of return values in a return statement | 1                                                                                                                 | Unlimited                                                    |


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

No. Keep in mind that a kernel is the smallest unit for Taichi's runtime execution. You cannot call a kernel from within a Taichi function (in the Taichi scope). You can *only* call a kernel from the Python scope.
