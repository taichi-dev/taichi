---
sidebar_position: 1
---

# Kernels and functions

Taichi is a domain-specific language embedded in Python. The meaning of "embedded in Python" is twofold:  The code you've written in Taichi is valid Python code but is compiled and executed in Taichi's runtime; the rest of your code is treated as native Python code and executed in Python's virtual machine. 

To differentiate the code for Taichi from the code for Python, we use the two decorators `@ti.kernel` and `@ti.func`:

- Functions decorated with `@ti.kernel` are called kernels.
- Functions decorated with `@ti.func` are called Taichi functions. 

## Taichi scope vs. Python scope

We introduce the two terms "Taichi scope" and "Python scope" to make it easy to define the call range of a function or specify the effective range of a variable. So, before you proceed, familiarize yourself with these two terms. 

### Taichi scope

The code inside a kernel or a Taichi function is in the Taichi scope. The code in the Taichi scope is compiled by Taichi's runtime and executed in parallel on CPU or GPU devices for high-performance computation. 

> Taichi scope corresponds to the *device side* in CUDA.  

### Python scope

Code outside of the Taichi scope is in the Python scope. The code in the Python scope is native Python and executed by Python's virtual machine, *not* by Taichi's runtime.  

> Python scope corresponds to the *host side* in CUDA. 

## Kernel

A kernel is the entry point from which Taichi's runtime takes control and the smallest unit for runtime execution. You can define multiple kernels in your program, and each kernel is an *independent* child program. You call a kernel the same way you call a Python function, and you are allowed to switch back and forth between Taichi's runtime and Python's virtual machine.

Taichi's runtime compiles and executes kernels in the order you call them. It stores compiled kernels in the cache so that the next call to the same kernel does not need to be compiled again. 

:::caution WARNING
You must *not* call a kernel from inside another kernel or from inside a Taichi function. You can only call a kernel directly or from inside a native Python function. To put it differently, you can only call a kernel from the *Python scope*. 
:::

> A kernel corresponds to the `main()` function in C/C++, or the `__global__` function in CUDA. 

### Arguments

A kernel can take multiple arguments, supporting scalar, `ti.Matrix`, and `ti.Vector` as argument types. This makes it easier and more flexible to pass values from the Python scope to the Taichi scope. 

:::caution WARNING
Arguments in scalar, `ti.Matrix`, or `ti.Vector` are passed by value, so changes to the arguments of a kernel do not affect the original variables in the caller function. 
:::

Follow these rules when defining arguments:  

- Type hint kernel arguments.
- Ensure that the total number of elements in the kernel arguments does not exceed a certain upper limit (see below).

#### Type hint kernel arguments

```Python
@ti.kernel

def my_kernel(x: ti.i32, y: ti.f32):

    print(x + y)

my_kernel(24, 3.2)  # The system prints 27.2
```

#### Ensure that the total number of elements in the kernel arguments does not exceed a certain upper limit

The upper limit for element numbers is backend-specific: 

- 8 on OpenGL or CC backend. 
- 64 on CPU, Vulkan, CUDA, or Metal.

- > The number of elements in a scalar argument is always 1. 
- > The number of the elements in a `ti.Matrix` or in a `ti.Vector` is the actual number of scalars inside of them. 

```Python
@ti.kernel

def valid_scalar_argument(vx: ti.f32, vy: ti.f32):

    v = ti.Vector([vx, vy])

    ...



@ti.kernel

def valid_matrix_argument(u: ti.i32, v: ti.types.matrix(2, 2, ti.i32)):  # OK: has five elements in total

    ...



@ti.kernel

def error_too_many_arguments(u: ti.i32, v: ti.i64, w: ti.types.matrix(7, 9, ti.i64)):  # Error: has 65 elements in total

    ...
```

<details>

<summary><font color="#006284">Advanced arguments</font></summary>

*You can skip this part if you are just beginning.*

A kernel can also take the following two types of advanced arguments:

- Template arguments: Use `ti.template()` as their type hints. See [Template metaprogramming](../advanced/meta.md#template-metaprogramming). 
- External array arguments: Use `ti.ext_arr()` as their type hints. See [Interacting with external arrays](./external.md).

</details>

### Return value

A kernel can have *at most* one return value, which can be a scalar, `ti.Matrix`, and `ti.Vector`. Follow these rules when defining the return value of a kernel: 

- Type hint the return value of a kernel. 
- Ensure that you have *at most* one return value in a kernel. 
- Ensure that you have *at most* one return statement in a kernel. 
- Ensure that the number of elements in the return value does not exceed 30. 

#### Type hint the return value of a kernel

```Python
@ti.kernel

def test(x: ti.f32) -> ti.f32: # The return value is type hinted

    return 1.0
```

In addition, the return value is automatically cast into the hinted type:

```Python
@ti.kernel

def my_kernel() -> ti.i32:  # int32

    return 128.32

print(my_kernel())  # 128, the return value is cast into ti.i32
```

#### At most one return value in a kernel

```Python
@ti.kernel

def error_multiple_return() -> (ti.i32, ti.f32):

    x = 1
    y = 0.5
    return x, y  # Compilation error: more than one return value
```

#### At most one return statement in a kernel

```Python
@ti.kernel

def test_sign(x):

    if x >= 0:
        return 1.0
    else:
        return -1.0
    # Error: multiple return statements
```

As a workaround, you can save the result in a local variable and return it at the end:

```Python
@ti.kernel

def test_sign(x):

    sign = 1.0

    if x < 0:

        sign = -1.0

    return sign

    # One return statement works fine
```

#### At most 30 elements in the return value

```Python
N = 6

matN = ti.types.matrix(N, N, ti.i32)



@ti.kernel

def test_kernel() -> matN:

    return matN([[0] * N for _ in range(N)]) 

    # Compilation error: The number of elements  is 36 > 30
```

### Global variables are compile-time constants

A kernel treats global variables as compile-time constants. This means that it takes in the current values of the global variables at the time it is compiled and that it does not track changes to them afterwards. Then, if the value of a global variable is updated between two calls of the same kernel, the second call does not use the updated value.  

Let's take a look at the following example, where the global variable `a` is updated after the first call of `kernel_1`. 

- Because `kernel_1` does not track changes to `a` after it is compiled, the second call of `kernel_1` still prints `1`. 
- Because `kernerl_2` is compiled after `a` is updated, it takes in the current value of `a` and prints `2`.

```Python
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

Taichi functions are the building blocks of a kernel.  All Taichi functions are force-inlined. Therefore, no runtime recursion is allowed.

:::caution WARNING

You must call a Taichi function from inside a kernel or from inside another Taichi function. In other words, you must call a Taichi function from within the Taichi scope, *not* from within the Python scope. 

:::

> A Taichi function corresponds to the `__device__` function in CUDA. 

The following example shows the difference between a kernel and a Taichi function: 

```Python
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

A Taichi function can have multiple arguments, supporting scalar, `ti.Matrix`, and `ti.Vector` as argument types. Note that the restrictions applied to a kernel's arguments do not apply here:

- You are *not* required to type hint arguments.
- You can have *an unlimited* number of elements in the arguments. 

:::caution WARNING

Arguments in scalar, `ti.Matrix`, or `ti.Vector` are passed by value, so changes to the arguments of a Taichi function do not affect the original variables in the caller function. See the following example:

```Python
@ti.func

def my_func(x):

    x = x + 1  # Will not change the original value of x





@ti.kernel

def my_kernel():

    x = 24

    my_func(x)

    print(x)  # 24
```

:::

<details>

<summary><font color="#006284">Advanced arguments</font></summary>

*You can skip this part if you are just beginning.*

A kernel can also take the following two types of advanced arguments:

- Template arguments: Use `ti.template()` as their type hints. See [Template metaprogramming](../advanced/meta.md#template-metaprogramming). 

- External array arguments: Use `ti.ext_arr()` as their type hints. See [Interacting with external arrays](./external.md).

:::caution WARNING

By using `ti.template()` as type hint, you force arguments to pass by reference. Here's an example:

```Python
@ti.func

def my_func(x: ti.template()): # x is forced to pass by reference

    x = x + 1  # This line changes the original value of x





@ti.kernel

def my_kernel():

    ...

    x = 24

    my_func(x)

    print(x)  # The system prints 25

    ...
```

:::

</details>

### Return values

The return values of a Taichi function can be scalar, `ti.Matrix`, `ti.Vector`, `ti.Struct`, and more. Note that:

- Unlike a kernel, a Taichi function can have multiple return values. 

- You do not need to type hint the return values of a Taichi function.

- There is no limit on the number of elements in the return values. 

However, you *cannot* have more than one `return` statement in a Taichi function.

#### At most one return statement

You can only have one return statement in a Taichi function. 

```Python
@ti.func

def test_sign(x):

    if x >= 0:

        return 1.0

    else:

        return -1.0

    # Error: multiple return statements
```

As a workaround, you can save the result in a local variable and return it at the end:

```Python
@ti.func

def test_sign(x):

    sign = 1.0

    if x < 0:

        sign = -1.0

    return sign

    # One return statement works just fine
```

### 

## A recap: Taichi kernel vs. Taichi function

|                                                       | **Kernel**                          | **Taichi Function**                            |
| ----------------------------------------------------- | ----------------------------------- | ---------------------------------------------- |
| Call scope                                            | Python scope                        | Taichi scope                                   |
| Type hint arguments                                   | Required                            | Optional                                       |
| Type hint return values                               | Required                            | Optional                                       |
| Return type                                           | Scalar/`ti.Vector`/`ti.Matrix`      | Scalar/`ti.Vector`/`ti.Matrix`/`ti.Struct`/... |
| Maximum number of elements in arguments               | 8 (for OpenGL and CC) or 64 (other) | Unlimited                                      |
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
