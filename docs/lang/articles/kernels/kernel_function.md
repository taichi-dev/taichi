---
sidebar_position: 1
---

# Kernels and Functions

Taichi and Python have similar but *not exactly the same* syntax. To differentiate Taichi code from the native Python code, we use two decorators `@ti.kernel` and `@ti.func`:

- Functions decorated with `@ti.kernel` are *Taichi kernels* or *kernels* for short. They are the entry points where Taichi's runtime begins to take over the tasks, and *must* be called directly by the Python code. You can prepare your tasks, such as read data from the disk and preprocess them, in native Python and then call the kernels to let Taichi take over those computation-intensive tasks.

- Functions decorated with `@ti.func` are *Taichi functions*. They are the building blocks of kernels and can *only* be called by a kernel or another Taichi function. Just as you do with normal Python functions, you can split your tasks into multiple Taichi functions to improve readability and reuse them in different kernels.

In the following example, `inv_square()` is decorated with `@ti.func` and is a Taichi function; `partial_sum()` is decorated with `@ti.kernel` and is a kernel. The former (`inv_square()`) is called by the latter (`partial_sum()`). The argument and the return value in `partial_sum()` are type hinted, whilst those in the *Taichi function* `inv_square()` are not.

Here comes an important difference between Python and Taichi, type hinting:

- Type hinting in Python is recommended, *not* mandatory.
- Taichi makes it mandatory that you type hint the arguments and the return value of a kernel unless it does not have an argument or a return statement.

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
```

:::caution WARNING

Taichi raises a syntax error if you call a Taichi function from within the native Python code (the *Python scope*). For example:

:::

```python
import taichi as ti
ti.init(arch=ti.cpu)

@ti.func
def inv_square(x):
    return 1.0 / (x * x)

print(inv_square(1.0))  # Syntax error
```

You must call Taichi functions from within the Taichi scope, a concept as opposed to the *Python scope*.

:::tip IMPORTANT

For convenience, we introduce two concepts, *Taichi scope* and *Python scope*:

- The code inside a kernel or a Taichi function is in the *Taichi scope*. The code in the Taichi scope is compiled by Taichi's runtime and executed in parallel on multi-core CPU or GPU devices for high-performance computation. The Taichi scope corresponds to the device side in CUDA.

- Code outside of the Taichi scope is in the *Python scope*. The code in the Python scope is native Python and executed by Python's virtual machine, *not* by Taichi's runtime. The Python scope corresponds to the host side in CUDA.

:::

Do not confuse kernels with Taichi functions. They have slightly different syntax. The following sections describe their usages.


## Kernel

A kernel is the basic unit for execution in Taichi and the entry point from which Taichi's runtime takes over from Python's virtual machine. You call a kernel the same way as you call a Python function, and you can switch back and forth between Taichi's runtime and Python's virtual machine.

For example, you can call the kernel `partial_sum()` from inside a Python function:

```python {1,6,7}
@ti.kernel
def partial_sum(n: int) -> float:
    ...

def main():
    print(partial_sum(100))
    print(partial_sum(1000))

main()
```

You are allowed to define multiple kernels in your program. They are *independent* of each other and are compiled and executed in the same order as they are *first* called (the compiled kernels are stored in the cache to save the launch overhead for the subsequent calls).

:::caution WARNING

You call a kernel either directly or from inside a native Python function. You must *not* call a kernel from inside another kernel or from inside a Taichi function. To put it differently, you can only call a kernel from the Python scope.

:::


### Arguments


A kernel can take multiple arguments. Note that you *cannot* pass any arbitrary Python object to a kernel because Python objects can be highly dynamic and may hold data that Taichi's compiler cannot recognize.

The argument types that a kernel accepts are scalars, `ti.Matrix`, `ti.Vector` (vectors are essentially matrices), `ti.types.ndarray()`, and `ti.template()`, allowing you to easily pass data from the Python scope to the Taichi scope. The supported types are defined in the `ti.types` module (see the [Type System](../type_system/type.md) for more information).

- Scalars and `ti.Matrix` are *passed by value*.
- `ti.types.ndarray()` and `ti.template()` are passed by reference. This means that any modification to the arguments in the kernel being called also affects the original values.

> We skip `ti.template()` here and leave it to a more advanced topic: [Metaprogramming](../advanced/meta.md#template-metaprogramming).

In the following example, the arguments `x` and `y` are passed to `my_kernel()` *by value*:

```python {1}
@ti.kernel
def my_kernel(x: int, y: float):
    print(x + y)

my_kernel(1, 1.0)  # Prints 2.0
```

You can use `ti.types.ndarray()` as type hint to pass a NumPy's `ndarray` or a PyTorch's `tensor` to a kernel. Taichi recognizes the shape and data type of such a data structure and allows you to access these attributes in a kernel. In the following example, `x` is updated after `my_kernel()` is called because it is passed by reference.

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

A kernel can have *at most* one return value, which can be a scalar, `ti.Matrix`, or `ti.Vector`. Follow these rules when defining the return value of a kernel:

- Type hint the return value of a kernel.
- Ensure that you have *at most* one return value in a kernel.
- Ensure that you have *at most* one return statement in a kernel.
- Ensure that the number of elements in the return value does not exceed 30.

#### At most one return value

In the following code snippet, the kernel `test()` cannot have more than one return value:

```python
vec2 = ti.math.vec2

@ti.kernel
def test(x: float, y: float) -> vec2: # Return value must be type hinted
    # Return x, y  # Compilation error: Only one return value is allowed
    return vec2(x, y)  # Fine
```

#### Automatic type cast

In the following code snippet, the return value is automatically cast into the hinted type:

```python
@ti.kernel
def my_kernel() -> ti.i32:  # int32
    return 128.32
# The return value is cast into the hinted type ti.i32
print(my_kernel())  # 128
```

#### At most one return statement

In the following code snippet, Taichi raises an error because the kernel `test_sign()` has more than one return statement:

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

A kernel treats global variables as compile-time constants. This means that it takes in the current values of the global variables at the time it is compiled and that it does not track changes to them afterwards. Then, if the value of a global variable is updated between two calls of the same kernel, the second call does not take the updated value.

Let's take a look at the following example, where the global variable `a` is updated after the first call of `kernel_1`.

- The second call of `kernel_1` still prints `1`, because `kernel_1` does not track changes to `a` after it is compiled.
- `kernerl_2` takes in the current value of `a` and prints `2`, because  it is compiled after `a` is updated.

```python {15-17}
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

## Taichi function

Taichi functions are the building blocks of a kernel. *You must call a Taichi function from inside a kernel or from inside another Taichi function*.

In the following code snippet, Taichi raises an error because the Taichi function `foo_1()` must be called from the Taichi scope:

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

All Taichi functions are force-inlined. Therefore, no runtime recursion is allowed.

:::

### Arguments

A Taichi function can have multiple arguments, supporting scalar, `ti.Matrix`, `ti.Vector`, `ti.types.ndarray()`, `ti.template()`, `ti.field`, and `ti.Struct` as argument types. Note that some of the restrictions on a kernel's arguments do not apply here:

- It is *not* required (but still recommended) to type hint arguments.
- You can have an *unlimited* number of elements in the arguments.


### Return values

The return values of a Taichi function can be scalars, `ti.Matrix`, `ti.Vector`, `ti.Struct`, or others. Note that:

- Unlike a kernel, a Taichi function can have multiple return values.
- It is *not* required (but still recommended) to type hint the return values of a Taichi function.

Still, you *cannot* have more than one `return` statement in a Taichi function.

## A recap: Taichi kernel vs. Taichi function

|                                                       | **Kernel**                                                   | **Taichi Function**                                          |
| ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Call scope                                            | Python scope                                                 | Taichi scope                                                 |
| Type hint arguments                                   | Mandatory                                                    | Recommended                                                  |
| Type hint return values                               | Mandatory                                                    | Recommended                                                  |
| Return type                                           | <ul><li>Scalar</li><li>`ti.Vector`</li><li>`ti.Matrix`</li></ul> | <ul><li>Scalar</li><li>`ti.Vector`</li><li>`ti.Matrix`</li><li>`ti.Struct`</li><li>...</li></ul> |
| Maximum number of elements in arguments               | <ul><li>32 (OpenGL)</li><li>64 (otherwise)</li></ul>         | Unlimited                                                    |
| Maximum number of return values in a return statement | 1                                                            | Unlimited                                                    |


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
