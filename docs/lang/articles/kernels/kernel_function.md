---
sidebar_position: 1
---
:::note IMPORTANT

If new to programming and/or Python, it is best to recap what [decorators](https://python101.pythonlibrary.org/chapter25_decorators.html) are.

:::


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

:::note IMPORTANT

For convenience, we introduce two concepts, *Taichi scope* and *Python scope*:

- The code inside a kernel or a Taichi function is in the *Taichi scope*. The code in the Taichi scope is compiled by Taichi's runtime and executed in parallel on multi-core CPU or GPU devices for high-performance computation. The Taichi scope corresponds to the device side in CUDA.

- Code outside of the Taichi scope is in the *Python scope*. The code in the Python scope is native Python and executed by Python's virtual machine, *not* by Taichi's runtime. The Python scope corresponds to the host side in CUDA.

:::

:::note WARNING

Do not confuse kernels with Taichi functions. They have slightly different syntax. The following sections describe their usages:

:::

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


A kernel can take multiple arguments. Note that you *cannot* pass any arbitrary Python object to a kernel because Python objects can be highly dynamic and may hold data that Taichi's compiler cannot recognize. Here are some common arguments that allow you to easily pass data from the Python scope to the Taichi scope.


:::note IMPORTANT

- We skip `ti.template()` here and leave it to a more advanced topic: [Metaprogramming](../advanced/meta.md#template-metaprogramming).

- The supported types are defined in the `ti.types` module (see the [Type System](../type_system/type.md) for more information).

:::

#### Primative Types
Primative Types are predefined by the python language. This includes types such as Integers, Floats, and Booleans.

In the following example, the arguments `x` and `y` are passed to `myKernel()` *by value*:

```python {1}
@ti.kernel
def myKernel(x: int, y: float):
    print(x + y)

myKernel(1, 1.0)  # Prints 2.0
```

#### Vectors/Matrices
In addition to primitive types, we can also pass vectors and matrices into the kernel function. Vectors are represented as 1D arrays and Matrices are represented as 2D rectangular arrays.

In the following example, we pass a 2x3 matrix `arr` into `myKernel()` *by value*:
```python {13,17}
import taichi as ti
ti.init()

mat2x3 = ti.types.matrix(2, 3, float)

@ti.kernel
def myKernel(m: mat2x3) -> float:
    # We simply return the element m[1][1]
    return m[1][1]

m = mat2x3([2,4,6], [8,10,12])
print(myKernel(m))  #will return 10

#### Matrix Fields
We can use the ti.Matrix() function to declare our own [Matrix Field](https://docs.taichi-lang.org/docs/master/field#matrix-fields) and pass it into the kernel. This also works for [Vector Fields](https://docs.taichi-lang.org/docs/master/field#vector-fields)

In the following example, we declare a Matrix Field, generate its values, and pass it into `myKernel()`:

```python
import taichi as ti
ti.init()


@ti.kernel
def myKernel(a: ti.Matrix.field):
    for i in ti.grouped(a):
        a[i] = [[1,1,1], [1,1,1]]

a = ti.Matrix.field(n=2, m=3, dtype=ti.f32, shape=(2, 2))  #Declares a 2x2 matrix field, with each of its elements being a 3x2 matrix
myKernel(a)
print(a[0][0,0])  #prints 1

```



#### ti.types.ndarray()

You can use `ti.types.ndarray()` as type hint to pass a NumPy's `ndarray` or a PyTorch's `tensor` to a kernel. Taichi recognizes the shape and data type of such a data structure and allows you to access these attributes in a kernel. In the following example, `x` is updated after `myKernel()` is called because it is passed by reference.

```python {9,10,11}
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

@ti.kernel
def myKernel(x: ti.types.ndarray(), y: ti.types.ndarray()):
    # Taichi recognizes the shape of the array x and allows you to access it in a kernel
    for i in range(x.shape[0]):
        x[i] += y[i]

myKernel(x, y)
print(x)  # Prints [5, 7, 9]
```


### Return value

A kernel can have *at most* one return value, which can be a scalar, `ti.Matrix`, or `ti.Vector`. Follow these rules when defining the return value of a kernel:

#### At most one return value

In the following code snippet, the kernel `test()` cannot have more than one return value:

```python
vec2 = ti.math.vec2

@ti.kernel
def test(x: float, y: float) -> vec2: # Return value must be type hinted

    # Return x, y  # Compilation error: Only one return value is allowed
    return vec2(x, y)  # Fine
```

#### Automatic type casting

In the following code snippet, the return value is automatically cast into the hinted type:

```python
@ti.kernel
def myKernel() -> ti.i32:  # int32
    return 128.32

# The return value is cast into the hinted type ti.i32
print(myKernel())  # 128
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

To fix this, we recommend that you can save the result in a local variable and return it at the end:

```python
@ti.kernel
def test_sign(x: float) -> float:
    sign = 1.0
    if x < 0:
        sign = -1.0
    return sign
    # One return statement works fine
```


### Global Variables

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


## Taichi Function

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

A Taichi function can have multiple arguments, supporting [Scalar](https://docs.taichi-lang.org/docs/master/field#scalar-fields), [ti.Matrix](https://docs.taichi-lang.org/docs/master/field#matrix-fields), [ti.Vector](https://docs.taichi-lang.org/docs/master/field#vector-fields), `ti.types.ndarray()`, `ti.template()`, and [ti.Struct](https://docs.taichi-lang.org/docs/master/field#struct-fields) as argument types. Note that some of the restrictions on a kernel's arguments do not apply here:

- It is *not* required (but still recommended) to type hint arguments.
- You can have an *unlimited* number of elements in the arguments.


### Return values

The return values of a Taichi function can be scalars, `ti.Matrix`, `ti.Vector`, `ti.Struct`, or others. Note that:

- Unlike a kernel, a Taichi function can have multiple return values.
- It is recommended to type hint the return values of a Taichi function.

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


## FAQ

#### Can I call a kernel from within a Taichi function?

No. Keep in mind that a kernel is the smallest unit for Taichi's runtime execution. You cannot call a kernel from within a Taichi function (in the Taichi scope). You can *only* call a kernel from the Python scope.
