---
sidebar_position: 1
---

# Metaprogramming


## What is metaprogramming?

> Metaprogramming is a programming technique in which computer programs have the ability to treat other programs as their data. It means that a program can be designed to read, generate, analyze or transform other programs, and even modify itself while running.
>
> from Wikipedia: https://en.wikipedia.org/wiki/Metaprogramming.

To put it shortly: A metaprogram is a program that writes (or modifies) programs.

That sounds good, but you might wonder, code writes code, who needs that? Why don't I write all the code myself? Let's take a practical example to illustrate its use.

In simulations and graphics, a 4D vector `v` is composed by 4 components `x`, `y`, `z`, `w`. One can use `v.x` to access its first component, `v.y` to access its second component, and so on. It would be very handy if we can use any combination of the letters `x`, `y`, `z`, `w` to access the components of `v`, and returns a vector that matches this pattern. This is called swizzling (for reader with some knowledge of OpenGL shading language, or have experienc in Blender's scripting mode, this should be a familiar concept). For example, we want `v.xy` to return a 2D vector `[v.x, v.y]`, `v.zyx` to return a 3D vector `[v.z, v.y, v.x]`, `v.xxxx` to return a 4D vector `[v.x, v.x, v.x, v.x]`, and so on for other patterns.

But how do we implement this in a 4D vector class? There are 4 possible combinations consist of a single letter, 16 combinations for two letters, 64 combinations for three letters, and 256 combinations for four letters. The total is 4 + 16 + 64 + 256 = 340! You won't want to manually list them out one by one, that would cost a lot of labor, and the code would be be too cubersome! Well, as a scripting language, Python offer great functionality for metaprogramming. It turns out you can use the magic method `__getattr__` to intercept calls to an undefined property, and use `__setattr__` to set it! In other words, that property did not exist before you called it! To be more precise, let's say we are calling the non-existent `v.xxx` property of our 4D vector class, in `__getattr__` we can parsed its name as a string "xxx", it knows that you are trying to get a 3D vector of repeated components `v.x`. Therefore, it then checks to see if it can find a property with the name "xxx", and if it cannot, it calls `__setattr__` which writes a property that constructs that query for you, define it on the 4D vector class, and finally returns the result! Now, every time you call `v.xxx` on the instance `v`, the newly defined property gets called instead of going through that whole process every time!

To summarize, the benefits of metaprogramming are: It reduces repetition of the code; It makes the code more readable.


## Metaprogramming in Taichi

Taichi is a static and compile language. After Taichi's JIT finishes the compiling, all the control flow and variable types are immutable. It's not that obvious how one could do metaprogramming in Taichi. But Taichi does provide a few metaprogramming features, as listed below.

- Template metaprogramming. This enables the development of dimensionality-independent code, e.g., code which is adaptive for both 2D/3D physical simulations.
- Compile-time evaluations. This improves runtime performance by moving computations from runtime to compile time.
- Simplifying the development of Taichi standard library.

We will discuss them in more detail in later sections.

## Template metaprogramming

Template metaprogramming is a well-known concept to C++ developers. Let's quickly review what it is.

Assume you are going to write a function `sum`, which takes in an array-like object whose entries are all floating numbers, and returns the sum of all the entries. The array-like object might be a `std::vector`, `std::pair`, or even an user-defined array which you won't know the type. The best practice is not to implement the same function for all possible types. Instead, you use template programming:

```C++
<template T>
float sum(T &arr) {
    float result = 0.0;
    for (int i=0; i<arr.length(); i++) result += arr[i];
    return result;
}
```

When this function is called in the program, maybe in different places and operates on different array-like types `T`, the compiler will generate a version of `sum` for each `T`, as long as `T` implements the `length` method to allow you get the array length, and can be intrated over through indices. In other words, with template programming, you only write the same code once and the compiler automatically generates other versions for you.

Taichi has a counterpart functionality for template programming: By using `ti.template()` as an argument type hint, you can pass any Python object that Taichi's JIT compiler accepts into a kernel (see [../kernels/kernel_function#arguments]).

Let's write a function `sum` (our old friend) to illustrate this. This `sum` function will take in a Taichi field and return the sum of all its entries.

```python {2}
@ti.kernel
def sum(x: ti.template()) -> float:
    result = 0.0
    for i in ti.grouped(x):
        result += x[i]

f1d = ti.field(float, shape=10)
f2d = ti.field(float, shape=(10, 10))
f3d = ti.field(float, shape=(10, 10, 10))
g3d = ti.field(int, shape=(10, 10, 10))

sum(f1d)
sum(f2d)
sum(f3d)
sum(g3d)
```

As can be seen from the code above, you won't need to bother about the shape of the field, as the code works for fields of any shape. This is very handy for physical simulations as the same function can be used in both 2D and 3D scenatios.

Note the function `ti.group()` is critial to our dimensionality-independent programming: In general, to loop over a field of dimension `d`, you will need `d` independent indices, one for each axis. Taichi's `ti.grouped` puts the `d` loop indices into a `d`-dimentional index of integer vector type, and the `for` loop is parallelized for all such vector-type indices.

We should mention a difference between Taichi and C++ in template metaprogramming: C++ compilers will generate a version of `sum` for each different type `T`; meanwhile Taichi's compiler **recompiles** the kernel each time it finds an argument of a different type is encounted. In the example above, since the fields are of different shapes, or the same shape but of different dtypes, each of the four calls to `sum` will trigger a compilation:

```python
sum(f1d)  # Compilation
sum(f2d)  # Recompilation
sum(f3d)  # Recompilation
sum(g3d)  # Recompilation
```

:::note
If a template parameter is not a Taichi object, it cannot be reassigned inside Taichi kernel. For example:

```python
x = [1, 2, 3]
@ti.kernel
def error_reassign(x: ti.template()):
    x = ti.math.vec3(1, 2, 3)  # Error!
```

:::

:::note
The template parameters are inlined into the generated kernel after compilation.
:::


## Compile-time evaluations

Using compile-time evaluation allows for some computation to be executed when kernels are instantiated. This helps the compiler to conduct optimization and reduce
computational overhead at runtime:

### Static Scope
`ti.static` is a function which receives one argument. It is a hint for the compiler to evaluate the argument at compile time.
The scope of the argument of `ti.static` is called static-scope.

### Compile-time branching

- Use `ti.static` for compile-time branching (for those who are familiar with
  C++17, this is similar to [if
  constexpr](https://en.cppreference.com/w/cpp/language/if).):

```python {5}
enable_projection = True

@ti.kernel
def static():
  if ti.static(enable_projection): # No runtime overhead
    x[0] = 1
```

:::note
One of the two branches of the `static if` will be discarded after compilation.
:::

### Loop unrolling

- Use `ti.static` for forced loop unrolling:

```python {3}
@ti.kernel
def func():
  for i in ti.static(range(4)):
      print(i)

  # The code snippet above is equivalent to:
  print(0)
  print(1)
  print(2)
  print(3)
```

## When to use `ti.static` with for loops

There are two reasons to use `ti.static` with for loops:

- Loop unrolling for improving runtime performance (see [Compile-time evaluations](#compile-time-evaluations)).
- Accessing elements of Taichi matrices/vectors. Indices for accessing Taichi fields can be runtime variables, while indices for Taichi matrices/vectors **must be a compile-time constant**.

For example, when accessing a vector field `x` with `x[field_index][vector_component_index]`, the `field_index` can be a runtime variable, while the `vector_component_index` must be a compile-time constant:

```python {6}
# Here we declare a field contains 3 vector. Each vector contains 8 elements.
x = ti.Vector.field(8, ti.f32, shape=(3))
@ti.kernel
def reset():
  for i in x:
    for j in ti.static(range(x.n)):
      # The inner loop must be unrolled since j is an index for accessing a vector
      x[i][j] = 0
```

## Compile-time recursion of `ti.func`

A compile-time recursive function is a function with recursion that can be recursively inlined at compile time. The condition which determines whether to recurse is evaluated at compile time.

You can combine [compile-time branching](#compile-time-branching) and [template](#template-metaprogramming) to write compile-time recursive functions.

For example, `sum_from_one_to` is a compile-time recursive function that calculates the sum of numbers from `1` to `n`.

```python {1-6}
@ti.func
def sum_from_one_to(n: ti.template()) -> ti.i32:
    ret = 0
    if ti.static(n > 0):
        ret = n + sum_from_one_to(n - 1)
    return ret

@ti.kernel
def sum_from_one_to_ten():
    print(sum_from_one_to(10))  # prints 55
```

:::caution WARNING
When the recursion is too deep, it is not recommended to use compile-time recursion because deeper compile-time recursion expands to longer code during compilation, resulting in increased compilation time.
:::
