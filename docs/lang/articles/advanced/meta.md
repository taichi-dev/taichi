---
sidebar_position: 1
---

# Metaprogramming

Taichi provides metaprogramming infrastructures. There are many benefits of metaprogramming in Taichi:

- Enabling the development of dimensionality-independent code, e.g., code which is
  adaptive for both 2D/3D physical simulations.
- Improving runtime performance by moving computations from runtime to compile time.
- Simplifying the development of Taichi standard library.

:::note
Taichi kernels are **lazily instantiated** and large amounts of computation can be executed at **compile-time**.
Every kernel in Taichi is a template kernel, even if it has no template arguments.
:::

## Template metaprogramming

By using `ti.template()` as an argument type hint, a Taichi field or a python object can be passed into a kernel. Template programming also enables the code to be reused for fields with different shapes:

```python {2}
@ti.kernel
def copy_1D(x: ti.template(), y: ti.template()):
    for i in x:
        y[i] = x[i]

a = ti.field(ti.f32, 4)
b = ti.field(ti.f32, 4)
c = ti.field(ti.f32, 12)
d = ti.field(ti.f32, 12)

# Pass field a and b as arguments of the kernel `copy_1D`:
copy_1D(a, b)

# Reuse the kernel for field c and d:
copy_1D(c, d)
```

:::note
The template parameters are inlined into the generated kernel after compilation.
:::

## Dimensionality-independent programming using grouped indices

Taichi provides `ti.grouped` syntax which supports grouping loop indices into a `ti.Vector`.
It enables dimensionality-independent programming, i.e., code are adaptive to scenarios of
different dimensionalities automatically:

```python {2,7,12,18}
@ti.kernel
def copy_1D(x: ti.template(), y: ti.template()):
    for i in x:
        y[i] = x[i]

@ti.kernel
def copy_2d(x: ti.template(), y: ti.template()):
    for i, j in x:
        y[i, j] = x[i, j]

@ti.kernel
def copy_3d(x: ti.template(), y: ti.template()):
    for i, j, k in x:
        y[i, j, k] = x[i, j, k]

# Kernels listed above can be unified into one kernel using `ti.grouped`:
@ti.kernel
def copy(x: ti.template(), y: ti.template()):
    for I in ti.grouped(y):
        # I is a vector with dimensionality same to y
        # If y is 0D, then I = ti.Vector([]), which is equivalent to `None` used in x[I]
        # If y is 1D, then I = ti.Vector([i])
        # If y is 2D, then I = ti.Vector([i, j])
        # If y is 3D, then I = ti.Vector([i, j, k])
        # ...
        x[I] = y[I]
```

## Field metadata

The two attributes **data type** and **shape** of fields can be accessed by `field.dtype` and  `field.shape`, in both Taichi-scope and Python-scope:

```python {3,7}
x = ti.field(dtype=ti.f32, shape=(3, 3))

# Print field metadata in Python-scope
print("Field dimensionality is ", x.shape)
print("Field data type is ", x.dtype)

# Print field metadata in Taichi-scope
@ti.kernel
def print_field_metadata(x: ti.template()):
    print("Field dimensionality is ", len(x.shape))
    for i in ti.static(range(len(x.shape))):
        print("Size along dimension ", i, "is", x.shape[i])
    ti.static_print("Field data type is ", x.dtype)
```

:::note
For sparse fields, the full domain shape will be returned.
:::

## Matrix & vector metadata

For matrices, `matrix.m` and `matrix.n` returns the number of columns and rows, respectively.
For vectors, they are treated as matrices with one column in Taichi, where `vector.n` is the number of elements of the vector.

```python {4-5,7-8}
@ti.kernel
def foo():
    matrix = ti.Matrix([[1, 2], [3, 4], [5, 6]])
    print(matrix.n)  # number of row: 3
    print(matrix.m)  # number of column: 2
    vector = ti.Vector([7, 8, 9])
    print(vector.n)  # number of elements: 3
    print(vector.m)  # always equals to 1 for a vector
```

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

- Loop unrolling for improving runtime performance (see section [Compile-time evaluations](##Compile-time evaluations)).
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

You can combine [compile-time branching](#compile-time-evaluations) and [template](#template-metaprogramming) to write compile-time recursive functions.

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
