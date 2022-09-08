---
sidebar_position: 1

---

# Type system


Taichi is a statically typed programming language. The type of a variable in the Taichi scope is determined at compile time. After a variable is declared, you cannot assign to it a value of a different type.

Let's see a quick example:

```python
@ti.kernel
def test():
    x = 1  # x is the integer 1
    x = 3.14  # 3.14 is cast to 3 since x is an integer, so x takes the value 3
    x = ti.Vector([1, 1])  # Error!
```

Line 3 indicates that `x` should be an integer since it is assigned by an integer value 1 upon its declaration. In Line 4, we re-assign a floating-point number 3.14 to `x`. This scalar will be automatically cast to integer 3 to match the type of `x`; hence, `x` takes the value 3 after this line. Line 5 will raise a syntax error when we try to assign a `ti.Vector` to `x` because `ti.Vector` is a different data type, which cannot be cast into an integer.

We can briefly summarize the data types supported by Taichi as follows:

Taichi supports two categories of data types in the [Taichi scope](../kernels/syntax.md#taichi-scope): primitive types and compound types.

- Primitive types: Various commonly used numerical data types, such as `ti.i32` (`int32`), `ti.u8` (`uint8`), and `ti.f64` (`float64`).
- Compound types: Array-like or struct-like data types, including `ti.types.matrix`, `ti.types.struct`, and `ti.types.ndarray`, which comprise multiple members of primitive types or other compound types.

More details will be provided in the following sections.

## Primitive types

Primitive types refer to scalars, which are the smallest building blocks of compound types. Each primitive type is denoted with a character indicating its category followed by a number indicating its precision bits (number of bits for storing the data). The _category_ can be `i` (signed integers), `u` (unsigned integers), or `f` (floating-point numbers); the _precision bits_ can be `8`, `16`, `32`, or `64`. Following are the two most commonly used types:

- `i32`: 32-bit signed integer
- `f32` : 32-bit floating-point number.

Not all primitive types are supported across all backends. Check out the following table for the supported types on various backends. Note that some backends may require extensions to support a specific primitive type.

| Backend | `i8`               | `i16`              | `i32`              | `i64`              | `u8`                 | `u16`                | `u32`                | `u64`                | `f16`                | `f32`                | `f64`                |
| ------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| CPU     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CUDA    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| OpenGL  | :x:                | :x:                | :heavy_check_mark: | :o:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: |
| Metal   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                |
| Vulkan  | :o:                | :o:                | :heavy_check_mark: | :o:                | :o:                | :o:                | :heavy_check_mark: | :o:                | :heavy_check_mark: | :heavy_check_mark: | :o:                |

> :o:: Requiring extensions for the backend.

### Customize the default primitive types for integers and floating-point numbers

When you call the `ti.init()` method to initialize the Taichi runtime, Taichi automatically applies the default primitive data types:

- The default integer type in Taichi is `ti.i32`.
- The default floating-point type in Taichi is `ti.f32`.

You can customize the default setting by explicitly specifying the primitive data type(s) you want to use in the `init()` call:

```python
ti.init(default_ip=ti.i64)  # Set the default integer type to ti.i64
ti.init(default_fp=ti.f64)  # Set the default floating-point type to ti.f64
```

It should be noted that the numeric literals in the Taichi scope also have default integer/floating-point types. For example, if the default floating-point type is `ti.f32`, then a numeric literal `3.14159265358979` in the Taichi scope will be cast into a 32-bit floating-point number, hence only accurate up to about seven decimal places. If you work on high-precision workloads such as numeric simulations for engineering, use `ti.f64` as the `default_fp`.

### Use `int` and `float` as aliases for default primitive types

Taichi supports using `int` as an alias for the default integer type and `float` as an alias for the default floating-point type. For example, after changing the default primitive types to `i64` and `f64` when initializing Taichi, you can use `int` as an alias for `i64` and `float` as an alias for `f64`.

```python
ti.init(default_ip=ti.i64, default_fp=ti.f64)

x = ti.field(float, 5)
y = ti.field(int, 5)
# is equivalent to:
x = ti.field(ti.f64, 5)
y = ti.field(ti.i64, 5)

def func(a: float) -> int:
    ...
# is equivalent to:
def func(a: ti.f64) -> ti.i64:
    ...
```

### Explicit type casting

As mentioned at the beginning of this article, the type of a variable in the Taichi scope is *statically typed* upon initialization. Taichi's compiler performs type checking at compile time, which means you *cannot* change a variable's type once it is initialized. However, from time to time, you may run into a situation where you need to switch to a different data type because the original one is not feasible for an assignment or calculation. In such cases, you need explicit type casting:

- You can use `ti.cast()` to convert a value to the target type:

  ```python
  @ti.kernel
  def foo():
      a = 3.14
      b = ti.cast(a, ti.i32)  # 3
      c = ti.cast(b, ti.f32)  # 3.0
  ```

- As of v1.1.0, you are allowed to use primitive types such as `ti.f32` and `ti.i64` to convert a scalar variable to a different scalar type:

  ```python
  @ti.kernel
  def foo():
      a = 3.14
      x = int(a)    # 3
      y = float(a)  # 3.14
      z = ti.i32(a)  # 3
      w = ti.f64(a)  # 3.14
  ```

### Implicit type casting

Implicit type casting occurs when you *accidentally* put or assign a value in a place where a different data type is expected.

:::caution WARNING
As a rule of thumb, implicit type casting is a major source of bugs. And Taichi does *not* recommend resorting to this mechanism.

:::

Implicit type casting can happen in binary operations or in assignments, as explained below.

#### Implicit type casting in binary operations

Taichi implements its own implicit type casting rules for binary operations, which are slightly different from [those for the C programming language](https://en.cppreference.com/w/c/language/conversion). In general we have three rules in descending order of priority:

1. Integer + floating point -> floating point
   - `i32 + f32 -> f32`
   - `i16 + f16 -> f16`

2. Low-precision bits + high-precision bits -> high-precision bits
   - `i16 + i32 -> i32`
   - `f16 + f32 -> f32`
   - `u8 + u16 -> u16`

3. Signed integer + unsigned integer -> unsigned integer
   - `u32 + i32 -> u32`
   - `u8 + i8 -> u8`

When it comes to rule conflicts, the rule of the highest priority applies:
  - `u8 + i16 -> i16` (when rule #2 conflicts with rule #3, rule #2 applies.)
  - `f16 + i32 -> f16` (when rule #1 conflicts with rule #2, rule #1 applies.)

A few exceptions:

- bit-shift operations return lhs' (left hand side's) data type:
  - `u8 << i32 -> u8`
  - `i16 << i8 -> i16`
- Logical operations return `i32`.
- Comparison operations return `i32`.

#### Implicit type casting in assignments

When you assign a value to a variable of a different data type, Taichi implicitly casts the value into that type. Further, if the value is of a higher precision than the variable, a warning of precision loss will be printed.

- Example 1: Variable `a` is initialized with type `float` and immediately reassigned `1`. The reassignment implicitly casts `1` from `int` to `float` without warning:

  ```python
  @ti.kernel
  def foo():
      a = 3.14
      a = 1
      print(a)  # 1.0
  ```

- Example 2: Variable `a` is initialized with type `int` and immediately reassigned `3.14`. The reassignment implicitly casts `3.14` from `float` to `int` with a warning because the type of `a` is `int` and has a lower precision than `float`:

  ```python
  @ti.kernel
  def foo():
      a = 1
      a = 3.14
      print(a)  # 3
  ```

## Compound types

Compound types are user-defined data types, which comprise multiple elements. Supported compound types include vectors, matrices, ndarrays, and structs.

Taichi allows you to use all types supplied in the `ti.types` module as scaffolds to customize *higher-level* compound types.

:::note

The `ndarray` type is discussed in another article [interacting with External Arrays](../basic/external.md).

:::


### Matrices and vectors

You can use the two functions `ti.types.matrix()` and `ti.types.vector()` to create your own matrix and vector types:

```python
vec4d = ti.types.vector(4, ti.f64)  # a 64-bit floating-point 4D vector type
mat4x3i = ti.types.matrix(4, 3, int)  # a 4x3 integer matrix type
```

You can use these customized types to instantiate vectors and matrices or annotate the data types of function arguments and struct members. For example:

```python
v = vec4d(1, 2, 3, 4)  # Create a vector instance, here v = [1.0 2.0 3.0 4.0]

@ti.func
def length(w: vec4d):  # vec4d as type hint
    return w.norm()

@ti.kernel
def test():
    print(length(v))
```

In practice, `ti.types.matrix` only would suffice your need for vector/matrix customization because Taichi treats vectors as a special kind of matrices, i.e., matrices with one column.

In fact, calling the function `ti.types.vector()` produces a matrix type of a single column:

```
vec3 = ti.types.vector(3, float)  # equivalent to vec3 = ti.types.matrix(3, 1, float)
```

Similarly, `ti.Vector()` simply converts the input into a matrix of a single column:

```python
v = ti.Vector([1, 1, 1])  # equivalent to v = ti.Matrix([[1], [1], [1]])
```


### Struct types and dataclass

You can use the funtion `ti.types.struct()` to create a struct type. Try customizing compound types to represent a sphere in the 3D space, which can be abstracted with its center and radius. In the following example, you call `ti.types.vector()` and `ti.types.struct()` to create compound types `vec3` and `sphere_type`, respectively. These two types are the *higher-level* compound types that fit better with your scenario. Subsequently, you can use them as templates to create two instances of spheres (initialize two local variables `sphere1` and `sphere2`):

```python
# Define a compound type vec3 to represent a sphere's center
vec3 = ti.types.vector(3, float)
# Define a compound type sphere_type to represent a sphere
sphere_type = ti.types.struct(center=vec3, radius=float)
# Initialize sphere1, whose center is at [0,0,0] and whose radius is 1.0
sphere1 = sphere_type(center=vec3([0, 0, 0]), radius=1.0)
# Initialize sphere2, whose center is at [1,1,1] and whose radius is 1.0
sphere2 = sphere_type(center=vec3([1, 1, 1]), radius=1.0)
```

When a struct contains many members, `ti.types.struct` may make your code look messy. Taichi offers a more intuitive way to define a struct: The decorator `@ti.dataclass` is a thin wrapper of the struct type:

```python
@ti.dataclass
class Sphere:
    center: vec3
    radius: float
```

The code above serves the same purpose as the line below does but provides better readability:

```python
Sphere = ti.types.struct(center=vec3, radius=float)
```

### Initialization

Just as you do with any other data type, you can call a compound type directly to create vector, matrix, or struct instances in Taichi.

As of v1.1.0, you are given more options to initialize a struct or a dataclass.

- Pass positional arguments to a struct in the order they are defined.
- Pass keyword arguments to a struct to set the corresponding struct members.
- Unspecified struct members are automatically set to zero.

For example:

  ```python
  @ti.dataclass
  class Ray:
      ro: vec3
      rd: vec3
      t: float

  # the above definition is equivalent to
  #Ray = ti.types.struct(ro=vec3, rd=vec3, t=float)

  # use positional arguments to set struct members in order
  ray = Ray(vec3(0), vec3(1, 0, 0), 1.0)
  # ro is set to vec3(0) and t will be set to 0
  ray = Ray(vec3(0), rd=vec3(1, 0, 0))
  # both ro and rd are set to vec3(0)
  ray = Ray(t=1.0)
  # ro is set to vec3(1), rd=vec3(0) and t=0.0
  ray = Ray(1)
  # all members are set to 0.
  ray = Ray()
  ```

  :::note

  You can create vectors, matrices, and structs using GLSL-like broadcast syntax because their shapes are already known.

  :::

### Type casting

For now, the only compound types that support type casting in Taichi are vectors and matrices. Type casting of vectors and matrices is element-wise and results in new vectors and matrices being created:

```python
@ti.kernel
def foo():
    u = ti.Vector([2.3, 4.7])
    v = int(u)              # ti.Vector([2, 4])
    # If you are using ti.i32 as default_ip, this is equivalent to:
    v = ti.cast(u, ti.i32)  # ti.Vector([2, 4])
```
