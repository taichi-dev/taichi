---
sidebar_position: 1

---

# Type System

Taichi is a statically typed programming language: The type of a variable in the Taichi scope is determined at compile time; once a variable is declared, you cannot assign to it a value of a different type.

Let's see a quick example:

```python {3-5}
@ti.kernel
def test():
    x = 1  # x is the integer 1
    x = 3.14  # x is an integer, so the value 3.14 is cast to 3 and x takes the value 3
    x = ti.Vector([1, 1])  # Error!
```

- Line 3: `x` is an integer because it is assigned an integer value the first time it is declared.
- Line 4: `x` is reassigned a floating-point number 3.14 but takes the value 3. This is because 3.14 is automatically cast to integer 3 to match the type of `x`.
- Line 5:  The system throws an error, because `ti.Vector` cannot be cast into an integer.

Taichi's `ti.types` module defines all the supported data types, and they are classified into two categories: primitive types and compound types.

- Primitive types refer to various commonly-used numerical data types, such as `ti.i32` (`int32`), `ti.u8` (`uint8`), and `ti.f64` (`float64`).
- Compound types refer to various array-like or struct-like data types, including `ti.types.matrix`, `ti.types.ndarray`, and `ti.types.struct`. Compound types comprise multiple members of primitive types or of other compound types.

## Primitive types

Primitive types refer to scalars, which are the smallest building blocks of compound types. Each primitive type is denoted with a character indicating its category followed by a number indicating its precision bits (number of bits for storing the data). The _category_ can be `i` (signed integers), `u` (unsigned integers), or `f` (floating-point numbers); the _precision bits_ can be `8`, `16`, `32`, or `64`. Following are the two most commonly used types:

- `i32`: 32-bit signed integer
- `f32` : 32-bit floating-point number.

Not all backends support Taichi's primitive types. See the following table for how a primitive type is supported by various backends. Note that some backends may require extensions to support a specific primitive type.

| Backend | `i8`               | `i16`              | `i32`              | `i64`              | `u8`               | `u16`              | `u32`              | `u64`              | `f16`              | `f32`              | `f64`              |
| ------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| CPU     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CUDA    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| OpenGL  | :x:                | :x:                | :heavy_check_mark: | :o:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: |
| Metal   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                |
| Vulkan  | :o:                | :o:                | :heavy_check_mark: | :o:                | :o:                | :o:                | :heavy_check_mark: | :o:                | :heavy_check_mark: | :heavy_check_mark: | :o:                |

> :o:: Requiring extensions for the backend.

### Customize default primitive types

When initializing the Taichi runtime, Taichi automatically uses the following data types as the default primitive data types:

- `ti.i32`: the default integer type.
- `ti.f32`: the default floating-point type.

Taichi allows you to specify the default primitive data type(s) when calling `init()`:

```python
ti.init(default_ip=ti.i64)  # Sets the default integer type to ti.i64
ti.init(default_fp=ti.f64)  # Sets the default floating-point type to ti.f64
```

:::note

The numeric literals in the Taichi scope also have default integer/floating-point types. For example, if the default floating-point type is `ti.f32`, then a numeric literal `3.14159265358979` in the Taichi scope suffers a precision loss because it is cast to a 32-bit floating-point number, which has a precision of about seven decimal digits.

If you are working on a high-precision application scenario, such as numeric simulation for engineering, set `default_fp` as `ti.f64`.

:::

### Set default primitive type alias

Taichi supports using `int` as the alias for the default integer type and `float` as the alias for the default floating-point type. For example, after changing the default primitive types to `i64` and `f64`, you can use `int` as the alias for `i64` and `float` as the alias for `f64`.

```python
ti.init(default_ip=ti.i64, default_fp=ti.f64)

x = ti.field(float, 5)
y = ti.field(int, 5)
# Is equivalent to:
x = ti.field(ti.f64, 5)
y = ti.field(ti.i64, 5)

def func(a: float) -> int:
    ...
# Is equivalent to:
def func(a: ti.f64) -> ti.i64:
    ...
```

### Explicit type casting

As mentioned at the beginning of this document, the type of a variable in the Taichi scope is *statically typed* upon initialization. Taichi's compiler performs type checking at compile time, meaning that you *cannot* change a variable's type once it is initialized. However, from time to time, you may run into situations where you need to switch to a different data type because the original is not feasible for an assignment or calculation. In such situations, you need explicit type casting:

- You can use `ti.cast()` to convert a value to the target type:

  ```python
  @ti.kernel
  def foo():
      a = 3.14
      b = ti.cast(a, ti.i32)  # 3
      c = ti.cast(b, ti.f32)  # 3.0
  ```

- As of v1.1.0, Taichi allows you to use primitive types such as `ti.f32` and `ti.i64` to convert a scalar variable to a different scalar type:

  ```python {6,7}
  @ti.kernel
  def foo():
      a = 3.14
      x = int(a)    # 3
      y = float(a)  # 3.14
      z = ti.i32(a)  # 3
      w = ti.f64(a)  # 3.14
  ```

### Implicit type casting

Implicit type casting happens when you *accidentally* put or assign a value where a different data type is expected.

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

The `ndarray` type is discussed in another document [interacting with External Arrays](../basic/external.md).

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

In practical terms, `ti.types.matrix` only would suffice your need for vector/matrix customization because Taichi treats vectors as a special kind of matrices, i.e., matrices with one column.

In fact, calling `ti.types.vector()` produces a matrix type of a single column:

```
vec3 = ti.types.vector(3, float)  # Equivalent to vec3 = ti.types.matrix(3, 1, float)
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

Another advantage of using `@ti.dataclass` over `ti.types.struct` is that you can define member functions in a dataclass and call them in the Taichi scope, making object-oriented programming (OOP) possible. See the [objective data-oriented programming](../advanced/odop.md) for more information.


### Initialization

Just as you do with any other data type, you can call a compound type directly to create vector, matrix, or struct instances in Taichi.

As of v1.1.0, Taichi supports more options for initializing a struct or a dataclass.

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

  # The definition above is equivalent to
  #Ray = ti.types.struct(ro=vec3, rd=vec3, t=float)
  # Use positional arguments to set struct members in order
  ray = Ray(vec3(0), vec3(1, 0, 0), 1.0)
  # ro is set to vec3(0) and t will be set to 0
  ray = Ray(vec3(0), rd=vec3(1, 0, 0))
  # both ro and rd are set to vec3(0)
  ray = Ray(t=1.0)
  # ro is set to vec3(1), rd=vec3(0) and t=0.0
  ray = Ray(1)
  # All members are set to 0
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
