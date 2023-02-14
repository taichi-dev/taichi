---
sidebar_position: 1

---

# Type System

Taichi is a statically typed programming language, meaning that the type of a variable in the Taichi scope is determined at compile time. This means that once a variable has been declared, it cannot be assigned a value of a different type.

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

The `ti.types` module in Taichi defines all of the supported data types. These data types are categorized into two groups: primitive and compound.

- Primitive types encompass commonly utilized numerical data types, such as `ti.i32` (`int32`), `ti.u8` (`uint8`), and `ti.f64` (`float64`).
- Compound types, on the other hand, encompass array-like or struct-like data types, including `ti.types.matrix`, `ti.types.ndarray`, and `ti.types.struct`. These types are composed of multiple members, which can be primitive or other compound types.

## Primitive types

The primitive data types in Taichi are scalars, which are the smallest units that make up compound data types. These types are denoted with a letter indicating their category, followed by a number indicating their precision in bits. The category letter can be `i` for signed integers, `u` for unsigned integers, or `f` for floating-point numbers. The precision bits can be 8, 16, 32, or 64. The two most commonly used primitive types are:

- `i32`: 32-bit signed integer
- `f32` : 32-bit floating-point number.

The support of Taichi's primitive types by various backends may vary. Consult the following table for detailed information, and note that some backends may require extensions for complete support of a specific primitive type.


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

The numeric literals in Taichi's scope have default integer or floating-point types. For example, if the default floating-point type is `ti.f32`, a numeric literal 3.14159265358979 will be cast to a 32-bit floating-point number with a precision of approximately seven decimal digits. To ensure high precision in applications such as engineering simulations, it is recommended to set `default_fp` to `ti.f64`.

:::

### Set default primitive type alias

In Taichi, the keywords `int` and `float` serve as aliases for the default integer and floating-point types, respectively. These default types can be changed using the configuration option `default_ip` and `default_fp`. For instance, setting the `default_ip` to `i64` and `default_fp` to `f64` would allow you to use `int` as an alias for `i64` and `float` as an alias for `f64` in your code.

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

As mentioned at the beginning of this document, the type of a variable in the Taichi scope is determined at compile time, meaning that it is *statically typed*. The Taichi compiler performs type checking at compile time and therefore, once a variable is declared, you *cannot* assign to it a value of a different type. However, in certain situations, you may need to switch to a different data type due to the unavailability of the original type for an assignment or calculation. In these cases, you must perform an explicit type casting.

- The `ti.cast()` function allows you to convert a given value to a specific target type. For instance, you can use `ti.cast(x, float)` to transform a variable `x` into a floating-point type.

  ```python
  @ti.kernel
  def foo():
      a = 3.14
      b = ti.cast(a, ti.i32)  # 3
      c = ti.cast(b, ti.f32)  # 3.0
  ```

As of Taichi v1.1.0, the capability to perform type casting on scalar variables has been introduced using primitive types such as `ti.f32` and `ti.i64`. This allows you to convert scalar variables to different scalar types with ease.

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
::


#### Implicit type casting in binary operations

In Taichi, implicit type casting can occur during binary operations or assignments. The casting rules are implemented specifically for Taichi and are slightly different from [those for the C programming language](https://en.cppreference.com/w/c/language/conversion). These rules are prioritized as follows:

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

In Taichi, implicit type casting is performed when assigning a value to a variable with a different data type. In cases where the value has a higher precision than the target variable, a warning indicating potential precision loss will be displayed.

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

You can utilize the customized compound types to instantiate vectors and matrices, as well as annotate the data types of function arguments and struct members. For instance:

```python
v = vec4d(1, 2, 3, 4)  # Create a vector instance, here v = [1.0 2.0 3.0 4.0]

@ti.func
def length(w: vec4d):  # vec4d as type hint
    return w.norm()

@ti.kernel
def test():
    print(length(v))
```


### Struct types and dataclass

You can use the function `ti.types.struct()` to create a struct type, which can be utilized to represent a sphere in 3D space, abstracted by its center and radius. To achieve this, you can call `ti.types.vector()` and `ti.types.struct()` to create two higher-level compound types: `vec3` and `sphere_type`, respectively. These types can then be used as templates to initialize two local variables, `sphere1` and `sphere2`, to represent two instances of spheres.

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

When defining a struct with numerous members, the use of `ti.types.struct` can lead to cluttered and unorganized code. Taichi provides a more elegant solution with the `@ti.dataclass` decorator, which acts as a lightweight wrapper around the struct type.

```python
@ti.dataclass
class Sphere:
    center: vec3
    radius: float
```

The code above accomplishes the same task as the following line, however it offers improved comprehensibility:

```python
Sphere = ti.types.struct(center=vec3, radius=float)
```

Another benefit of utilizing the `@ti.dataclass` over the `ti.types.struct` is the ability to define member functions within a dataclass, enabling object-oriented programming (OOP) capabilities. For more information on the topic of objective data-oriented programming, refer to the [objective data-oriented programming](../advanced/odop.md) documentation.


### Initialization

In Taichi, creating instances of vector, matrix, or struct compound types can be achieved by directly calling the type, similar to how it is done with any other data type.

As of Taichi v1.1.0, multiple options are available for initializing instances of structs or dataclasses. The conventional method of calling a compound type directly still holds true. In addition, the following alternatives are also supported:

- Pass positional arguments to the struct, in the order in which the members are defined.
- Utilize keyword arguments to set the specific struct members.
- Members that are not specified will be automatically set to zero.

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

For now, the only compound data types that support type casting in Taichi are vectors and matrices. When casting the type of a vector or matrix, it is performed element-wise, resulting in the creation of new vectors and matrices.


```python
@ti.kernel
def foo():
    u = ti.Vector([2.3, 4.7])
    v = int(u)              # ti.Vector([2, 4])
    # If you are using ti.i32 as default_ip, this is equivalent to:
    v = ti.cast(u, ti.i32)  # ti.Vector([2, 4])
```
