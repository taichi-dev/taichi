---
sidebar_position: 1

---

# Type system


Taichi is a statically typed programming language. The type of a variable in the Taichi scope is determined at compile time. Once a variable is declared, you cannot bind its name to a value of a different type afterward.

Let's see a quick example:

```python
@ti.kernel
def test():
    x = 1  # x is the integer 1
    x = 3.14  # 3.14 is cast to 3 hence x is equal to 3
    x = ti.Vector([1, 1])  # Error!
```

At Line 3 in the code above, Taichi will treat `x` as an integer since it's assigned by 1 upon its declaration. In Line 4 we assign the floating-point number 3.14 to `x`. This scalar will be automatically cast to integer 3 to match the type of `x`, hence `x` is equal to 3 after this line. Line 5 will raise a syntax error where we try to assign a `ti.Vector` to `x`, since `ti.Vector` is of a completely different type and cannot be cast into integers.

We now give a brief summary of the data types in Taichi. More detailed explanations will follow below.

Taichi supports two categories of data types in the [Taichi scope](../kernels/syntax.md#taichi-scope): primitive types and compound types.

- Primitive types: Various commonly used numerical data types, such as `int32`, `uint8`, and `float64`.
- Compound types: Array-like or struct-like data types  that comprise multiple members of primitive types, or other compound types, such as `matrix`, `struct` and `ndarray`.


## Primitive types

Primitive types are scalar types and are the smallest units of building blocks for compound types. Each primitive type is denoted with a character indicating its category followed by a number indicating its precision bits (number of bits for storing the data). The _category_ can be `i` (signed integers), `u` (unsigned integers), or `f` (floating-point numbers); the _precision bits_ can be `8`, `16`, `32`, or `64`. Following are the two most commonly used types:

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

When you call the `ti.init()` method to initialize the Taichi runtime, Taichi automatically sets the default types for integer and floating-point arithmetic:

- The default integer type in Taichi is `ti.i32`.
- The default floating-point type in Taichi is `ti.f32`.

You can customize these default primitive types by explicitly specifying them in the `init()` call:

```python
ti.init(default_ip=ti.i64)  # Set the default integer type to ti.i64
ti.init(default_fp=ti.f64)  # Set the default floating-point type to ti.f64
```

### Use `int` and `float` as alias for default primitive types

Taichi supports using `int` as an alias for the default integer type and `float` as an alias for the default floating-point type. In the following example, you change the default primitive types to `i64` and `f64` when initializing Taichi, then you can use `int` as an alias for `i64` and `float` as an alias for `f64`.

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

### Explicit typecasting

As we have mentioned at the beginning of this article, in the Taichi scope, the type of a variable is *statically typed* upon initialization. Taichi's compiler does type check at compile time, so you *cannot* change a variable's type once it is initialized. Still, from time to time, you may run into a situation where you have a certain type of data but it is not feasible for an assignment or calculation. Then, you need explicit typecasting:

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
      x1 = ti.i32(a)  # 3
      y1 = ti.f64(a)  # 3.14
  ```

### Implicit typecasting

Implicit typecasting occurs when you *accidentally* put or assign a value in a place where a different data type is expected.

:::caution WARNING
As a rule of thumb, implicit typecasting is a major source of bugs. And Taichi does *not* recommend resorting to this mechanism.

:::

Implicit typecasting can happen in binary operations or in assignments, as explained below.

#### Implicit typecasting in binary operations

Taichi implements its own implicit typecasting rules for binary operations, which are slightly different from [those for the C programming language](https://en.cppreference.com/w/c/language/conversion). In general we have three rules in descending order of priority:

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

#### Implicit typecasting in assignments

When you assign a value to a variable of a different data type, Taichi implicitly casts the value into that type. Further, if the value is of a higher precision than the variable, a warning of precision loss will be printed.

- Example 1: Variable `a` is initialized with type `float` and immediately reassigned `1`. The reassignment implicitly casts `1` from `int` to `float` without warning:

  ```python
  @ti.kernel
  def foo():
      a = 3.14
      a = 1
      print(a)  # 1.0
  ```

- Example 2: Variable `a` is initialized with type `int` and immediately reassigned `3.14`. The reassignment implicitly casts `3.14` from `float` to `int` with a warning because the type of `a` is `int` and has a low precision than `float`:

  ```python
  @ti.kernel
  def foo():
      a = 1
      a = 3.14
      print(a)  # 3
  ```

## Compound types

Compound types are user-defined data types, which comprise multiple members. Supported compound types include vectors, matrices, ndarrays, and structs.

Taichi allows you to use all types supplied in the `ti.types` module as scaffolds to customize *higher-level* compound types.

:::note

The `ndarray` type is discussed in another article [interacting with External Arrays](../basic/external.md).

:::


### Matrix and Vector types

You can use the two functions `ti.types.matrix()` and `ti.types.vector()` to create your own matrix and vector types:

```python
vec4d = ti.types.vector(4, ti.f64)  # 64-bit floating-point 4D vector type
mat4x3i = ti.types.matrix(4, 3, int)  # integer 4x3 matrix type
```

Such types can be directly called to create the corresponding vector/matrix instances, and can also be used as type hints in function arguments and struct members. For example:

```python
v = vec4d(1, 2, 3, 4)  # Create a vector instance, here v = [1.0, 2.0 3.0 4.0]

@ti.func
def length(w: vec4d):  # vec4d as type hint
    return w.norm()

@ti.kernel
def test():
    print(length(v))
```

### Struct types and dataclass

You can use the funtion `ti.types.struct()` to create a struct type, and use this type as a template to create struct instances. For example suppose you are using Taichi to represent a sphere. A sphere in the 3D space can be abstracted with its center and radius. In the following example, you call `ti.types.vector()` and `ti.types.struct()` to create compound types `vec3` and `sphere_type`. These two types are the *higher-level* compound types that fit better with your scenario. Once you have customized your compound types, you can use them as templates to create two instances of spheres (initialize two local variables `sphere1` and `sphere2`):

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

When there are a dozen of members in a struct, the above way of using `ti.types.struct` may make your code look messy, and there is a better way to do so. Indeed, we recommend you use the decorator `@ti.dataclass` as a more intuitive way to define your custom struct type in the form of a class. The dataclass is merely a thin wrapper of struct:

```python
@ti.dataclass
class Sphere:
    center: vec3
    radius: float
```

The code above is equivalent to the line of code below:

```python
Sphere = ti.types.struct(center=vec3, radius=float)
```

### Initialization

Just as you do with any other data type, you can call a compound type directly to create vector, matrix, or struct instances in Taichi.

As the release of v1.1.0, you are given more options to initialize a struct or a dataclass. The positional arguments are passed to the struct members in the order they are defined; the keyword arguments set the corresponding struct members. Unspecified struct members are automatically set to zero. For example:

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

  - In the definition of `vec2`, `my_vec3f()` performs an implicit cast operation when combining `my_vec2i(0)` with `1`.
  - You can create vectors, matrices, and structs using GLSL-like broadcast syntax because their shapes are already known.

  :::

### Typecasting

For now, the only compound types in Taichi that support typecasting are vectors and matrices, the casting is performed element-wise and a new vector/matrix is created:

```python
@ti.kernel
def foo():
    u = ti.Vector([2.3, 4.7])
    v = int(u)              # ti.Vector([2, 4])
    # If you are using ti.i32 as default_ip, this is equivalent to:
    v = ti.cast(u, ti.i32)  # ti.Vector([2, 4])
```
