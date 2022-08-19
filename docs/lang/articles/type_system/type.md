---
sidebar_position: 1

---

# Type system

Taichi supports two data types in the [Taichi scope](../kernels/syntax.md#taichi-scope): primitive type and compound type.

- Primitive types: Various commonly-used numerical data types, such as `int32`, `uint8`, and `float64`
- Compound types: User-defined data types, which comprise multiple members.

## Primitive types

Each primitive type is denoted with a character indicating its category followed by a number indicating its precision bits (number of bits for storing the data). The _category_ can be `i` (signed integers), `u` (unsigned integers), or `f` (floating-point numbers); the _precision bits_ can be `8`, `16`, `32`, or `64`. Following are the two most commonly-used types:

- `i32`: 32-bit signed integer
- `f32` : 32-bit floating-point number.

### Supported primitive types

Not all primitive types are supported by your backend. Check out the following table for the supported types. Note that some backends may require extensions to support a specific primitive type.

| Backend | `i8`               | `i16`              | `i32`              | `i64`              | `u8`                 | `u16`                | `u32`                | `u64`                | `f16`                | `f32`                | `f64`                |
| ------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| CPU     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CUDA    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| OpenGL  | :x:                | :x:                | :heavy_check_mark: | :o:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: |
| Metal   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                |
| Vulkan  | :o:                | :o:                | :heavy_check_mark: | :o:                | :o:                | :o:                | :heavy_check_mark: | :o:                | :heavy_check_mark: | :heavy_check_mark: | :o:                |

> :o:: Requiring extensions for the backend.

### Default primitive types for integers and floating-point numbers

- The default integer type in Taichi is `ti.i32`.
- The default floating-point type in Taichi is `ti.f32`.

#### Customize the default type

You can change the default primitive types by explicitly specifying the default types when initializing Taichi:

```python
ti.init(default_ip=ti.i64)  # Set the default integer type to ti.i64
ti.init(default_fp=ti.f64)  # Set the default floating-point type to ti.f64
```

#### Set alias

Taichi supports using `int` as an alias for the default integer type and `float` as an alias for the default floating-point type. In the following example, you change the default primitive types to `i64` and `f64` when initializing Taichi, then you can use `int` to represent `i64` and `float` to represent `f64`.

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

In the Taichi scope, the type of a variable is *statically typed* upon initialization. Taichi's compiler does type check at compile time, so you *cannot* change a variable's type once it is initialized. Still, from time to time, you may run into a situation where you have a certain type of data but it is not feasible for an assignment or calculation. Then, you need explicit type casting:

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

### Implicit type casting

Implicit type casting occurs when you *accidentally* put or assign a value in a place where a different data type is expected.

:::caution WARNING
As a rule of thumb, implicit type casting is a major source of bugs. And Taichi does *not* recommend resorting to this mechanism.

:::

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
- atan2 operations return `f64` if either side is `f64`, or `f32` otherwise.
  - `i32 atan f32 -> f32`
  - `i32 atan f64 -> f64`
- Logical operations return `i32`.
- Comparison operations return `i32`.

#### Implicit type casting in assignments

When you assign a value to a variable of a different data type, Taichi implicitly casts the value into that type. Further, if the type of the variable is *not* the common type, a warning of precision loss occurs.

- Example 1: Variable `a` is initialized with type `float` and immediately reassigned `1`. The reassignment implicitly casts `1` from `int` to `float` without warning because the data type of `a` is the common type `float`:

  ```python
  @ti.kernel
  def foo():
      a = 3.14
      a = 1
      print(a)  # 1.0
  ```

- Example 2: Variable `a` is initialized with type `int` and immediately reassigned `3.14`. The reassignment implicitly casts `3.14` from `float` to `int` with a warning because the type of `a` is *not* the common type `float`:

  ```python
  @ti.kernel
  def foo():
      a = 1
      a = 3.14
      print(a)  # 3
  ```

## Compound types

Compound types are user-defined data types, which comprise multiple members. Supported compound types include vectors, metrics, and structs.

Taichi allows you to use all types supplied in the `ti.type` module as scaffolds to customize *higher-level* compound types.

Suppose you are using Taichi to represent a sphere. A sphere in the 3D space can be abstracted with its center and radius. In the following example, you call `ti.types.vector()` and `ti.types.struct()` to create compound types `vec3` and `sphere_type`. These two types are the *higher-level* compound types that fit better with your scenario. Once you have customized your compound types, you can use them as templates to create two instances of spheres (initialize two local variables `sphere1` and `sphere2`):

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

### Initialization

Just as you do with any other data type, you can call a compound type directly to create vector, matrix, or struct instances in Taichi.

- In the following code snippet, four compound types `my_vec2i`, `my_vec3f`, `my_mat2f`, and `my_ray3f` are defined:

  ```python
  my_vec2i = ti.types.vector(2, ti.i32)
  my_vec3f = ti.types.vector(3, float)
  my_mat2f = ti.types.matrix(2, 2, float)
  my_ray3f = ti.types.struct(ro=my_vec3f, rd=my_vec3f, l=ti.f32)
  ```

- In the following code snippet, you initialize five local variables using the created four compound types.

  ```python
  ray1 = my_ray3f(0.0)            # ti.Struct(ro=[0.0, 0.0, 0.0], rd=[0.0, 0.0, 0.0], l=0.0)
  vec1 = my_vec3f(0.0)            # ti.Vector([0.0, 0.0, 0.0])
  mat1 = my_mat2f(1.0)            # ti.Matrix([[1.0, 1.0], [1.0, 1.0]])
  vec2 = my_vec3f(my_vec2i(0), 1) # ti.Vector([0.0, 0.0, 1.0]) performs implicit cast
  ray2 = my_ray3f(ro=vec1, rd=vec2, l=1.0)
  ```

  :::note

  - In the definition of `vec2`, `my_vec3f()` performs an implicit cast operation when combining `my_vec2i(0)` with `1`.
  - You can create vectors, matrices, and structs using GLSL-like broadcast syntax because their shapes are already known.

  :::

### Type casting

Type casting on vectors and matrices in Taichi is element-wise; type casting on structs is *not*.

```python
@ti.kernel
def foo():
    u = ti.Vector([2.3, 4.7])
    v = int(u)              # ti.Vector([2, 4])
    # If you are using ti.i32 as default_ip, this is equivalent to:
    v = ti.cast(u, ti.i32)  # ti.Vector([2, 4])
```
