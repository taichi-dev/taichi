---
sidebar_position: 2
---

# Type system

Data types in Taichi consist of _primitive types_ and _compound types_. Primitive types are the numerical data types used by different backends, while compound types are user-defined types of data records composed of multiple members.

## Primitive types

Taichi supports common numerical data types. Each type is denoted as a
character indicating its _category_ and a number of _precision bits_,
e.g., `i32` and `f64`.

The _category_ can be one of:

- `i` for signed integers, e.g. 24, -32
- `u` for unsigned integers, e.g. 128, 256
- `f` for floating point numbers, e.g. 3.14, 1.0, 1e-4

The _digital number_ can be either `8`, `16`, `32`, or `64`.
It represents how many **bits** are used in storing the data. The larger
the bit number is, the higher the precision is.

For example, the two most commonly used types:

- `i32` represents a 32-bit signed integer.
- `f32` represents a 32-bit floating point number.

### Supported primitive types

Currently in Taichi, the supported primitive types on each backend are:


| type | CPU/CUDA |  OpenGL | Metal |  Vulkan  |
| ---- | -------- | ------- | ----- | -------- |
| i8   |:heavy_check_mark:|:x:                   |:heavy_check_mark:|:large_orange_diamond:|
| i16  |:heavy_check_mark:|:x:                   |:heavy_check_mark:|:large_orange_diamond:|
| i32  |:heavy_check_mark:|:heavy_check_mark:    |:heavy_check_mark:|:heavy_check_mark:    |
| i64  |:heavy_check_mark:|:large_orange_diamond:|:x:               |:large_orange_diamond:|
| u8   |:heavy_check_mark:|:x:                   |:heavy_check_mark:|:large_orange_diamond:|
| u16  |:heavy_check_mark:|:x:                   |:heavy_check_mark:|:large_orange_diamond:|
| u32  |:heavy_check_mark:|:x:                   |:heavy_check_mark:|:heavy_check_mark:    |
| u64  |:heavy_check_mark:|:x:                   |:x:               |:large_orange_diamond:|
| f16  |:heavy_check_mark:|:x:                   |:x:               |:heavy_check_mark:    |
| f32  |:heavy_check_mark:|:heavy_check_mark:    |:heavy_check_mark:|:heavy_check_mark:    |
| f64  |:heavy_check_mark:|:heavy_check_mark:    |:x:               |:large_orange_diamond:|

(:large_orange_diamond: means extension required)

### Default precisions

By default, all numerical literals have 32-bit precisions. For example,
`42` has type `ti.i32` and `3.14` has type `ti.f32`.

Default integer and float-point precisions (`default_ip` and
`default_fp`) can be specified when initializing Taichi:

```python
ti.init(default_fp=ti.f32)
ti.init(default_fp=ti.f64)

ti.init(default_ip=ti.i32)
ti.init(default_ip=ti.i64)
```

In addition, you can use `float` or `int` in type definitions as
aliases for default precisions, e.g.:

```python
ti.init(default_ip=ti.i64, default_fp=ti.f32)

x = ti.field(float, 5)
y = ti.field(int, 5)
# is equivalent to:
x = ti.field(ti.f32, 5)
y = ti.field(ti.i64, 5)

def func(a: float) -> int:
    ...

# is equivalent to:
def func(a: ti.f32) -> ti.i64:
    ...
```

### Type promotion

When Taichi performs binary operations on different types, the result is a
promoted type. This is known as type promotion. Following the C programming
language convention, Taichi always chooses the more precise type to contain the
result value. For example:

- `i32 + f32 = f32` (integer + float = float)
- `i32 + i64 = i64` (less-bits + more-bits = more-bits)

### Type casts

When using Taichi, you may encounter situations when certain type of data needs
to be assigned to another type. In this case, type casts are needed. Here, we
list four use cases commonly seen during Taichi programming, namely implicit
casts, explicit casts, casting vectors and matrices, and bit-casts.

:::caution
In Taichi, the type of a variable is **determined on its initialization**.
All data types are static in the **Taichi scope**.
:::

#### Implicit casts

When a _low-precision_ variable is assigned to a _high-precision_
variable, it will be implicitly promoted to the _high-precision_ type
and no warning will be raised:

```python {4}
@ti.kernel
def foo():
    a = 3.14
    a = 1
    print(a)  # 1.0
```
In this example, variable `a` is initialized with `float` type. On the next line, the assign statement will cast `1` from `int` type to `float` type implicitly.



When a _high-precision_ variable is assigned to a _low-precision_ type,
it will be implicitly down-cast into the _low-precision_ type and Taichi
will raise a warning:

```python {4}
@ti.kernel
def foo():
    a = 1
    a = 3.14
    print(a)  # 3
```
In this example, variable `a` is initialized with an `int` type. On the next line, the assignment statement casts `3.14` from `float` to `int` implicitly.


#### Explicit casts

You may use `ti.cast` to explicitly cast scalar values between different
types:

```python {4-5}
@ti.kernel
def foo():
    a = 3.14
    b = ti.cast(a, ti.i32)  # 3
    c = ti.cast(b, ti.f32)  # 3.0
```

Equivalently, use `int()` and `float()` to convert values to float-point
or integer types of default precisions:

```python {4-5}
@ti.kernel
def foo():
    a = 3.14
    b = int(a)    # 3
    c = float(b)  # 3.0
```

#### Casting vectors and matrices

Type casts applied to vectors/matrices are element-wise:

```python {4,6}
@ti.kernel
def foo():
    u = ti.Vector([2.3, 4.7])
    v = int(u)              # ti.Vector([2, 4])
    # If you are using ti.i32 as default_ip, this is equivalent to:
    v = ti.cast(u, ti.i32)  # ti.Vector([2, 4])
```

#### Bit-casts

Use `ti.bit_cast` to bit-cast a value into another data type. The
underlying bits will be preserved in this cast. The new type must have
the same width as the the old type. For example, bit-casting `i32` to
`f64` is not allowed. Use this operation with caution.

```python {4-5}
@ti.kernel
def foo():
    a = 3.14
    b = ti.bit_cast(a, ti.i32) # 1078523331
    c = ti.bit_cast(b, ti.f32) # 3.14
```

:::note
For people from C++, `ti.bit_cast` is equivalent to `reinterpret_cast`.
:::

## Compound types

User-defined compound types can be created using the `ti.types` module. Supported compound types include vectors, matrices, and structs:

```python
my_vec2i = ti.types.vector(2, ti.i32)
my_vec3f = ti.types.vector(3, float)
my_mat2f = ti.types.matrix(2, 2, float)
my_ray3f = ti.types.struct(ro=my_vec3f, rd=my_vec3f, l=ti.f32)
```
In this example, we define four compound types for creating fields and local variables.

### Creating fields

Fields of a user-defined compound type can be created with the `.field()` method of a Compound Type:

```python
vec1 = my_vec2i.field(shape=(128, 128, 128))
mat2 = my_mat2f.field(shape=(24, 32))
ray3 = my_ray3f.field(shape=(1024, 768))

# is equivalent to:
vec1 = ti.Vector.field(2, dtype=ti.i32, shape=(128, 128, 128))
mat2 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=(24, 32))
ray3 = ti.Struct.field({'ro': my_vec3f, 'rd': my_vec3f, 'l': ti.f32}, shape=(1024, 768))
```
In this example, we define three fields in two different ways but of exactly the same effect.

### Creating local variables

Compound types can be directly called to create vector, matrix or struct instances. Vectors, matrices and structs can be created using GLSL-like broadcast syntax since their shapes are already known:
```python
ray1 = my_ray3f(0.0)            # ti.Struct(ro=[0.0, 0.0, 0.0], rd=[0.0, 0.0, 0.0], l=0.0)
vec1 = my_vec3f(0.0)            # ti.Vector([0.0, 0.0, 0.0])
mat1 = my_mat2f(1.0)            # ti.Matrix([[1.0, 1.0], [1.0, 1.0]])
vec2 = my_vec3f(my_vec2i(0), 1) # ti.Vector([0.0, 0.0, 1.0]), will perform implicit cast
ray2 = my_ray3f(ro=vec1, rd=vec2, l=1.0)
```
In this example, we define five local variables, each of a different type. In the definition statement of `vec2`, `my_vec3f()` performs an implicit cast operation when combining `my_vec2i(0)` with `1`.
