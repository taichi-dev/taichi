---
sidebar_position: 2
---

# Type system

Data types in Taichi consist of Primitive Types and Compound Types. Primitive Types are the numerical data types used by backends, while Compound Types are user-defined types of data records composed of multiple members.

## Primitive types

Taichi supports common numerical data types. Each type is denoted as a
character indicating its _category_ and a number of _precision bits_,
e.g., `i32` and `f64`.

The _category_ can be one of:

- `i` for signed integers, e.g. 233, -666
- `u` for unsigned integers, e.g. 233, 666
- `f` for floating point numbers, e.g. 2.33, 1e-4

The _digital number_ can be one of:

- `8`
- `16`
- `32`
- `64`

It represents how many **bits** are used in storing the data. The larger
the bit number, the higher the precision is.

For example, the two most commonly used types:

- `i32` represents a 32-bit signed integer.
- `f32` represents a 32-bit floating pointer number.

## Supported primitive types

Currently, supported primitive types in Taichi are

- int8 `ti.i8`
- int16 `ti.i16`
- int32 `ti.i32`
- int64 `ti.i64`
- uint8 `ti.u8`
- uint16 `ti.u16`
- uint32 `ti.u32`
- uint64 `ti.u64`
- float32 `ti.f32`
- float64 `ti.f64`

:::note

Supported types on each backend:

| type | CPU/CUDA | OpenGL | Metal | C source |
| ---- | -------- | ------ | ----- | -------- |
| i8   | > OK     | > N/A  | > OK  | > OK     |
| i16  | > OK     | > N/A  | > OK  | > OK     |
| i32  | > OK     | > OK   | > OK  | > OK     |
| i64  | > OK     | > EXT  | > N/A | > OK     |
| u8   | > OK     | > N/A  | > OK  | > OK     |
| u16  | > OK     | > N/A  | > OK  | > OK     |
| u32  | > OK     | > N/A  | > OK  | > OK     |
| u64  | > OK     | > N/A  | > N/A | > OK     |
| f32  | > OK     | > OK   | > OK  | > OK     |
| f64  | > OK     | > OK   | > N/A | > OK     |

(OK: supported, EXT: require extension, N/A: not available)
:::

:::note
Boolean types are represented using `ti.i32`.
:::

## Type promotion

Binary operations on different types will give you a promoted type,
following the C programming language convention, e.g.:

- `i32 + f32 = f32` (integer + float = float)
- `i32 + i64 = i64` (less-bits + more-bits = more-bits)

Basically it will try to choose the more precise type to contain the
result value.

## Default precisions

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

Also note that you may use `float` or `int` in type definitions as
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

## Type casts

### Implicit casts

:::caution
The type of a variable is **determinated on it's initialization**.
:::

When a _low-precision_ variable is assigned to a _high-precision_
variable, it will be implicitly promoted to the _high-precision_ type
and no warning will be raised:

```python {3}
a = 1.7
a = 1
print(a)  # 1.0
```

When a _high-precision_ variable is assigned to a _low-precision_ type,
it will be implicitly down-cast into the _low-precision_ type and Taichi
will raise a warning:

```python {3}
a = 1
a = 1.7
print(a)  # 1
```

### Explicit casts

You may use `ti.cast` to explicitly cast scalar values between different
types:

```python {2-3}
a = 1.7
b = ti.cast(a, ti.i32)  # 1
c = ti.cast(b, ti.f32)  # 1.0
```

Equivalently, use `int()` and `float()` to convert values to float-point
or integer types of default precisions:

```python {2-3}
a = 1.7
b = int(a)    # 1
c = float(a)  # 1.0
```

### Casting vectors and matrices

Type casts applied to vectors/matrices are element-wise:

```python {2,4}
u = ti.Vector([2.3, 4.7])
v = int(u)              # ti.Vector([2, 4])
# If you are using ti.i32 as default_ip, this is equivalent to:
v = ti.cast(u, ti.i32)  # ti.Vector([2, 4])
```

### Bit casting

Use `ti.bit_cast` to bit-cast a value into another data type. The
underlying bits will be preserved in this cast. The new type must have
the same width as the the old type. For example, bit-casting `i32` to
`f64` is not allowed. Use this operation with caution.

:::note
For people from C++, `ti.bit_cast` is equivalent to `reinterpret_cast`.
:::

## Compound types

User-defined compound types are created using the `ti.types` module. Supported compound types include vectors, matrices, and structs:

```python
vec2i = ti.types.vector(2, ti.i32)
vec3f = ti.types.vector(3, float)
mat2f = ti.types.matrix(2, 2, float)
ray = ti.types.struct(ro=vec3f, rd=vec3f, l=ti.f32)
```

### Creating fields

Fields of a given compound type can be created with the `.field()` method of a Compound Type:

```python
# ti.Vector.field(2, dtype=ti.i32, shape=(233, 666))
x = vec2i.field(shape=(233, 666))

# ti.Matrix.field(2, 2, dtype=ti.i32, shape=(233, 666))
x = mat2f.field(shape=(233, 666))

# ti.Struct.field({'ro': vec3f, 'rd': vec3f, 'l': ti.f32}, shape=(233, 666))
x = ray.field(shape=(233, 666))
```

### Creating local variables
Compound types can be directly called to create matrix or struct instances. Vectors and matrices can be created using GLSL-like broadcast syntax since the shape of the vector or matrix is already known:
```python
ray = ray3f(0.0) # ti.Struct(ro=[0.0, 0.0, 0.0], rd=[0.0, 0.0, 0.0], l=0.0)
ro = vec3f(0.0) # ti.Vector([0.0, 0.0, 0.0])
rd = vec3f(vec2i(0), 1) # ti.Vector([0.0, 0.0, 1.0]), will perform implicit cast
ray2 = ray3f(ro=ro, rd=rd, l=1.0)
```
