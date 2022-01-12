---
sidebar_position: 2
---

# Type system

Data types in Taichi consist of Primitive Types and Compound Types. Primitive Types are the numerical data types used by different backends, while Compound Types are user-defined types of data records composed of multiple members.

## Primitive types

Taichi supports common numerical data types. Each type is denoted as a
character indicating its _category_ and a number of _precision bits_,
e.g., `i32` and `f64`.

The _category_ can be one of:

- `i` for signed integers, e.g. 24, -32
- `u` for unsigned integers, e.g. 128, 256
- `f` for floating point numbers, e.g. 3.14, 1.0, 1e-4

The _digital number_ can be one of:

- `8`
- `16`
- `32`
- `64`

It represents how many **bits** are used in storing the data. The larger
the bit number, the higher the precision is.

For example, the two most commonly used types:

- `i32` represents a 32-bit signed integer.
- `f32` represents a 32-bit floating point number.

### Supported primitive types

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
| f32  |:heavy_check_mark:|:heavy_check_mark:    |:heavy_check_mark:|:heavy_check_mark:    |
| f64  |:heavy_check_mark:|:heavy_check_mark:    |:x:               |:large_orange_diamond:|

(:large_orange_diamond: requires extension)
:::

:::note
Boolean types are represented using `ti.i32`.
:::

### Supported operations

#### Arithmetic operators

| Operation | Result                          |
| --------- | ------------------------------- |
| `-a`      | `a` negated                     |
| `+a`      | `a` unchanged                   |
| `a + b`   | sum of `a` and `b`              |
| `a - b`   | difference of `a` and `b`       |
| `a * b`   | product of `a` and `b`          |
| `a / b`   | quotient of `a` and `b`         |
| `a // b`  | floored quotient of `a` and `b` |
| `a % b`   | remainder of `a / b`          |
| `a ** b`  | `a` to the power of `b`         |

:::note

The `%` operator in Taichi follows the Python style instead of C style,
e.g.,

```python
# In Taichi-scope or Python-scope:
print(2 % 3)   # 2
print(-2 % 3)  # 1
```

For C-style mod (`%`), please use `ti.raw_mod`:

```python
print(ti.raw_mod(2, 3))   # 2
print(ti.raw_mod(-2, 3))  # -2
```
:::

:::note

Python 3 distinguishes `/` (true division) and `//` (floor division), e.g., `1.0 / 2.0 = 0.5`, `1 / 2 = 0.5`, `1 // 2 = 0`,
`4.2 // 2 = 2`. Taichi follows the same design:

- **True divisions** on integral types will first cast their
  operands to the default floating point type.
- **Floor divisions** on floating point types will first cast their
  operands to the default integral type.

To avoid such implicit casting, you can manually cast your operands to
desired types, using `ti.cast`. Please see
[Default precisions](#default-precisions) for more details on
default numerical types.
:::

#### Logic operators

| Operation          | Result                                                        |
| ------------------ | ------------------------------------------------------------- |
| `a == b`           | if `a` is equal to `b`, then True, else False                 |
| `a != b`           | if `a` is not equal to `b`, then True, else False             |
| `a > b`            | if `a` is strictly greater than `b`, then True, else False    |
| `a < b`            | if `a` is strictly less than `b`, then True, else False       |
| `a >= b`           | if `a` is greater than or equal to `b`, then True, else False |
| `a <= b`           | if `a` is less than or equal to `b`, then True, else False    |
| `not a`            | if `a` is False, then True, else False                        |
| `a or b`           | if `a` is False, then `b`, else `a`                           |
| `a and b`          | if `a` is False, then `a`, else `b`                           |
| `a if cond else b` | if `cond` is True, then `a`, else `b`                         |

#### Bitwise operators

| Operation               | Result                              |
| ----------------------- | ----------------------------------- |
| `~a`                    | the bits of `a` inverted            |
| `a & b`                 | bitwise and of `a` and `b`          |
| `a ^ b`                 | bitwise exclusive or of `a` and `b` |
| <code>a &#124; b</code> | bitwise or of `a` and `b`           |

#### Trigonometric functions

```python
ti.sin(x)
ti.cos(x)
ti.tan(x)
ti.asin(x)
ti.acos(x)
ti.atan2(x, y)
ti.tanh(x)
```

#### Other arithmetic functions

```python
ti.sqrt(x)
ti.rsqrt(x)  # A fast version for `1 / ti.sqrt(x)`.
ti.exp(x)
ti.log(x)
ti.round(x)
ti.floor(x)
ti.ceil(x)
```

#### Builtin-alike functions

```python
abs(x)
max(x, y, ...)
min(x, y, ...)
pow(x, y)  # Same as `x ** y`.
```

#### Random number generator

```python
ti.random(dtype=float)
```

### Supported atomic operations

In Taichi, augmented assignments (e.g., `x[i] += 1`) are automatically
[atomic](https://en.wikipedia.org/wiki/Fetch-and-add).

:::caution

When modifying global variables in parallel, make sure you use atomic
operations. For example, to sum up all the elements in `x`,

```python
@ti.kernel
def sum():
    for i in x:
        # Approach 1: OK
        total[None] += x[i]

        # Approach 2: OK
        ti.atomic_add(total[None], x[i])

        # Approach 3: Wrong result since the operation is not atomic.
        total[None] = total[None] + x[i]
```
:::

:::note

When atomic operations are applied to local values, the Taichi compiler
will try to demote these operations into their non-atomic counterparts.
:::

Apart from the augmented assignments, explicit atomic operations, such
as `ti.atomic_add`, also do read-modify-write atomically. These
operations additionally return the **old value** of the first argument.
For example,

```python
x[i] = 3
y[i] = 4
z[i] = ti.atomic_add(x[i], y[i])
# now x[i] = 7, y[i] = 4, z[i] = 3
```

Below is a list of all explicit atomic operations:

| Operation             | Behavior                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `ti.atomic_add(x, y)` | atomically compute `x + y`, store the result in `x`, and return the old value of `x`                 |
| `ti.atomic_sub(x, y)` | atomically compute `x - y`, store the result in `x`, and return the old value of `x`                 |
| `ti.atomic_and(x, y)` | atomically compute `x & y`, store the result in `x`, and return the old value of `x`                 |
| `ti.atomic_or(x, y)`  | atomically compute <code>x &#124; y</code>, store the result in `x`, and return the old value of `x` |
| `ti.atomic_xor(x, y)` | atomically compute `x ^ y`, store the result in `x`, and return the old value of `x`                 |
| `ti.atomic_max(x, y)` | atomically compute `max(x, y)`, store the result in `x`, and return the old value of `x`             |
| `ti.atomic_min(x, y)` | atomically compute `min(x, y)`, store the result in `x`, and return the old value of `x`             |

:::note

Supported atomic operations on each backend:

| type | CPU/CUDA | OpenGL | Metal | C source |
| ---- | -------- | ------ | ----- | -------- |
| i32  | > OK     | > OK   | > OK  | > OK     |
| f32  | > OK     | > OK   | > OK  | > OK     |
| i64  | > OK     | > EXT  | > N/A | > OK     |
| f64  | > OK     | > EXT  | > N/A | > OK     |

(OK: supported; EXT: require extension; N/A: not available)
:::

### Type promotion

Binary operations on different types will give you a promoted type,
following the C programming language convention, e.g.:

- `i32 + f32 = f32` (integer + float = float)
- `i32 + i64 = i64` (less-bits + more-bits = more-bits)

Basically it will try to choose the more precise type to contain the
result value.

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

### Type casts

All data types are static in the **Taichi scope**. Therefore, casts are needed when you want to assign a certain type of data to another one.

#### Implicit casts

:::caution
The type of a variable is **determined on its initialization**.
:::

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

### Creating local variables

Compound types can be directly called to create vector, matrix or struct instances. Vectors, matrices and structs can be created using GLSL-like broadcast syntax since their shapes are already known:
```python
ray1 = my_ray3f(0.0)            # ti.Struct(ro=[0.0, 0.0, 0.0], rd=[0.0, 0.0, 0.0], l=0.0)
vec1 = my_vec3f(0.0)            # ti.Vector([0.0, 0.0, 0.0])
mat1 = my_mat2f(1.0)            # ti.Matrix([[1.0, 1.0], [1.0, 1.0]])
vec2 = my_vec3f(my_vec2i(0), 1) # ti.Vector([0.0, 0.0, 1.0]), will perform implicit cast
ray2 = my_ray3f(ro=vec1, rd=vec2, l=1.0)
```

### Supported operations

[Supported operations on primitive types](#supported-operations) can also be applied on compound types. In these cases, they are applied in an element-wise manner. For example:

```python
B = ti.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
C = ti.Matrix([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

A = ti.sin(B)
# is equivalent to
for i in ti.static(range(2)):
    for j in ti.static(range(3)):
        A[i, j] = ti.sin(B[i, j])

A = B ** 2
# is equivalent to
for i in ti.static(range(2)):
    for j in ti.static(range(3)):
        A[i, j] = B[i, j] ** 2

A = B ** C
# is equivalent to
for i in ti.static(range(2)):
    for j in ti.static(range(3)):
        A[i, j] = B[i, j] ** C[i, j]

A += 2
# is equivalent to
for i in ti.static(range(2)):
    for j in ti.static(range(3)):
        A[i, j] += 2

A += B
# is equivalent to
for i in ti.static(range(2)):
    for j in ti.static(range(3)):
        A[i, j] += B[i, j]
```
