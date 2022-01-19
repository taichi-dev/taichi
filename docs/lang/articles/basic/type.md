---
sidebar_position: 2
---

# Type system

Data types in Taichi consist of _primitive types_ and _compound types_. Primitive types are the numerical data types used by different backends, while compound types are user-defined types of data records composed of multiple members.

## Primitive types

Taichi supports common numerical data types as its primitive types. Each type is denoted as a
character indicating its _category_ followed by a number indicating its _precision bits_. The
_category_ can be either `i` (for signed integers), `u` (for unsigned integers), or `f` (for floating-point numbers). The _precision bits_ can be either `8`, `16`, `32`, or `64`,
which represents the number of **bits** for storing the data. For example, the two most commonly used types:

- `i32` represents a 32-bit signed integer;
- `f32` represents a 32-bit floating-point number.

### Supported primitive types on each backend

| type | CPU              | CUDA             | OpenGL               | Metal            |  Vulkan              |
| ---- | ---------------- | ---------------- | -------------------- | ---------------- | -------------------- |
| i8   |:heavy_check_mark:|:heavy_check_mark:|:x:                   |:heavy_check_mark:|:large_orange_diamond:|
| i16  |:heavy_check_mark:|:heavy_check_mark:|:x:                   |:heavy_check_mark:|:large_orange_diamond:|
| i32  |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:    |:heavy_check_mark:|:heavy_check_mark:    |
| i64  |:heavy_check_mark:|:heavy_check_mark:|:large_orange_diamond:|:x:               |:large_orange_diamond:|
| u8   |:heavy_check_mark:|:heavy_check_mark:|:x:                   |:heavy_check_mark:|:large_orange_diamond:|
| u16  |:heavy_check_mark:|:heavy_check_mark:|:x:                   |:heavy_check_mark:|:large_orange_diamond:|
| u32  |:heavy_check_mark:|:heavy_check_mark:|:x:                   |:heavy_check_mark:|:heavy_check_mark:    |
| u64  |:heavy_check_mark:|:heavy_check_mark:|:x:                   |:x:               |:large_orange_diamond:|
| f16  |:heavy_check_mark:|:heavy_check_mark:|:x:                   |:x:               |:heavy_check_mark:    |
| f32  |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:    |:heavy_check_mark:|:heavy_check_mark:    |
| f64  |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:    |:x:               |:large_orange_diamond:|

(:large_orange_diamond: Requiring extensions of the backend)

### Default types for integers and floating-point numbers

An integer literal, e.g., `42`, has default type `ti.i32`, while a floating-point literal,
e.g., `3.14`, has default type `ti.f32`. This behavior can be changed by explicitly specifying
default types when initializing Taichi:

```python
ti.init(default_ip=ti.i64)  # set default integer type to ti.i64
ti.init(default_fp=ti.f64)  # set default floating-point type to ti.f64
```

In addition, you can use `int` as an alias for the default integer type, and `float` as an alias
for the default floating-point type:

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

### Explicit type casting

Just like programming in other languages, you may encounter situations where you have a certain
type of data, but it is not feasible for the assignment or calculation you want to perform. In this
case, you can do *explicit type casting*. There are two kinds of explicit type casting in Taichi,
namely `ti.cast` and `ti.bit_cast`.

:::caution
In Taichi-scope, the type of a variable is **static** and **determined on its initialization**.
That is, you can never change the type of a variable. The compiler relies on this compile-time
information to check the validity of expressions in Taichi programs.
:::

#### `ti.cast`

`ti.cast()` is used for normal type casting as in other programming languages:

```python {4-5}
@ti.kernel
def foo():
    a = 3.14
    b = ti.cast(a, ti.i32)  # 3
    c = ti.cast(b, ti.f32)  # 3.0
```

You can also use `int()` and `float()` to convert values to default integer and floating-point
types:

```python {4-5}
@ti.kernel
def foo():
    a = 3.14
    b = int(a)    # 3
    c = float(b)  # 3.0
```

#### `ti.bit_cast`

Use `ti.bit_cast` to cast a value into another type **with its underlying bits preserved**:

```python {4-5}
@ti.kernel
def foo():
    a = 3.14
    b = ti.bit_cast(a, ti.i32)  # 1078523331
    c = ti.bit_cast(b, ti.f32)  # 3.14
```

Note that the new type must have the same precision bits as the old type (`i32`->`f64` is not
allowed). Use this operation with caution.

:::note
`ti.bit_cast` is equivalent to `reinterpret_cast` in C++.
:::

### Implicit type casting

When you accidentally use a value in a place where a different type is expected, implicit type
casting is triggered for the following cases.

:::caution
Relying on implicit type casting is bad practice and one major source of bugs.
:::

#### Implicit type casting in binary operations

Following the [implicit conversion rules] of the C programming language, Taichi implicitly casts
binary operation operands into a *common type* if they have different types. Some simple but most
commonly used rules are listed below:

- `i32 + f32 = f32` (int + float = float)
- `i32 + i64 = i64` (low precision bits + high precision bits = high precision bits)

#### Implicit type casting in assignments

When a value is assigned to a variable with a different type, the value is implicitly cast into that
type. If the type of the variable differs from the common type of the variable and the value, a
warning about losing precisions is raised.

In the following example, variable `a` is initialized with type `float`. On the next line, the
assignment casts `1` from `int` to `float` implicitly without any warning because the type of the
variable is the same as the common type `float`:

```python {4}
@ti.kernel
def foo():
    a = 3.14
    a = 1
    print(a)  # 1.0
```

In the following example, variable `a` is initialized with `int` type. On the next line, the
assignment casts `3.14` from `float` to `int` implicitly with a warning because the type of the
variable differs from the common type `float`:

```python {4}
@ti.kernel
def foo():
    a = 1
    a = 3.14
    print(a)  # 3
```

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

### Type casting on vectors and matrices

Type casting on vectors/matrices is element-wise:

```python {4,6}
@ti.kernel
def foo():
    u = ti.Vector([2.3, 4.7])
    v = int(u)              # ti.Vector([2, 4])
    # If you are using ti.i32 as default_ip, this is equivalent to:
    v = ti.cast(u, ti.i32)  # ti.Vector([2, 4])
```
