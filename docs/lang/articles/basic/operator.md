---
sidebar_position: 4
---

# Operators
Here we present the supported operators in Taichi for both primitive types and
compound types such as matrices.

## Supported operators for primitive types
### Arithmetic operators

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

For C-style mod (`%`), please use `ti.raw_mod`. This function also receives floating points as arguments.

`ti.raw_mod(a, b)` returns `a - b * int(float(a) / b)`.

```python
print(ti.raw_mod(2, 3))      # 2
print(ti.raw_mod(-2, 3))     # -2
print(ti.raw_mod(3.5, 1.5))  # 0.5
```
:::

:::note

Python3 distinguishes `/` (true division) and `//` (floor division), e.g., `1.0 / 2.0 = 0.5`, `1 / 2 = 0.5`, `1 // 2 = 0`,
`4.2 // 2 = 2`. Taichi follows the same design:

- **True divisions** on integral types first cast their
  operands to the default floating point type.
- **Floor divisions** on floating point types first cast their
  operands to the default integral type.

To avoid such implicit casting, you can manually cast your operands to
desired types, using `ti.cast`. Please see
[Default precisions](#default-precisions) for more details on
default numerical types.

Taichi also provides `ti.raw_div` function which performs true division if one of the operands is floating point type
and performs floor division if both operands are integral types.

```python
print(ti.raw_div(5, 2))    # 2
print(ti.raw_div(5, 2.0))  # 2.5
```

:::


### Comparison operators

| Operation          | Result                                                        |
| ------------------ | ------------------------------------------------------------- |
| `a == b`           | if `a` is equal to `b`, then True, else False                 |
| `a != b`           | if `a` is not equal to `b`, then True, else False             |
| `a > b`            | if `a` is strictly greater than `b`, then True, else False    |
| `a < b`            | if `a` is strictly less than `b`, then True, else False       |
| `a >= b`           | if `a` is greater than or equal to `b`, then True, else False |
| `a <= b`           | if `a` is less than or equal to `b`, then True, else False    |

### Logical operators

| Operation          | Result                                                        |
| ------------------ | ------------------------------------------------------------- |
| `not a`            | if `a` is False, then True, else False                        |
| `a or b`           | if `a` is False, then `b`, else `a`                           |
| `a and b`          | if `a` is False, then `a`, else `b`                           |

### Conditional operations

The result of conditional expression `a if cond else b` is `a` if `cond` is True, or `b` otherwise.
`a` and `b` must have a same type.

The conditional expression does short-circuit evaluation, which means the branch not chosen is not evaluated.

```python
a = ti.field(ti.i32, shape=(10,))
for i in range(10):
    a[i] = i

@ti.kernel
def cond_expr(ind: ti.i32) -> ti.i32:
    return a[ind] if ind < 10 else 0

cond_expr(3)  # returns 3
cond_expr(10)  # returns 0, a[10] is not evaluated
```


For element-wise conditional operations on Taichi vectors and matrices,
Taichi provides `ti.select(cond, a, b)` which **does not** do short-circuit evaluation.
```python {4}
cond = ti.Vector([1, 0])
a = ti.Vector([2, 3])
b = ti.Vector([4, 5])
ti.select(cond, a, b)  # ti.Vector([2, 5])
```

### Bitwise operators

| Operation               | Result                              |
| ----------------------- | ----------------------------------- |
| `~a`                    | the bits of `a` inverted            |
| `a & b`                 | bitwise and of `a` and `b`          |
| `a ^ b`                 | bitwise exclusive or of `a` and `b` |
| <code>a &#124; b</code> | bitwise or of `a` and `b`           |
| `a << b`                | left-shift `a` by `b` bits          |
| `a >> b`                | right-shift `a` by `b` bits         |

:::note

The `>>` operation denotes the
[Shift Arithmetic](https://en.wikipedia.org/wiki/Arithmetic_shift) Right (SAR) operation.
For the [Shift Logical](https://en.wikipedia.org/wiki/Logical_shift) Right (SHR) operation,
consider using `ti.bit_shr()`. For left shift operations, SAL and SHL are the
same.


:::

### Trigonometric functions

```python
ti.sin(x)
ti.cos(x)
ti.tan(x)
ti.asin(x)
ti.acos(x)
ti.atan2(x, y)
ti.tanh(x)
```

### Other arithmetic functions

```python
ti.sqrt(x)
ti.rsqrt(x)  # A fast version for `1 / ti.sqrt(x)`.
ti.exp(x)
ti.log(x)
ti.round(x)
ti.floor(x)
ti.ceil(x)
ti.sum(x)
ti.max(x, y, ...)
ti.min(x, y, ...)
ti.abs(x)  # Same as `abs(x)`
ti.pow(x, y)  # Same as `pow(x, y)` and `x ** y`
```

### Builtin-alike functions

```python
abs(x)  # Same as `ti.abs(x, y)`
pow(x, y)  # Same as `ti.pow(x, y)` and `x ** y`.
```

### Random number generator

```python
ti.random(dtype=float)
```

:::note

`ti.random` supports `u32`, `i32`, `u64`, `i64`, and all floating point types.
The range of the returned value is type-specific.

| Type | Range |
| --- | --- |
| i32 | -2,147,483,648 to 2,147,483,647 |
| u32 | 0 to 4,294,967,295 |
| i64 | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 |
| u64 | 0 to 18,446,744,073,709,551,615 |
| floating point | 0.0 to 1.0 |

:::

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

| type | CPU | CUDA | OpenGL | Metal | C source |
| ---- | ---- | ---- | ------ | ----- | -------- |
| i32  |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| f32  |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| i64  |:heavy_check_mark:|:heavy_check_mark:|:o:|:x:|:heavy_check_mark:|
| f64  |:heavy_check_mark:|:heavy_check_mark:|:o:|:x:|:heavy_check_mark:|

(:o: Requiring extensions for the backend.)
:::


## Supported operators for matrices

The previously mentioned operations on primitive types can also be applied on
compound types such as matrices.
In these cases, they are applied in an element-wise manner. For example:

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

In addition, the following methods are supported matrices operations:

```python
a = ti.Matrix([[2, 3], [4, 5]])
a.transpose()   # the transposed matrix of `a`, will not effect the data in `a`.
a.trace()       # the trace of matrix `a`, the returned scalar value can be computed as `a[0, 0] + a[1, 1] + ...`.
a.determinant() # the determinant of matrix `a`.
a.inverse()     # (ti.Matrix) the inverse of matrix `a`.
a@a             # @ denotes matrix multiplication
```

:::note
For now, determinant() and inverse() only works in Taichi-scope, and the
size of the matrix must be 1x1, 2x2, 3x3 or 4x4.
:::
