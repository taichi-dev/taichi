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

For C-style mod (`%`), please use `ti.raw_mod`:

```python
print(ti.raw_mod(2, 3))   # 2
print(ti.raw_mod(-2, 3))  # -2
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
:::


### Logic operators

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

### Bitwise operators

| Operation               | Result                              |
| ----------------------- | ----------------------------------- |
| `~a`                    | the bits of `a` inverted            |
| `a & b`                 | bitwise and of `a` and `b`          |
| `a ^ b`                 | bitwise exclusive or of `a` and `b` |
| <code>a &#124; b</code> | bitwise or of `a` and `b`           |
| `a << b`                | left-shift `a` by `b` bits          |
| `a >> b`                | right-shift `a` by `b` bits         |

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
```

### Builtin-alike functions

```python
abs(x)
max(x, y, ...)
min(x, y, ...)
pow(x, y)  # Same as `x ** y`.
```

### Random number generator

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
| i32  |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| f32  |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| i64  |:heavy_check_mark:|:large_orange_diamond:|:x:|:heavy_check_mark:|
| f64  |:heavy_check_mark:|:large_orange_diamond:|:x:|:heavy_check_mark:|

(:large_orange_diamond: requires extension)
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
