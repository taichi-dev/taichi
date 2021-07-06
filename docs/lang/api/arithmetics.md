---
sidebar_position: 4
---

# Scalar operations

## Operators

### Arithmetic operators

|     Operation   |               Result            |
| :-------------- | :-------------------------------|
|      `-a`       |  `a`negated                     |
|      `+a`       |  `a`unchanged                   |
|      `a + b`    |  sum of `a` and `b`             |
|      `a - b`    |  difference of `a` and `b`      |
|      `a * b`    |  product of `a` and `b`         |
|      `a / b`    |  quotient of `a` and `b`        |
|      `a // b`   |  floored quotient of `a` and `b`|
|      `a % b`    |  remainder of `a` / `b`         |
|      `a ** b`   |  `a` to the power `b`           |

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

Python 3 distinguishes `/` (true division) and `//` (floor division).
For example, `1.0 / 2.0 = 0.5`, `1 / 2 = 0.5`, `1 // 2 = 0`,
`4.2 // 2 = 2`. And Taichi follows the same design:

- **true divisions** on integral types will first cast their
  operands to the default float point type.
- **floor divisions** on float-point types will first cast their
  operands to the default integer type.

To avoid such implicit casting, you can manually cast your operands to
desired types, using `ti.cast`. Please see
[Default precisions](../articles/basic/type.md#default-precisions) for more details on
default numerical types.
:::

### Logic operators

|      Operation    |               Result            |
| :---------------  | :----------- ----------------------------------------|
| `a == b`          | if `a` equal `b`, then True, else False              |  
| `a != b`          | if `a` not equal `b`, then True, else False          |        
| `a > b`           | if `a` strictly greater than `b`, then True, else False  |     
| `a < b`           | if `a` strictly less than `b`, then True, else False |    
| `a >= b`          | if `a` greater than or equal `b`, then True, else False |         
| `a <= b`          | if `a` less than or equal `b`, then True, else False |            
| `not a`           | if `a` is False, then True, else False               |     
| `a or b`          | if `a` is False, then `b`, else `a`                  |  
| `a and b`         | if `a` is False, then `a`, else `b`                  |        
| `a if cond else b`| if `cond` is True, then `a`, else `b`                |

### Bitwise operators
|      Operation    |               Result            |
| :---------------  | :----------- ----------------------------------------|
| `~a`              | the bits of `a` inverted               |
| `a & b`|  bitwise and of `a` and  `b`                      |
| `a ^ b`|  bitwise exclusive or of `a` and `b`             |
| `a \| b`|  bitwise or of `a` and `b`                      |


## Functions

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

ti.floor(x)

ti.ceil(x)
```

### Casting types

```python
ti.cast(x, dtype)
```

See [Type system](../articles/basic/type.md#type-system) for more details.

```python
int(x)
```

A shortcut for `ti.cast(x, int)`.

```python
float(x)
```

A shortcut for `ti.cast(x, float)`.

### Builtin-alike functions

```python
abs(x)

max(x, y, \...)

min(x, y, \...)

pow(x, y)  # Same as `x ** y`.
```

### Random number generator

```python
ti.random(dtype = float)
```

## Element-wise arithmetics for vectors and matrices

When these scalar functions are applied on [Matrices](./matrix.md) and [Vectors](./vector.md), they are applied in an element-wise manner. For example:

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
