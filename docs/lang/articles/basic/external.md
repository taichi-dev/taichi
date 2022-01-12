---
sidebar_position: 5
---

# Interacting with external arrays

Although Taichi fields are mainly used in Taichi-scope, in some cases
efficiently manipulating Taichi field data in Python-scope could also be
helpful.

We provide various interfaces to copy the data between Taichi fields and
external arrays. External arrays refer to NumPy arrays or PyTorch tensors.
Let's take a look at the most common usage: interacting with NumPy arrays.

**Export data in Taichi fields to NumPy arrays** via `to_numpy()`. This
allows us to export computation results to other Python packages that
support NumPy, e.g. `matplotlib`.

```python {8}
@ti.kernel
def my_kernel():
	for i in x:
		x[i] = i * 2

x = ti.field(ti.f32, 4)
my_kernel()
x_np = x.to_numpy()
print(x_np)  # np.array([0, 2, 4, 6])
```

**Import data from NumPy arrays to Taichi fields** via `from_numpy()`.
This allows us to initialize Taichi fields via NumPy arrays:

```python {3}
x = ti.field(ti.f32, 4)
x_np = np.array([1, 7, 3, 5])
x.from_numpy(x_np)
print(x[0])  # 1
print(x[1])  # 7
print(x[2])  # 3
print(x[3])  # 5
```

## External array shapes

Shapes of Taichi fields and those of corresponding NumPy arrays are closely
connected via the following rules:

- For scalar fields, **the shape of NumPy array is exactly the same as
  the Taichi field**:

```python
field = ti.field(ti.i32, shape=(256, 512))
field.shape  # (256, 512)

array = field.to_numpy()
array.shape  # (256, 512)

field.from_numpy(array)  # the input array must be of shape (256, 512)
```

- For vector fields, if the vector is `n`-D, then **the shape of NumPy
  array should be** `(*field_shape, vector_n)`:

```python
field = ti.Vector.field(3, ti.i32, shape=(256, 512))
field.shape  # (256, 512)
field.n      # 3

array = field.to_numpy()
array.shape  # (256, 512, 3)

field.from_numpy(array)  # the input array must be of shape (256, 512, 3)
```

- For matrix fields, if the matrix is `n`-by-`m` (`n x m`), then **the shape of NumPy
array should be** `(*field_shape, matrix_n, matrix_m)`:

```python
field = ti.Matrix.field(3, 4, ti.i32, shape=(256, 512))
field.shape  # (256, 512)
field.n      # 3
field.m      # 4

array = field.to_numpy()
array.shape  # (256, 512, 3, 4)

field.from_numpy(array)  # the input array must be of shape (256, 512, 3, 4)
```

- For struct fields, the external array will be exported as **a dictionary of arrays** with the keys being struct member names and values being struct member arrays. Nested structs will be exported as nested dictionaries:

```python
field = ti.Struct.field({'a': ti.i32, 'b': ti.types.vector(float, 3)} shape=(256, 512))
field.shape # (256, 512)

array_dict = field.to_numpy()
array_dict.keys() # dict_keys(['a', 'b'])
array_dict['a'].shape # (256, 512)
array_dict['b'].shape # (256, 512, 3)

field.from_numpy(array_dict) # the input array must have the same keys as the field
```

## Using external arrays as Taichi kernel arguments

Use the type hint `ti.ext_arr()` for passing external arrays as kernel
arguments. For example:

```python {10}
import taichi as ti
import numpy as np

ti.init()

n, m = 4, 7


@ti.kernel
def test_numpy(arr: ti.ext_arr()):
    for i in range(n):
        for j in range(m):
            arr[i, j] += i + j


a = np.empty(shape=(n, m), dtype=np.int32)

for i in range(n):
    for j in range(m):
        a[i, j] = i * j

test_numpy(a)

for i in range(n):
    for j in range(m):
        assert a[i, j] == i * j + i + j
```
