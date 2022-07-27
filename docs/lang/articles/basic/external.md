---
sidebar_position: 5
---

# Interacting with External Arrays

Although Taichi fields are mainly used in Taichi-scope, in some cases efficiently manipulating Taichi field data in Python-scope could also be
helpful.

We provide various interfaces to copy the data between Taichi fields and external arrays. External arrays refer to NumPy arrays, PyTorch tensors or Paddle Tensors. Let's take a look at the most common usage: interacting with NumPy arrays.

**Export data in Taichi fields to NumPy arrays** via `to_numpy()`. This allows us to export computation results to other Python packages that support NumPy, e.g. `matplotlib`.

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

**Import data from NumPy arrays to Taichi fields** via `from_numpy()`. This allows us to initialize Taichi fields via NumPy arrays:

```python {3}
x = ti.field(ti.f32, 4)
x_np = np.array([1, 7, 3, 5])
x.from_numpy(x_np)
print(x[0])  # 1
print(x[1])  # 7
print(x[2])  # 3
print(x[3])  # 5
```

Likewise, Taichi fields can be **imported from and exported to PyTorch tensors**:
```python
@ti.kernel
def my_kernel():
    for i in x:
        x[i] = i * 2

x = ti.field(ti.f32, 4)
my_kernel()
x_torch = x.to_torch()
print(x_torch)  # torch.tensor([0, 2, 4, 6])

x.from_numpy(torch.tensor([1, 7, 3, 5]))
print(x[0])  # 1
print(x[1])  # 7
print(x[2])  # 3
print(x[3])  # 5
```
And Taichi fields also can be **imported from and exported to Paddle tensors**:

```python
@ti.kernel
def my_kernel():
    for i in x:
        x[i] = i * 2

x = ti.field(ti.f32, 4)
my_kernel()
x_paddle = x.to_paddle()
print(x_paddle)  # paddle.Tensor([0, 2, 4, 6])

x.from_numpy(paddle.to_tensor([1, 7, 3, 5]))
print(x[0])  # 1
print(x[1])  # 7
print(x[2])  # 3
print(x[3])  # 5
```

When calling `to_torch()`, specify the PyTorch device where the Taichi field is exported using the `device` argument:

```python
x = ti.field(ti.f32, 4)
x.fill(3.0)
x_torch = x.to_torch(device="cuda:0")
print(x_torch.device) # device(type='cuda', index=0)
```

For Paddle, specify the device by `paddle.CPUPlace()` or `paddle.CUDAPlace(n)` where n is an optional ID, default is 0.

## External array shapes

Shapes of Taichi fields and those of corresponding NumPy arrays, PyTorch tensors or Paddle Tensors are closely connected via the following rules:

- For scalar fields, **the shape of NumPy array, PyTorch tensor or Paddle Tensor equals the shape of the Taichi field**

```python
field = ti.field(ti.i32, shape=(256, 512))
field.shape  # (256, 512)

array = field.to_numpy()
array.shape  # (256, 512)

field.from_numpy(array)  # the input array must be of shape (256, 512)
```

- For vector fields, if the vector is `n`-D, then **the shape of NumPy array, PyTorch tensor or Paddle Tensor should be** `(*field_shape, vector_n)`:

```python
field = ti.Vector.field(3, ti.i32, shape=(256, 512))
field.shape  # (256, 512)
field.n      # 3

array = field.to_numpy()
array.shape  # (256, 512, 3)

field.from_numpy(array)  # the input array must be of shape (256, 512, 3)
```

- For matrix fields, if the matrix is `n`-by-`m` (`n x m`), then **the shape of NumPy array, PyTorch tensor or Paddle Tensor should be** `(*field_shape, matrix_n, matrix_m)`:

```python
field = ti.Matrix.field(3, 4, ti.i32, shape=(256, 512))
field.shape  # (256, 512)
field.n      # 3
field.m      # 4

array = field.to_numpy()
array.shape  # (256, 512, 3, 4)

field.from_numpy(array)  # the input array must be of shape (256, 512, 3, 4)
```

- For struct fields, the external array will be exported as **a dictionary of NumPy arrays, PyTorch tensors or Paddle Tensors** with keys being struct member names and values being struct member arrays. Nested structs will be exported as nested dictionaries:

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

Use type hint `ti.types.ndarray()` to pass external arrays as kernel arguments. For example:

```python {10}
import taichi as ti
import numpy as np

ti.init()

n, m = 4, 7
a = np.empty(shape=(n, m), dtype=np.int32)


@ti.kernel
def test_numpy(arr: ti.types.ndarray()):
    # You can access the shape of the passed array in the kernel
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] += i + j


for i in range(n):
    for j in range(m):
        a[i, j] = i * j

test_numpy(a)

for i in range(n):
    for j in range(m):
        assert a[i, j] == i * j + i + j
```

Note that the elements in an external array must be indexed using a single square bracket. This contrasts with a Taichi vector or matrix field where field and matrix indices are indexed separately:
```python
@ti.kernel
def copy_vector(x: ti.template(), y: ti.types.ndarray()):
    for i, j in ti.ndrange(n, m):
        for k in ti.static(range(3)):
            y[i, j, k] = x[i, j][k] # correct
            # y[i][j][k] = x[i, j][k] incorrect
            # y[i, j][k] = x[i, j][k] incorrect
```
Also, external arrays in a Taichi kernel are indexed using its **physical memory layout**. For PyTorch users, this implies that the PyTorch tensor [needs to be made contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) before being passed into a Taichi kernel:

```python
@ti.kernel
def copy_scalar(x: ti.template(), y: ti.types.ndarray()):
    for i, j in x:
        y[i, j] = x[i, j]

x = ti.field(dtype=int, shape=(3, 3))
y = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = y.T # Transposing the tensor returns a view of the tensor which is not contiguous
copy(x, y) # error!
copy(x, y.clone()) # correct
copy(x, y.contiguous()) # correct
```
