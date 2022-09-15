---
sidebar_position: 5
---

# Interacting with External Arrays


This article will show you how to transfer data from external arrays to the Taichi scope and vice versa. Here external arrays refer to NumPy arrays, PyTorch tensors or Paddle Tensors, for now these are the only supported arrays.

We will use NumPy arrays as an example to illustrate the procedure since this is the most common usage. The cases for PyTorch tentors and Paddle tensors are very similar.

There are two ways to import a NumPy array `arr` to the Taichi scope:

1. Create a field `f` whose shape and dtype match the shape and dtype of `arr`, and call the method `f.from_numpy(arr)` to copy the data in `arr` into `f`.

2. Pass `arr` as an argument to a kernel or a Taichi function using `ti.types.ndarray()` as the type hint. This argument will be passed by reference and won't invoke a copy of `arr`, and modificaitions on this argument inside the kernel will also take affect on the original array `arr`.

These two ways should suit in different cases. If your array will be visited frequently from elsewhere in your program (e.g. sample a texture), it's best to use 1; or if you have a kernel that serves a special purpose of processing arrays (e.g. soring or filtering), it's best to use 2.

We now explain them in more details.

## Import and export data between NumPy arrays and Taichi fields

To import data from a NumPy array to a Taichi field, you need to make sure the field and the array have the same shape:

```python
x = ti.field(float, shape=(3, 3))
a = np.arange(9).reshape(3, 3).astype(np.int32)
x.from_numpy(a)
print(x)
#[[0 1 2]
# [3 4 5]
# [6 7 8]]
```
The field should also have the same dtype with the array, otherwise an implicit type casting will be performed, see [type system](../type_system/type.md).

To export the data in `x` to a NumPy array, simply call its `to_numpy()` method:

```python
arr = x.to_numpy()
#array([[0, 1, 2],
#       [3, 4, 5],
#       [6, 7, 8]], dtype=int32)
```

## Import and export data between PyTorch/Paddle tensors and Taichi fields

To import data from a PyTorch tensor, simply replace the `from_numpy()` call by `from_torch()`. And replace `to_numpy()` by `to_torch()` to export data to a PyTorch tensor. But note when calling `to_torch()` to export data to a PyTorch tensor, you need also specify the PyTorch device using the `device` argument:

```python
tensor = x.to_torch(device="cuda:0")
print(tensor.device) # device(type='cuda', index=0)
```

Likewise for Paddle, you need to specify the device by `paddle.CPUPlace()` or `paddle.CUDAPlace(n)` where `n` is an optional ID, the default is 0.


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
field = ti.Struct.field({'a': ti.i32, 'b': ti.types.vector(3, float)}, shape=(256, 512))
field.shape # (256, 512)

array_dict = field.to_numpy()
array_dict.keys() # dict_keys(['a', 'b'])
array_dict['a'].shape # (256, 512)
array_dict['b'].shape # (256, 512, 3)

field.from_numpy(array_dict) # the input array must have the same keys as the field
```

## Using external arrays as Taichi kernel arguments

You can use type hint `ti.types.ndarray()` to pass external arrays as kernel arguments. For example:

```python {10}
import taichi as ti
import numpy as np
ti.init()

a = np.zeros((5, 5))
    
@ti.kernel
def test(a: ti.types.ndarray()):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] += i + j

test()
print(a)
```

Note that elements in an external array must be indexed using a single square bracket. This contrasts with a Taichi vector or matrix field where field and matrix indices are indexed separately:

```python
x = ti.Vector.field(3, float, shape=(5, 5))
y = np.random.random((5, 5, 3))

@ti.kernel
def copy_vector(x: ti.template(), y: ti.types.ndarray()):
    for i, j in ti.ndrange(5, 5):
        for k in ti.static(range(3)):
            y[i, j, k] = x[i, j][k] # correct
            # y[i][j][k] = x[i, j][k] incorrect
            # y[i, j][k] = x[i, j][k] incorrect
```

Also, external arrays in a Taichi kernel are indexed using its **physical memory layout**. For PyTorch users, this implies that the PyTorch tensor [needs to be made contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) before being passed into a Taichi kernel:

```python
x = ti.field(dtype=int, shape=(3, 3))
y = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = y.T # Transposing the tensor returns a view of the tensor which is not contiguous

@ti.kernel
def copy_scalar(x: ti.template(), y: ti.types.ndarray()):
    for i, j in x:
        y[i, j] = x[i, j]

copy(x, y) # error!
copy(x, y.clone()) # correct
copy(x, y.contiguous()) # correct
```