---
sidebar_position: 5
---

# Interacting with External Arrays


This article will show you how to transfer data from external arrays to the Taichi scope and vice versa. Here external arrays refer to NumPy arrays, PyTorch tensors, and Paddle tensors. For now, these are all the external arrays supported by Taichi.

We use NumPy arrays as an example to illustrate the data transfer process because NumPy arrays are the most commonly used external arrays in Taichi. The same steps apply to PyTorch tensors and Paddle tensors.

There are two ways to import a NumPy array `arr` to the Taichi scope:

1. Create a field `f` whose shape and dtype match the shape and dtype of `arr`, and call the method `f.from_numpy(arr)` to copy the data in `arr` into `f`. This is often used when your array is visited frequently from elsewhere in the Taichi scope in your program (e.g. sample a texture).

2. Pass `arr` as an argument to a kernel or a Taichi function using `ti.types.ndarray()` as the type hint. The argument is passed by reference without creating a copy of `arr`, and thus any modifications of this argument inside the kernel or Taichi function will also change the original array `arr`. This approach is preferred when the kernel or Taichi function that takes in the argument needs to manipulate the original array (for storage or filtering, for example).

We now explain them in more details.

## Import and export data between NumPy arrays and Taichi fields

To import data from a NumPy array to a Taichi field, firstly make sure the field and the array have the same shape:

```python
x = ti.field(float, shape=(3, 3))
a = np.arange(9).reshape(3, 3).astype(np.int32)
x.from_numpy(a)
print(x)
#[[0 1 2]
# [3 4 5]
# [6 7 8]]
```

In the above exmaple, the scalar field `x` and the array `a` both have shape `(3, 3)`. You won't be able to perform this operation if their shapes don't match. For vector and matrix fields their shape matching rules with NumPy arrays are a bit subtle and will be discussed in a later section.

The field should also have the same dtype with the array, otherwise an implicit type casting will be performed, see [type system](../type_system/type.md).

To export the data in `x` to a NumPy array, simply call its `to_numpy()` method:

```python
arr = x.to_numpy()
#array([[0, 1, 2],
#       [3, 4, 5],
#       [6, 7, 8]], dtype=int32)
```

## Import and export data between PyTorch/Paddle tensors and Taichi fields

To import data from a PyTorch tensor, simply replace the `from_numpy()` call by `from_torch()`, and replace `to_numpy()` by `to_torch()` to export data to a PyTorch tensor. But note `to_torch()` requires one more argument here: you need also specify the PyTorch device using the `device` argument:

```python
tensor = x.to_torch(device="cuda:0")
print(tensor.device) # device(type='cuda', index=0)
```

Likewise for Paddle, you need to specify the device by `paddle.CPUPlace()` or `paddle.CUDAPlace(n)` where `n` is an optional ID, the default is 0.


## External array shapes

As mentioned before, when importing/exporing data between a `ti.field/ti.Vector.field/ti.Matrix` and a NumPy array, you need to make sure the shape of the field matches the corresponding array. The matching rule is summarized below:

- For scalar fields, **the shape of NumPy array, PyTorch tensor or Paddle Tensor equals the shape of the Taichi field**

    ```python
    field = ti.field(int, shape=(256, 512))
    field.shape  # (256, 512)

    array = field.to_numpy()
    array.shape  # (256, 512)

    field.from_numpy(array)  # the input array must be of shape (256, 512)
    ```

    An illustration is shown below:

    ```
                                   field.shape[1]=array.shape[1]
                                               (=512)
                                      ┌───────────────────────┐

                                   ┌  ┌───┬───┬───┬───┬───┬───┐  ┐
                                   │  │   │   │   │   │   │   │  │
                                   │  ├───┼───┼───┼───┼───┼───┤  │
    field.shape[0]=array.shape[0]  │  │   │   │   │   │   │   │  │
             (=256)                │  ├───┼───┼───┼───┼───┼───┤  │
                                   │  │   │   │   │   │   │   │  │
                                   └  └───┴───┴───┴───┴───┴───┘  ┘
    ```

- For vector fields, if the vector is `n`-dimensional, then **the shape of NumPy array, PyTorch tensor or Paddle Tensor should be** `(*field_shape, n)`:

    ```python
    field = ti.Vector.field(3, int, shape=(256, 512))
    field.shape  # (256, 512)
    field.n      # 3

    array = field.to_numpy()
    array.shape  # (256, 512, 3)

    field.from_numpy(array)  # the input array must be of shape (256, 512, 3)
    ```

    An illustration is shown below:

    ```
                                     field.shape[1]=array.shape[1]
                                                (=512)
                                     ┌─────────────────────────────┐

                                  ┌  ┌─────────┬─────────┬─────────┐  ┐
                                  │  │[*, *, *]│[*, *, *]│[*, *, *]│  │
                                  │  ├─────────┼─────────┼─────────┤  │
    field.shape[0]=array.shape[0] │  │[*, *, *]│[*, *, *]│[*, *, *]│  │        [*, *, *]
             (=256)               │  ├─────────┼─────────┼─────────┤  │        └───────┘
                                  │  │[*, *, *]│[*, *, *]│[*, *, *]│  │   n=array.shape[2]=3
                                  └  └─────────┴─────────┴─────────┘  ┘
    ```

- For matrix fields, if the matrix is `n`-by-`m` (`n x m`), then **the shape of NumPy array, PyTorch tensor or Paddle Tensor should be** `(*field_shape, n, m)`:

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
            a[i, j] = i + j

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
