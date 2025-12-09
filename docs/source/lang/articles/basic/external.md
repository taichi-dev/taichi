---
sidebar_position: 5
---

# Interacting with External Arrays


This document provides instructions on how to transfer data from external arrays to the Taichi scope and vice versa. For now, the external arrays supported by Taichi are NumPy arrays, PyTorch tensors, and Paddle tensors.

We use NumPy arrays as an example to illustrate the data transfer process because NumPy arrays are the most commonly used external arrays in Taichi. The same steps apply to PyTorch tensors and Paddle tensors.

There are two ways to import a NumPy array `arr` to the Taichi scope:

- Create a Taichi field `f`, whose shape and dtype match the shape and dtype of `arr`, and call `f.from_numpy(arr)` to copy the data in `arr` into `f`. This approach is preferred when the original array is visited frequently from elsewhere in the Taichi scope (for example, in texture sampling).

- Pass `arr` as an argument to a kernel or a Taichi function using `ti.types.ndarray()` as type hint. The argument is passed by reference without creating a copy of `arr`. Thus, any modification to this argument from inside a kernel or Taichi function also changes the original array `arr`. This approach is preferred when the kernel or Taichi function that takes in the argument needs to process the original array (for storage or filtering, for example).

:::note
`from_numpy() / from_torch()` can take in any numpy array or torch Tensor, no matter it's contiguous or not. Taichi will manage its own copy of data. However, when passing an argument to a Taichi kernel, only contiguous numpy arrays or torch Tensors are supported.
:::

## Data transfer between NumPy arrays and Taichi fields

To import data from a NumPy array to a Taichi field, first make sure that the field and the array have the same shape:

```python
x = ti.field(float, shape=(3, 3))
a = np.arange(9).reshape(3, 3).astype(np.int32)
x.from_numpy(a)
print(x)
#[[0 1 2]
# [3 4 5]
# [6 7 8]]
```

In the example above, the scalar field `x` and the array `a` have the same shape `(3, 3)`. This operation would fail if their shapes did not match. Shape matching of a vector or matrix field with a NumPy array is slightly different, which will be discussed in a later section.

The field should also have the same dtype as the array; otherwise, an implicit type casting would occur - see [type system](../type_system/type.md).

Conversely, to export the data in `x` to a NumPy array, call `to_numpy()`:

```python cont
arr = x.to_numpy()
#array([[0, 1, 2],
#       [3, 4, 5],
#       [6, 7, 8]], dtype=int32)
```

## Data transfer between PyTorch/Paddle tensors and Taichi fields

Data transfer between a PyTorch tensor and a Taichi field is similar to the NumPy case above: Call `from_torch()` for data import and `to_torch()` for data export. But note that `to_torch()` requires one more argument `device`, which specifies the PyTorch device:

```python cont
tensor = x.to_torch(device="cuda:0")
print(tensor.device) # device(type='cuda', index=0)
```
For Paddle, you need to specify the device by calling `paddle.CPUPlace()` or `paddle.CUDAPlace(n)`, where `n` is an optional ID set to 0 by default.

```python skip-ci:NotTestingPaddle
device = paddle.CPUPlace()
tensor = x.to_paddle(device=device)
```

## External array shapes

When transferring data between a `ti.field/ti.Vector.field/ti.Matrix.field` and a NumPy array, you need to make sure that the shapes of both sides are aligned. The shape matching rules are summarized as below:

1. When importing data to or exporting data from a scalar field, ensure that **the shape of the corresponding NumPy array, PyTorch tensor, or Paddle tensor equals the shape of the scalar field**

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

2. When importing data to or exporting data from an `n`-dimensional vector field, ensure that **the shape of the corresponding NumPy array, PyTorch tensor, or Paddle tensor is set to** `(*field_shape, n)`:

    ```python
    field = ti.Vector.field(3, int, shape=(256, 512))
    field.shape  # (256, 512)
    field.n      # 3

    array = field.to_numpy()
    array.shape  # (256, 512, 3)

    field.from_numpy(array)  # the input array must in the shape (256, 512, 3)
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

3. When importing data to or exporting data from an `n`-by-`m` (`n x m`) matrix field,  ensure that **the shape of the corresponding NumPy array, PyTorch tensor, or Paddle tensor is set to** `(*field_shape, n, m)`:

    ```python
    field = ti.Matrix.field(3, 4, ti.i32, shape=(256, 512))
    field.shape  # (256, 512)
    field.n      # 3
    field.m      # 4

    array = field.to_numpy()
    array.shape  # (256, 512, 3, 4)

    field.from_numpy(array)  # the input array must be of shape (256, 512, 3, 4)
    ```

4. When importing data to a struct field, export the data of the corresponding external array as **a dictionary of NumPy arrays, PyTorch tensors, or Paddle tensors** with keys being struct member names and values being struct member arrays. Nested structs are exported as nested dictionaries:

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

Use type hint `ti.types.ndarray()` to pass external arrays as kernel arguments.

### An entry-level example

The following example shows the most basic way to call `ti.types.ndarray()`:

```python {10}
import taichi as ti
import numpy as np
ti.init()

a = np.zeros((5, 5))

@ti.kernel
def test(a: ti.types.ndarray()):
    for i in range(a.shape[0]):  # a parallel for loop
        for j in range(a.shape[1]):
            a[i, j] = i + j

test(a)
print(a)
```

### Advanced usage


Assume that `a` and `b` are both 2D arrays of the same shape and dtype. For each cell `(i, j)` in `a`, we want to calculate the difference between its value and the average of its four neighboring cells while storing the result in the corresponding cell in `b`. In this case, cells on the boundary, which are cells with fewer than four neighbors, are ruled out for simplicity. This operation is usually denoted as the *Discrete Laplace Operator*:

```python skip-ci:Trivial
b[i, j] = a[i, j] - (a[i-1, j] + a[i, j-1] + a[i+1, j] + a[i, j+1]) / 4
```

Such an operation is typically very slow, even with NumPy's vectorization as shown below:

```python skip-ci:NumpyOnly
b[1:-1, 1:-1] += (               a[ :-2, 1:-1] +
                  a[1:-1, :-2]                 + a[1:-1, 2:] +
                                 a[2:  , 1:-1])
```

But Taichi can meet the same purpose in one parallel `for` loop only:

```python
@ti.kernel
def test(a: ti.types.ndarray(), b: ti.types.ndarray()):  # assume a, b have the same shape
    H, W = a.shape[0], a.shape[1]
    for i, j in ti.ndrange(H, W):  # one parallel for loop
        if 0 < i < H - 1 and 0 < j < W - 1:
            b[i, j] = a[i, j] - (a[i-1, j] + a[i, j-1] + a[i+1, j] + a[i, j+1]) / 4
```

Not only is this code snippet more readable than the NumPy version above, but it also runs way faster even on the CPU backend.

:::note
The elements in an external array must be indexed using a single square bracket. This contrasts with a Taichi vector field or matrix field where field members and elements are indexed separately:
:::

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

In addition, external arrays in a Taichi kernel are indexed using their **physical memory layout**. For PyTorch users, this means that a PyTorch tensor [needs to be made contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) before being passed into a Taichi kernel:

```python known-error:CopyNotDefined
x = ti.field(dtype=int, shape=(3, 3))
y = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = y.T # Transposing the tensor returns a view of the tensor which is not contiguous

@ti.kernel
def copy_scalar(x: ti.template(), y: ti.types.ndarray()):
    for i, j in x:
        y[i, j] = x[i, j]

# copy(x, y) # error!
copy(x, y.clone()) # correct
copy(x, y.contiguous()) # correct
```

## FAQ

### Can I use `@ti.kernel` to accelerate a NumPy function?

Unlike other Python acceleration frameworks, such as Numba, Taichi does not compile NumPy functions. Calling NumPy functions inside the Taichi scope is not supported, as the following example shows:

```python
import numpy as np

@ti.kernel
def invalid_sum(arr: ti.types.ndarray()):
    total = np.sum(arr)  # Not supported!
    ...
```


If you want to use a NumPy function, which lacks a counterpart in Taichi, you can call the function in the Python scope as usual and pass the processed array to Taichi kernels via `ti.types.ndarray()`. For example:

```python
arr = np.random.random(233)
indices = np.argsort(arr)  # arr is a NumPy ndarray

@ti.kernel
def valid_example(arr: ti.types.ndarray(), indices: ti.types.ndarray()):
    min_element = arr[indices[0]]
    ...
```
