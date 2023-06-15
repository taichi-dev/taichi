---
sidebar_position: 3
---

# Taichi Ndarray

The Taichi ndarray is an array object that holds contiguous multi-dimensional data. Generally speaking, it plays a similar role to its counterpart `numpy.ndarray` in NumPy, but its underlying memory is allocated on the user-specified Taichi arch and managed by Taichi runtime.

## When to use ndarray

You can use fields as data containers in most cases. However, fields might have very complicated tree-structured layouts with even sparsity in them. It is hard for external libraries to interpret or use computed results stored in `ti.field` directly. An ndarray, however, always allocates a contiguous memory block to allow straightforward data exchange with external libraries.

In a nutshell, fields are mainly used for maximizing performance with complex data layouts. If you process dense data only or need external interop, just move forward with ndarrays!

## Python scope usages

The statement below instantiates an instance of a Taichi ndarray. `dtype` refers to the data type, which can either be a scalar data type like `ti.f32/ti.i32` or a vector/matrix data type like `ti.math.vec2/mat2`. `shape` denotes the array size with respect to the data type. Like fields, ndarray can only be constructed in the Python scope rather than in the Taichi scope. That is to say, an ndarray cannot be constructed inside Taichi kernels or functions.

```python
arr = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
```

Similar to `ti.field`, ndarrays are allocated on the arch as specified in ti.init and are initialized to 0 by default.

Apart from the constructor, Taichi provides some basic operations to interact with ndarray data from the Python scope.

- Fill in the ndarray with a scalar value

    ```python cont
    arr.fill(1.0)
    ```

- Read/write ndarray elements from the Python scope

    ```python cont
    # Returns a ti.Vector, which is a copy of the element
    print(arr[0, 0]) # [1.0, 1.0, 1.0]

    # Writes to an element
    arr[0, 0] = [1.0, 2.0, 3.0] # arr[0, 0] is now [1.0, 2.0, 3.0]

    # Writes to a scalar inside vector element
    arr[0, 0][1] = 2.2  # arr[0, 0] is now [1.0, 2.2, 3.0]
    ```

:::note
Accessing elements of an ndarray from the Python scope can be convenient, but it can also result in the creation and launch of multiple small Taichi kernels. This is not the most efficient approach from a performance standpoint. It is recommended that computationally intensive tasks be performed within a single Taichi kernel rather than operating on array elements individually from the Python scope.
:::

- Data copy of ndarrays

    Both shallow copy and deep copy from one ndarray to another are supported.

    ```python cont
    # Copies from another ndarray with the same size
    b = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    b.copy_from(arr)  # Copies all data from arr to b

    import copy
    # Deep copy
    c = copy.deepcopy(b)  # c is a new ndarray that has a copy of b's data.

    # Shallow copy
    d = copy.copy(b)  # d is a shallow copy of b; they share the underlying memory
    d[0, 0][0] = 1.2  # This mutates b as well, so b[0, 0][0] is now 1.2
    ```

- Bidirectional data exchange with NumPy ndarrays

    ```python cont
    # to_numpy returns a NumPy array with the same shape as d and a copy of d's value
    e = d.to_numpy()

    # from_numpy copies the data in the NumPy array e to the Taichi ndarray d
    e.fill(10.0)  # Fills in the NumPy array with value 10.0
    d.from_numpy(e)  # Now d is filled in with 10.0
    ```

## Using Taichi ndarrays in `ti.kernel`

To use an ndarray in a Taichi kernel, you need to properly annotate its type in the kernel definition and pass the `Ndarray` object to the Taichi kernel at runtime. `Ndarray`s are passed by reference. Therefore, you can mutate their content inside Taichi kernels. The following example shows how to specify the type annotation for an ndarray:

```python
@ti.kernel
def foo(A: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    do_something()
```

It is important to note that the `dtype` and `ndim` arguments are optional when an ndarray is instantiated. If left unspecified, the data type and the number of dimensions are inferred from the passed-in array at runtime. However, if the arguments are specified, Taichi validates that the specified data type and dimensions match those of the passed-in array. If a mismatch is detected, an error is thrown.

In certain scenarios, it may be necessary to process arrays with vector or matrix elements, such as an RGB pixel map (vec3). Taichi provides support for these types of arrays through the use of vector and matrix data types. An example of this would be creating an ndarray for a pixel map with vec3 elements, as demonstrated in the following code snippet:

```python
import taichi as ti

ti.init(arch=ti.cuda)

arr_ty = ti.types.ndarray(dtype=ti.math.vec3, ndim=2)

@ti.kernel
def proc(rgb_map : arr_ty):
    for I in ti.grouped(rgb_map):
        rgb_map[I] = [0.1, 0.2, 0.3]
    # do something

rgb = ti.ndarray(dtype=ti.types.vector(3, ti.f32), shape=(8,8))
proc(rgb)
```

It does not matter whether you use the [range-for](https://docs.taichi-lang.org/docs/language_reference#the-range-for-statement) or [struct-for](https://docs.taichi-lang.org/docs/language_reference#the-struct-for-statement) to iterate over the ndarrays.

:::tip

In the above code, we use `arr_ty` as a type alias for the 2D ndarray type of vec3 elements. The type alias makes type annotations shorter and easier to read.

:::

## Using external data containers in `ti.kernel`

The introduction of Taichi ndarrays achieves the seemless interop with external data containers. With the argument annotation `ti.types.ndarray`, a kernel accepts not only Taichi ndarrays but also external arrays. Currently, the external arrays supported by Taichi are NumPy ndarrays and PyTorch tensors.

The code snippet below defines a Taichi kernel that adds `1.0` to every element in the ndarray `arr`.

```python
ti.init(arch=ti.cuda)

@ti.kernel
def add_one(arr : ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + 1.0
```

The external arrays can be fed into the Taichi kernel without further type conversions.

To feed a NumPy ndarray:

```python cont
arr_np = np.ones((3, 3), dtype=np.float32)
add_one(arr_np) # arr_np is updated by taichi kernel
```

To feed a PyTorch tensor:

```python cont
arr_torch = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device='cuda:0')
add_one(arr_torch) # arr_torch is updated by taichi kernel
```

:::note

Every element in the external array (`arr_np` and `arr_torch`) is added by `1.0` when the Taichi kernel `add_one` finishes.

:::

When the external data container and Taichi are utilizing the same device, passing arguments incurs no additional overhead. This can be seen in the PyTorch example above, where the tensor is allocated on the CUDA device, which is also the device utilized by Taichi. As a result, the kernel function can access and manipulate the data in the original CUDA buffer allocated by PyTorch without incurring any extra costs.

On the other hand, if the devices being used are different, as is the case in the first example where Numpy utilizes CPUs and Taichi utilizes CUDA, Taichi automatically manages the transfer of data between devices, eliminating the need for manual intervention on the part of the user.

:::tip

NumPy's default data precision is 64-bit, which is still inefficient for most desktop GPUs. It is recommended to explicitly specify 32-bit data types.

:::

:::tip

Only contiguous NumPy arrays and PyTorch tensors are supported.

:::

```python known-error:NotWorkingAsExpected
# Transposing the tensor returns a view of the tensor, which is not contiguous
p = arr_torch.T
# add_one(p) # Error!
z = p.clone()
add_one(z) # Correct
k = p.contiguous()
addd_one(k) # Correct
```

When a NumPy ndarray or a PyTorch tensor of scalar type is passed as the argument to a Taichi kernel, it can be interpreted as an array of scalar type, or an array of vector type, or an array of matrix type. This is controlled by the `dtype` and `ndim` options in the type hint `ti.types.ndarray()`.

When the array is interpreted as a vector/matrix array, you should set `dtype` to the correct vector/matrix type. For example, you can safely pass a NumPy ndarray in shape `(2, 2, 3, 3)` as an argument into the `add_one` kernel, as shown below:

```python
@ti.kernel
def add_one(arr : ti.types.ndarray(dtype=ti.math.mat3, ndim=2)):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + 1.0
```

## Kernel compilation with ndarray template

In the examples above, `dtype` and `ndim` were specified explicitly in the kernel type hints, but Taichi also allows you to skip such details and just annotate the argument as `ti.types.ndarray()`. When one `ti.kernel` definition works with different (dtype, ndim) inputs, you do not need to duplicate the definition each time.

For example:

```python
@ti.kernel
def test(arr: ti.types.ndarray()):
    for I in ti.grouped(arr):
        arr[I] += 2
```

Consider `ti.types.ndarray()` as a template type on parameters `dtype` and `ndim`. Equipped with a just-in-time (JIT) compiler, Taichi goes through the following two steps when a kernel with templated ndarray parameters is invoked:

1. First, Taichi checks whether a kernel with the same `dtype` and `ndim` inputs has been compiled. If yes, it loads and launches the compiled kernel directly with the input arguments.
2. If your templated kernel needs to be compiled - either because it has never been compiled before or because it is invoked with an input of different `(dtype, ndim)`, kernel compilation is automatically triggered and executed. Note that the compiled kernel is also cached for future use.

The code snippet below provides more examples to demonstrate the behavior:

```python cont
a = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
b = ti.ndarray(dtype=ti.math.vec3, shape=(5, 5))
c = ti.ndarray(dtype=ti.f32, shape=(4, 4))
d = ti.ndarray(dtype=ti.f32, shape=(8, 6))
e = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4, 4))
test(a) # New kernel compilation
test(b) # Reuse kernel compiled for a
test(c) # New kernel compilation
test(d) # Reuse kernel compiled for c
test(e) # New kernel compilation
```

The compilation rule also applies to external arrays from NumPy or PyTorch. Changing the shape values does not trigger compilation, but changing the data type or the number of array dimensions does.


## FAQ

### How to use automatic differentiation with ndarrays?

We recommend referring to [this project](https://github.com/taichi-dev/taichi-nerfs/blob/main/notebooks/autodiff.ipynb).

Currently, the support for automatic differentiation in Taichi's ndarray is still incomplete. We are working on improving this functionality and will provide a more detailed tutorial as soon as possible.
