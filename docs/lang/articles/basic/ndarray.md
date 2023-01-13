---
sidebar_position: 3
---

# Taichi Ndarray

The Taichi ndarray is an array object that holds contiguous multi-dimensional data. Generally speaking, it plays a similar role to its counterpart `numpy.ndarray` in NumPy, but its underlying memory is allocated on the user-specified Taichi arch and managed by Taichi runtime.

## When to use ndarray

You can use fields as data containers in most cases. However, fields might have very complicated tree-structured layouts with even sparsity in them. It is hard for external libraries to interpret or use computed results stored in `ti.field` directly. An ndarray, however, always allocates a contiguous memory block to allow straightforward data exchange with external libraries.

In a nutshell, fields are mainly used for maximizing performance with complex data layouts. If you process dense data only or need external interop, just move forward with ndarrays!

## Python scope usages

The statement below constructs a Taichi ndarray. `dtype` specifies the data type, which can be either a scalar data type like `ti.f32/ti.i32` or a vector/matrix data type like `ti.math.vec2/mat2`. `shape` denotes the array size with respect to the data type. Like fields, ndarray can only be constructed in the Python scope rather than in the Taichi scope. That is to say, an ndarray cannot be constructed inside Taichi kernels or functions.

```python
arr = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
```

Similar to `ti.field`, ndarrays are allocated on the arch as specified in ti.init and are initialized to 0 by default.

Apart from the constructor, Taichi provides some basic operations to interact with ndarray data from the Python scope.

- Fill in the ndarray with a scalar value

    ```python
    arr.fill(1.0)
    ```

- Read/write ndarray elements from the Python scope

    ```python
    # Returns a ti.Vector, which is a copy of the element
    print(arr[0, 0]) # [1.0, 1.0, 1.0]

    # Writes to an element
    arr[0, 0] = [1.0, 2.0, 3.0] # arr[0, 0] is now [1.0, 2.0, 3.0]

    # Writes to a scalar inside vector element
    arr[0, 0][1] = 2.2  # arr[0, 0] is now [1.0, 2.2, 3.0]
    ```

:::Note

:::note
Accessing ndarrray elements from the Python scope comes in handy but inevitably generates and launches multiple tiny Taichi kernels, which is not the best practice performance-wise. You are encouraged to keep compute-heavy work inside one Taichi kernel instead of operating on arrays element-by-element from the Python scope.
:::

- Data copy of ndarrays

    Both shallow copy and deep copy from one ndarray to another are supported.

    ```python
    # Copies from another ndarray with the same size
    b = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    b.copy_from(arr)  # Copies all data from arr to b

    import copy
    # Deep copy
    c = copy.deepcopy(b)  # c is a new ndarray that has a copy of b's data.

    # Shallow copy
    d = copy.copy(b) # d is a shallow copy of b; they share the underlying memory
    c[0, 0][0] = 1.2 # This mutates b as well, so b[0, 0][0] is now 1.2
    ```

- Bidirectional data exchange with NumPy ndarrays

    ```python
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
def foo(A : ti.types.ndarray(dtype=ti.f32, ndim=2)):
    # Do something
```

We should note that both of the `dtype` and `ndim` arguments are optional. If unspecified, the element data type and the number of array dimensions are instantialized from the actually passed-in array at runtime. If either or both of the arguments are specified, Taichi checks the type and the dimensions and throws an error if a mismatch is found.

Sometimes, we need to process arrays with vector or matrix elements, such as an RGB pixel map (vec3). We can use vector/matrix types in such scenarios. Take the pixel map for example. The following code snippet defines an ndarray with vec3 elements:

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

:::TIPS

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

```python
arr_np = numpy.ones((3, 3), dtype=numpy.float32)
add_one(arr_np) # arr_np is updated by taichi kernel
```

To feed a PyTorch tensor:

```python
arr_torch = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device='cuda:0')
add_one(arr_torch) # arr_torch is updated by taichi kernel
```

:::Note

Every element in the external array (`arr_np` and `arr_torch`) is added by `1.0` when the Taichi kernel `add_one` finishes.

:::

If the external data container and Taichi use the same device, argument passing incurs zero overhead. The above PyTorch example allocates the tensor on the CUDA device, which is the same device used by Taichi. Therefore, the kernel function loads and stores data to the original CUDA buffer allocated by PyTorch, without any extra overhead.

However, if the devices are different, which is the case in the first example where Numpy uses CPUs and Taichi uses CUDA, Taichi automatically handles memory transfers across devices, saving users from manual operation.

:::TIPS

NumPy's default data precision is 64-bit, which is still inefficient for most desktop GPUs. It is recommended to explicitly specify 32-bit data types.

:::

:::TIPS

Only contiguous NumPy arrays and PyTorch tensors are supported.

:::

```python
# Transposing the tensor returns a view of the tensor, which is not contiguous
p = arr_torch.T
add_one(p) # Error!
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

In the examples above, `dtype` and `ndim` are specified explicitly in the kernel type hints, but Taichi also allows you to skip such details and just annotate the argument as `ti.types.ndarray()`. When one `ti.kernel` definition works with different (dtype, ndim) inputs, you do not need to duplicate the definition each time.

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

```python
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
