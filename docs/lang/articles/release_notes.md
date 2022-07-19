---
sidebar_position: 15
---

# Release Notes

## V1.0.0

v1.0.0 was released on April 13, 2022.

### Compatibility changes

---

#### License change

Taichi's license is changed from MIT to Apache-2.0 after a public vote in [#4607](https://github.com/taichi-dev/taichi/issues/4607).

#### Python 3.10 support

This release supports Python 3.10 on all supported operating systems (Windows, macOS, and Linux).

#### Manylinux2014-compatible wheels

Before v1.0.0, Taichi works only on Linux distributions that support glibc 2.27+ (such as Ubuntu 18.04+). As of v1.0.0, in addition to the normal Taichi wheels, Taichi provides the manylinux2014-compatible wheels to work on most modern Linux distributions, including CentOS 7.

- The normal wheels support all backends; the incoming manylinux2014-compatible wheels support the CPU and CUDA backends only. Choose the wheels that work best for you.

- If you encounter any issue when installing the wheels, try upgrading your pip to the latest version first.

### Deprecations

- This release deprecates `ti.ext_arr()` and uses `ti.types.ndarray()` instead. `ti.types.ndarray()` supports both Taichi Ndarrays and external arrays, such as NumPy arrays.

- Taichi plans to drop support for Python 3.6 in the next minor release (v1.1.0). If you have any questions or concerns, please let us know at [#4772](https://github.com/taichi-dev/taichi/discussions/4772).

### New features

---

#### Non-Python deployment solution

By working together with OPPO US Research Center, Taichi delivers Taichi AOT, a solution for deploying kernels in non-Python environments, such as in mobile devices.

Compiled Taichi kernels can be saved from a Python process, then loaded and run by the [provided C++ runtime library](https://github.com/taichi-dev/taichi/releases/download/v1.0.0/libtaichi_export_core.so). With a set of APIs, your Python/Taichi code can be easily deployed in any C++ environment. We demonstrate the simplicity of this workflow by porting the [implicit FEM (finite element method) demo](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/implicit_fem.py) released in v0.9.0 to an Android application. Download the [Android package](https://github.com/taichi-dev/taichi/releases/download/v1.0.0/TaichiAOT.apk) and find out what Taichi AOT has to offer! If you want to try out this solution, please also check out [the taichi-aot-demo repo](https://github.com/taichi-dev/taichi-aot-demo).

![AOT demo](https://github.com/taichi-dev/taichi/releases/download/v1.0.0/taichi-aot-demo.gif)

```python
# In Python app.py
module = ti.aot.Module(ti.vulkan)
module.add_kernel(my_kernel, template_args={'x': x})
module.save('my_app')
```

The following code snippet shows the C++ workflow for loading the compiled AOT modules.

```C++
// Initialize Vulkan program pipeline
taichi::lang::vulkan::VulkanDeviceCreator::Params evd_params;
evd_params.api_version = VK_API_VERSION_1_2;
auto embedded_device =
    std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
std::vector<uint64_t> host_result_buffer;
host_result_buffer.resize(taichi_result_buffer_entries);
taichi::lang::vulkan::VkRuntime::Params params;
params.host_result_buffer = host_result_buffer.data();
params.device = embedded_device->device();
auto vulkan_runtime = std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));
// Load AOT module saved from Python
taichi::lang::vulkan::AotModuleParams aot_params{"my_app", vulkan_runtime.get()};
auto module = taichi::lang::aot::Module::load(taichi::Arch::vulkan, aot_params);
auto my_kernel = module->get_kernel("my_kernel");
// Allocate device buffer
taichi::lang::Device::AllocParams alloc_params;
alloc_params.host_write = true;
alloc_params.size = /*Ndarray size for `x`*/;
alloc_params.usage = taichi::lang::AllocUsage::Storage;
auto devalloc_x = embedded_device->device()->allocate_memory(alloc_params);
// Execute my_kernel without Python environment
taichi::lang::RuntimeContext host_ctx;
host_ctx.set_arg_devalloc(/*arg_id=*/0, devalloc_x, /*shape=*/{128}, /*element_shape=*/{3, 1});
my_kernel->launch(&host_ctx);
```

Note that Taichi only supports the Vulkan backend in the C++ runtime library. The Taichi team is working on supporting more backends.

#### Real functions (experimental)

All Taichi functions are inlined into the Taichi kernel during compile time. However, the kernel becomes lengthy and requires longer compile time if it has too many Taichi function calls. This becomes especially obvious if a Taichi function involves [compile-time recursion](https://docs.taichi-lang.org/docs/master/meta#compile-time-recursion-of-tifunc). For example, the following code calculates the Fibonacci numbers recursively:

```python
@ti.func
def fib_impl(n: ti.template()):
    if ti.static(n <= 0):
        return 0
    if ti.static(n == 1):
        return 1
    return fib_impl(n - 1) + fib_impl(n - 2)
@ti.kernel
def fibonacci(n: ti.template()):
    print(fib_impl(n))
```

In this code, `fib_impl()` recursively calls itself until `n` reaches `1` or `0`. The total time of the calls to `fib_impl()` increases exponentially as `n` grows, so the length of the kernel also increases exponentially. When `n` reaches `25`, it takes more than a minute to compile the kernel.

This release introduces "real function", a new type of Taichi function that compiles independently instead of being inlined into the kernel. It is an experimental feature and only supports scalar arguments and scalar return value for now.

You can use it by decorating the function with `@ti.experimental.real_func`. For example, the following is the real function version of the code above.

```python
@ti.experimental.real_func
def fib_impl(n: ti.i32) -> ti.i32:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fib_impl(n - 1) + fib_impl(n - 2)
@ti.kernel
def fibonacci(n: ti.i32):
    print(fib_impl(n))
```

The length of the kernel does not increase as `n` grows because the kernel only makes a call to the function instead of inlining the whole function. As a result, the code takes far less than a second to compile regardless of the value of `n`.

The main differences between a normal Taichi function and a real function are listed below:

- You can write return statements in any part of a real function, while you cannot write return statements inside the scope of non-static `if` / `for` / `while` statements in a normal Taichi function.

- A real function can be called recursively at runtime, while a normal Taichi function only supports compile-time recursion.

- The return value and arguments of a real function must be type hinted, while the type hints are optional in a normal Taichi function.

#### Type annotations for literals

Previously, you cannot explicitly give a type to a literal. For example,

```python
@ti.kernel
def foo():
    a = 2891336453  # i32 overflow (>2^31-1)
```

In the code snippet above, `2891336453` is first turned into a default integer type (`ti.i32` if not changed). This causes an overflow. Starting from v1.0.0, you can write type annotations for literals:

```python
@ti.kernel
def foo():
    a = ti.u32(2891336453)  # similar to 2891336453u in C
```

### Top-level loop configurations

You can use `ti.loop_config` to control the behavior of the subsequent top-level for-loop. Available parameters are:

- `block_dim`: Sets the number of threads in a block on GPU.

- `parallelize`: Sets the number of threads to use on CPU.

- `serialize`: If you set `serialize` to `True`, the for-loop runs serially, and you can write break statements inside it (Only applies to range/ndrange for-loops). Setting `serialize` to `True` equals setting `parallelize` to `1`.

Here are two examples:

```python
@ti.kernel
def break_in_serial_for() -> ti.i32:
    a = 0
    ti.loop_config(serialize=True)
    for i in range(100):  # This loop runs serially
        a += i
        if i == 10:
            break
    return a
break_in_serial_for()  # returns 55
```

```python
n = 128
val = ti.field(ti.i32, shape=n)
@ti.kernel
def fill():
    ti.loop_config(parallelize=8, block_dim=16)
    # If the kernel is run on the CPU backend, 8 threads will be used to run it
    # If the kernel is run on the CUDA backend, each block will have 16 threads
    for i in range(n):
        val[i] = i
```

#### `math` module

This release adds a `math` module to support GLSL-standard vector operations and to make it easier to port GLSL shader code to Taichi. For example, vector types, including `vec2`, `vec3`, `vec4`, `mat2`, `mat3`, and `mat4`, and functions, including `mix()`, `clamp()`, and `smoothstep()`, act similarly to their counterparts in GLSL. See the following examples:

##### Vector initialization and swizzling

You can use the `rgba`, `xyzw`, `uvw` properties to get and set vector entries:

```python
import taichi.math as tm
@ti.kernel
def example():
    v = tm.vec3(1.0)  # (1.0, 1.0, 1.0)
    w = tm.vec4(0.0, 1.0, 2.0, 3.0)
    v.rgg += 1.0  # v = (2.0, 3.0, 1.0)
    w.zxy += tm.sin(v)
```

##### Matrix multiplication

Each Taichi vector is implemented as a column vector. Ensure that you put the the matrix before the vector in a matrix multiplication.

```python
@ti.kernel
def example():
    M = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v = tm.vec3(1, 2, 3)
    w = (M @ v).xyz  # [1, 2, 3]
```

##### GLSL-standard functions

```python
@ti.kernel
def example():
    v = tm.vec3(0., 1., 2.)
    w = tm.smoothstep(0.0, 1.0, v.xyz)
    w = tm.clamp(w, 0.2, 0.8)
```

#### CLI command `ti gallery`

This release introduces a CLI command `ti gallery`, allowing you to select and run Taichi examples in a pop-up window. To do so:

1. Open a terminal:

```python
ti gallery
```

*A window pops up:*

![ti.gallery](https://github.com/taichi-dev/taichi/releases/download/v1.0.0/taichi-gallery.jpg)

2. Click to run any example in the pop-up window.

    *The console prints the corresponding source code at the same time.*

### Improvements

---

#### Enhanced matrix type

As of v1.0.0, Taichi accepts matrix or vector types as parameters and return values. You can use `ti.types.matrix` or `ti.types.vector` as the type annotations.

Taichi also supports basic, read-only matrix slicing. Use the `mat[:,:]` syntax to quickly retrieve a specific portion of a matrix. See [Slicings](https://docs.taichi-lang.org/docs/master/language_reference#slicings) for more information.

The following code example shows how to get numbers in four corners of a `3x3` matrix `mat`:

```python
import taichi as ti
ti.init()
@ti.kernel
def foo(mat: ti.types.matrix(3, 3, ti.i32)) -> ti.types.matrix(2, 2, ti.i32)
    corners = mat[::2, ::2]
    return corners
mat = ti.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
corners = foo(mat)  # [[1 3] [7 9]]
```

Note that in a slice, the lower bound, the upper bound, and the stride must be constant integers. If you want to use a variable index together with a slice, you should set `ti.init(dynamic_index=True)`. For example:

```python
import taichi as ti
ti.init(dynamic_index=True)
@ti.kernel
def foo(mat: ti.types.matrix(3, 3, ti.i32), ind: ti.i32) -> ti.types.matrix(3, 1, ti.i32):
    col = mat[:, ind]
    return col
mat = ti.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col = foo(mat, 2)  # [3 6 9]
```

#### More flexible Autodiff: Kernel Simplicity Rule removed

Flexiblity is key to the user experience of an automatic differentiation (AD) system. Before v1.0.0, Taichi AD system requires that a differentiable Taichi kernel only consist multiple simply nested for-loops (shown in `task1` below). This was once called the Kernel Simplicity Rule (KSR). KSR prevents Taichi's users from writing differentiable kernels with multiple serial for-loops (shown in `task2` below) or with a mixture of serial for-loop and non-for statements (shown in `task3` below).

```python
# OK: multiple simply nested for-loops
@ti.kernel
def task1():
    for i in range(2):
        for j in range(3):
            for k in range(3):
                y[None] += x[None]
# Error: multiple serial for-loops
@ti.kernel
def task2():
    for i in range(2):
        for j in range(3):
            y[None] += x[None]
        for j in range(3):
            y[None] += x[None]
# Error: a mixture of serial for-loop and non-for
@ti.kernel
def task3():
    for i in range(2):
        y[None] += x[None]
        for j in range(3):
            y[None] += x[None]
```

With KSR being removed from this release, code with different kinds of for-loops structures can be differentiated, as shown in the snippet below.

```python
# OK: A complicated control flow that is still differentiable in Taichi
for j in range(2):
    for i in range(3):
        y[None] += x[None]
    for i in range(3):
        for ii in range(2):
            y[None] += x[None]
        for iii in range(2):
            y[None] += x[None]
            for iv in range(2):
                y[None] += x[None]
    for i in range(3):
        for ii in range(2):
            for iii in range(2):
                y[None] += x[None]
```

Taichi provides a [demo](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/autodiff/diff_sph/diff_sph.py) to demonstrate how to implement a differentiable simulator using this enhanced Taichi AD system.

![magic fountain](https://github.com/taichi-dev/taichi/releases/download/v1.0.0/diff-sph-demo.gif)

#### f-string support in an `assert` statement

This release supports including an f-string in an `assert` statement as an error message. You can include scalar variables in the f-string. See the example below:

```python
import taichi as ti
ti.init(debug=True)
@ti.kernel
def assert_is_zero(n: ti.i32):
    assert n == 0, f"The number is {n}, not zero"
assert_is_zero(42)  # TaichiAssertionError: The number is 42, not zero
```

Note that the `assert` statement works only in debug mode.

### Documentation changes

---

#### Taichi language reference

This release comes with the [first version of the Taichi language specification](https://docs.taichi-lang.org/docs/master/language_reference), which attempts to provide an exhaustive description of the syntax and semantics of the Taichi language and makes a decent reference for Taichi's users and developers when they determine if a specific behavior is correct, buggy, or undefined.

### API changes

---

#### Deprecated

| Deprecated    | Replaced by         |
| ----------    | -----------         |
| `ti.ext_arr()`| `ti.types.ndarray()`|

***For full changelog, refer to <https://github.com/taichi-dev/taichi/releases>***
