---
sidebar_position: 2
---

# Synchronization between Kernels and Python Scope

When using the GPU backend, a kernel is compiled and sent to the GPU queue for execution. The Python program does not wait for the kernel to finish before continuing to execute the next statement. This can be problematic if subsequent computations depend on the result of the kernel. In most of the time, Taichi automatically handles such data dependencies, waiting for the kernel to finish before executing subsequent statements. However, in rare cases, users may need to manually call `ti.sync()` to ensure synchronization.

For example, to measure the execution time of a kernel on the GPU using Python's `time` module:

```python
import time
import taichi as ti
ti.init(arch=ti.gpu)

@ti.kernel
def benchmark():
    x = 1.0
    for i in range(1000):
        for j in range(10000):
            x += ti.sin(float(i + j))

start = time.time()
benchmark()
end = time.time()
print(end - start)
```

The above code defines a `benchmark` function that is a computationally intensive kernel. However, the program does not wait for the kernel to finish before executing subsequent statements, and the execution time of `benchmark` is excluded from the timing measurement.

To wait for the `benchmark` function to finish before executing subsequent statements, we can call `ti.sync()` explicitly. This function blocks the program and waits for all kernel tasks ahead in the GPU queue to finish before executing subsequent statements:


```python skip-ci
import time
import taichi as ti
ti.init(arch=ti.gpu)

@ti.kernel
def benchmark():
    x = 1.0
    for i in range(10000):
        for j in range(100000):
            x += ti.sin(float(i + j))

start = time.time()
benchmark()
ti.sync()
end = time.time()
print(end - start)
```

Most of the time, Taichi automatically handles data synchronization. For example, Taichi will automatically call `ti.sync()` to synchronize data in the following cases:

1. The kernel has a return value.
2. The `x.to_numpy()` method is called in the Python scope, and `x` is being used by other kernels. Taichi will wait for the kernel that is using `x` to finish before invoking the `x.to_numpy()` method.
3. An attempt is made to access or modify a Taichi field in the Python scope, and this field is being used by other kernels. Taichi will wait for the kernel that is using `x` to finish before proceeding with any statements that access or modify `x`.

In the example above, we need to explicitly call `ti.sync()` because its kernel does not have a return value, does not contain `print` statements, and does not involve any operations on fields.

:::note

On the CPU backend, all kernel calls are blocking, which means that the program waits for the current kernel to finish before executing the next statement. Therefore, users do not need to worry about data dependencies or use `ti.sync()` on the CPU backend.

:::
