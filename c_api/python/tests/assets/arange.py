import taichi as ti

ti.init([ti.vulkan, ti.metal])

@ti.aot.export
@ti.kernel
def arange(
    x: ti.types.ndarray(ti.i32, ndim=1)
):
    for i in x:
        x[i] = i