import taichi as ti

ti.init(arch=ti.vulkan)


@ti.aot.export
@ti.kernel
def arange(a: ti.types.ndarray(ti.math.mat2, ndim=1)):
    for i in a:
        a[i] = 1


x = ti.ndarray(ti.math.mat2, shape=(4))
arange(x)
