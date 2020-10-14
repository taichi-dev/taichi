import taichi as ti
import numpy as np

ti.init(arch=ti.opengl, print_ir=True, print_accessor_ir=True)

n = 10000


@ti.kernel
def inc(a: ti.ext_arr()):
    for i in ti.ndrange([0, n]):
        a[i] += i


x = np.zeros(dtype=np.int32, shape=n)
for i in range(10):
    inc(x)

for i in range(n):
    assert x[i] == i * 10
