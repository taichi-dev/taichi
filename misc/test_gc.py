import taichi as ti
import random

ti.init(arch=ti.cpu, print_ir=True)


res = (128, 128)
dx = 1 / 128
inv_dx = 1.0 / dx

indices = ti.ij

m = ti.var(dt=ti.f32)

grid = ti.root.pointer(indices, 32)
grid.pointer(indices, 32).dense(indices, 8).place(m)

@ti.kernel
def build_pid(i: ti.i32):
    ti.parallelize(2)
    for j in range(100):
        m[j, j] += 1

for i in range(100):
    grid.deactivate_all()
    build_pid(0)
    ti.memory_profiler_print()
