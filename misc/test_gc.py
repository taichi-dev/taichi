import taichi as ti
import random

ti.init()


res = (128, 128)
dx = 1 / 128
inv_dx = 1.0 / dx

x = ti.Vector(2, dt=ti.f32)

indices = ti.ij

grid_m = ti.var(dt=ti.f32)

grid = ti.root.pointer(indices, 32)
grid.pointer(indices, 8).dense(indices, 8).place(grid_m)

ti.root.dense(ti.i, 1000).place(x)
    
@ti.kernel
def build_pid():
    ti.block_dim(64)
    for p in x:
        base = int(ti.floor(x[p] * inv_dx - 0.5))
        grid_m[base] += 1


@ti.kernel
def move():
    for p in x:
        x[p] += ti.Vector([0.0, 0.01])

for i in range(1000):
    x[i] = [random.random() * 0.1 + 0.5, random.random() * 0.1 + 0.5]

while True:
    grid.deactivate_all()
    build_pid()
    move()
    ti.memory_profiler_print()
