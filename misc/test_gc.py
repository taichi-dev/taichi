import taichi as ti
import random

ti.init(arch=ti.cuda, debug=True)

N = 1000000

res = (128, 128)
dx = 1 / 128
inv_dx = 1.0 / dx

x = ti.Vector(2, dt=ti.f32)

indices = ti.ij

grid_m = ti.var(dt=ti.f32)

grid = ti.root.pointer(indices, 64)
grid.pointer(indices, 32).dense(indices, 8).place(grid_m)

ti.root.dense(ti.i, N).place(x)
    
@ti.kernel
def build_pid():
    for p in x:
        base = int(ti.floor(x[p] * inv_dx - 0.5))
        grid_m[base] += 1


@ti.kernel
def move():
    for p in x:
        x[p] += ti.Vector([0.0, 0.1])

@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random() * 0.1 + 0.5, ti.random() * 0.1 + 0.5]
    
init()

for i in range(100):
    grid.deactivate_all()
    build_pid()
    move()
    ti.memory_profiler_print()
