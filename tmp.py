import taichi as ti

ti.init(arch=ti.cuda)

N = 100000

dx = 1 / 128
inv_dx = 1.0 / dx

x = ti.Vector.field(2, dtype=ti.f32)

indices = ti.ij

grid_m = ti.field(dtype=ti.i32)

grid = ti.root.pointer(indices, 64)
grid.pointer(indices, 32).dense(indices, 8).place(grid_m)

ti.root.dense(ti.i, N).place(x)

assert grid.num_dynamically_allocated == 0
exit()

