import taichi as ti

ti.init(arch=ti.cpu, debug=True)

m = ti.field(dtype=ti.f32)

grid_size = 4
grid1 = ti.root.pointer(ti.i, grid_size)

use2 = True

if use2:
    grid2 = ti.root.dense(ti.i, grid_size)

grid1.pointer(ti.i, grid_size).place(m)

if use2:
    m2 = ti.field(dtype=ti.f32)
    grid2.place(m2)

ti.get_runtime().prog.print_snode_tree()

grid1.deactivate_all()
ti.sync()
print('successfully finishes')
