import taichi as ti
ti.init(arch=ti.cpu, debug=True)

x = ti.field(dtype=ti.f32)
y = ti.field(dtype=ti.f32)

grid_size = 4

grid1 = ti.root.pointer(ti.i, grid_size)
grid2 = ti.root.dense(ti.i, grid_size)
grid1.pointer(ti.i, grid_size).place(x)
grid2.place(y)

ti.get_runtime().prog.print_snode_tree()

grid1.deactivate_all()
ti.sync()
print('successfully finishes')
