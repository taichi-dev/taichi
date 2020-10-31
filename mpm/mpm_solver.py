import taichi as ti
ti.init(arch=ti.cpu)

x = ti.field(dtype=ti.f32)
y = ti.field(dtype=ti.f32)

grid1 = ti.root.pointer(ti.i, 1)
grid2 = ti.root.dense(ti.i, 1)
grid1.pointer(ti.i, 1).place(x)
grid2.place(y)

ti.get_runtime().prog.print_snode_tree()

grid1.deactivate_all()
ti.sync()
print('successfully finishes')
