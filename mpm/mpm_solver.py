import taichi as ti
ti.init(arch=ti.cpu, print_ir=True)

x = ti.field(dtype=ti.i32)
y = ti.field(dtype=ti.i32)

grid1 = ti.root.dense(ti.i, 1)
grid2 = ti.root.dense(ti.i, 1)
ptr = grid1.pointer(ti.i, 1)
ptr.place(x)
grid2.place(y)

ti.get_runtime().prog.print_snode_tree()

@ti.kernel
def foo():
    ti.activate(ptr, [0])

foo()
ti.sync()
print('successfully finishes')
