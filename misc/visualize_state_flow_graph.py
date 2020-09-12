import taichi as ti

ti.init(arch=ti.cpu, async_mode=True)

x = ti.field(ti.i32)
y = ti.field(ti.i32)
z = ti.field(ti.i32)

num_dense_layers = 1

block = ti.root.pointer(ti.i, 128)
for i in range(num_dense_layers):
    block = block.dense(ti.i, 2)
block.place(x, y, z)


@ti.kernel
def foo():
    for i in x:
        y[i] = x[i] + 1


@ti.kernel
def bar():
    for i in y:
        z[i] = y[i] + 1


foo()
bar()

ti.core.print_sfg()
dot = ti.dump_dot("sfg.dot")
print(dot)
ti.dot_to_pdf(dot, "sfg.pdf")
