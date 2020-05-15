import taichi as ti

ti.init(print_ir=True)

x = ti.var(ti.i32)
ti.root.dense(ti.i, 4).bitmasked(ti.i, 4).place(x)

@ti.kernel
def func():
    for i in x:
        print(i)

func()
