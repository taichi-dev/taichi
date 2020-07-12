import taichi as ti


ti.init(arch=ti.cpu, advanced_optimization=False, print_ir=True)
x = ti.var(ti.i32, shape=4)


@ti.kernel
def func():
    c = 0
    for i in ti.static(range(4)):
        x[c] = 1
        c += 1


func()
#print(x.to_numpy())
