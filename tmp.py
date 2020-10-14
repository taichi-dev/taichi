import taichi as ti

ti.init(print_ir=True, print_accessor_ir=True)


A = ti.field(ti.f32, shape=())

@ti.kernel
def func():
    a = 0
    for i in range(10):
        a -= i
    A[None] = a

func()
assert A[None] == -45
