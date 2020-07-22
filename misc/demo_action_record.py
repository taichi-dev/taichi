import taichi as ti

ti.init(ti.cc, log_level=ti.DEBUG)

@ti.kernel
def func():
    print(233)


with ti.ActionRecord('hello.c'):
    func()
