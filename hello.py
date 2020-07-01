import taichi as ti

ti.init(arch=ti.cc)

@ti.kernel
def func():
    print('Hello world!')
    print('Nice to meet you!', 233)

func()
