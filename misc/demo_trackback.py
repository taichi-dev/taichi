import taichi as ti

ti.init()


@ti.func
def func3():
    ti.static_assert(1 + 1 == 3)


@ti.func
def func2():
    func3()


@ti.func
def func1():
    func2()


@ti.kernel
def func0():
    func1()


func0()
