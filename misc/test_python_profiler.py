import taichi as ti


@ti.profiler.timed('foo', warmup=5)
@ti.kernel
def func0():
    print(233)


@ti.profiler.timed(warmup=5)
@ti.kernel
def func1():
    print(233)


@ti.profiler.timed('Func2')
@ti.kernel
def func2():
    print(666)


@ti.profiler.timed
@ti.kernel
def func3():
    pass


for k in range(5):
    ti.profiler.start('step')
    for i in range(10):
        func0()
        func1()
        func2()
        func3()

    ti.profiler.stop('step')

ti.profiler.print()
