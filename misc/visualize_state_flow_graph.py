import taichi as ti


def test_fusion():
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

    ti.sync()

    ti.core.print_sfg()
    dot = ti.dump_dot("fusion.dot")
    print(dot)
    ti.dot_to_pdf(dot, "fusion.pdf")


def test_write_after_read():
    ti.init(arch=ti.cpu, async_mode=True)

    x = ti.field(ti.i32, shape=16)

    @ti.kernel
    def p():
        print(x[ti.random(ti.i32) % 16])

    @ti.kernel
    def s():
        x[ti.random(ti.i32) % 16] = 3

    p()
    p()
    p()
    s()
    p()
    p()
    s()
    s()
    s()

    ti.sync()

    ti.core.print_sfg()
    dot = ti.dump_dot("war.dot")
    print(dot)
    ti.dot_to_pdf(dot, "war.pdf")


test_fusion()
test_write_after_read()
