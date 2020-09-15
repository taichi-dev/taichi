import taichi as ti


def test_fusion_range():
    ti.init(arch=ti.cpu, async_mode=True)

    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    z = ti.field(ti.i32)

    n = 128

    block = ti.root.dense(ti.i, n)
    block.place(x, y, z)

    @ti.kernel
    def foo():
        for i in range(n):
            y[i] = x[i] + 1

    @ti.kernel
    def bar():
        for i in range(n):
            z[i] = y[i] + 1

    foo()
    bar()

    ti.core.print_sfg()
    dot = ti.dump_dot("fusion_range.dot")
    print(dot)
    ti.dot_to_pdf(dot, "fusion_range.pdf")

    ti.sync()


def test_fusion():
    # TODO: fix fusion here
    ti.init(arch=ti.cpu,
            async_mode=True,
            async_opt_intermediate_file="fusion")

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

    ti.core.print_sfg()
    dot = ti.dump_dot("war.dot")
    print(dot)
    ti.dot_to_pdf(dot, "war.pdf")

    ti.sync()


test_fusion()
# test_write_after_read()
