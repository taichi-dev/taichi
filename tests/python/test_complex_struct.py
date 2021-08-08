import taichi as ti


@ti.all_archs
def test_complex_dense():
    a = ti.field(ti.i32, shape=(4, 4))
    b = ti.field(ti.i32, shape=(16, 16))
    c = ti.field(ti.i32, shape=(16, 4))
    d = ti.field(ti.i32, shape=(4, 4, 4))

    w = ti.field(ti.i32)
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    z = ti.field(ti.i32)

    blk = ti.root.dense(ti.ij, 4)
    blk.place(w)
    blk.dense(ti.ij, 2).dense(ti.ij, 2).place(x)
    blk.dense(ti.i, 4).place(y)
    blk.dense(ti.k, 4).place(z)

    @ti.kernel
    def set_w():
        for I in ti.grouped(ti.ndrange(4, 4)):
            w[I] = 1

    @ti.kernel
    def set_x():
        for I in ti.grouped(ti.ndrange(16, 16)):
            x[I] = 2

    @ti.kernel
    def set_y():
        for I in ti.grouped(ti.ndrange(16, 4)):
            y[I] = 3

    @ti.kernel
    def set_z():
        for I in ti.grouped(ti.ndrange(4, 4, 4)):
            z[I] = 4

    @ti.kernel
    def set_a():
        for I in ti.grouped(w):
            a[I] = w[I]

    @ti.kernel
    def set_b():
        for I in ti.grouped(x):
            b[I] = x[I]

    @ti.kernel
    def set_c():
        for I in ti.grouped(y):
            c[I] = y[I]

    @ti.kernel
    def set_d():
        for I in ti.grouped(z):
            d[I] = z[I]

    set_w()
    set_x()
    set_y()
    set_z()

    set_a()
    set_b()
    set_c()
    set_d()

    for i in range(4):
        for j in range(4):
            assert a[i, j] == 1

    for i in range(16):
        for j in range(16):
            assert b[i, j] == 2

    for i in range(16):
        for j in range(4):
            assert c[i, j] == 3

    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert d[i, j, k] == 4


@ti.test(require=ti.extension.sparse)
def test_complex_pointer():
    a = ti.field(ti.i32, shape=(4, 4))
    b = ti.field(ti.i32, shape=(16, 16))
    c = ti.field(ti.i32, shape=(16, 4))
    d = ti.field(ti.i32, shape=(4, 4, 4))

    w = ti.field(ti.i32)
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    z = ti.field(ti.i32)

    blk = ti.root.pointer(ti.ij, 4)
    blk.place(w)
    blk.pointer(ti.ij, 2).dense(ti.ij, 2).place(x)
    blk.dense(ti.i, 4).place(y)
    blk.dense(ti.k, 4).place(z)

    @ti.kernel
    def set_w():
        for I in ti.grouped(ti.ndrange(4, 4)):
            w[I] = 1

    @ti.kernel
    def set_x():
        for I in ti.grouped(ti.ndrange(16, 16)):
            x[I] = 2

    @ti.kernel
    def set_y():
        for I in ti.grouped(ti.ndrange(16, 4)):
            y[I] = 3

    @ti.kernel
    def set_z():
        for I in ti.grouped(ti.ndrange(4, 4, 4)):
            z[I] = 4

    @ti.kernel
    def set_a():
        for I in ti.grouped(w):
            a[I] = w[I]

    @ti.kernel
    def set_b():
        for I in ti.grouped(x):
            b[I] = x[I]

    @ti.kernel
    def set_c():
        for I in ti.grouped(y):
            c[I] = y[I]

    @ti.kernel
    def set_d():
        for I in ti.grouped(z):
            d[I] = z[I]

    set_w()
    set_x()
    set_y()
    set_z()

    set_a()
    set_b()
    set_c()
    set_d()

    for i in range(4):
        for j in range(4):
            assert a[i, j] == 1

    for i in range(16):
        for j in range(16):
            assert b[i, j] == 2

    for i in range(16):
        for j in range(4):
            assert c[i, j] == 3

    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert d[i, j, k] == 4
