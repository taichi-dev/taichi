import taichi as ti


@ti.all_archs
def test_rescale():
    a = ti.field(ti.f32)
    b = ti.field(ti.f32)
    ti.root.dense(ti.ij, 4).dense(ti.ij, 4).place(a)
    ti.root.dense(ti.ij, 4).place(b)

    @ti.kernel
    def set_b():
        for I in ti.grouped(a):
            Ib = ti.rescale_index(a, b, I)
            b[Ib] += 1.0

    @ti.kernel
    def set_a():
        for I in ti.grouped(b):
            Ia = ti.rescale_index(b, a, I)
            a[Ia] = 1.0

    set_a()
    set_b()

    for i in range(0, 4):
        for j in range(0, 4):
            assert b[i, j] == 16

    for i in range(0, 16):
        for j in range(0, 16):
            if i % 4 == 0 and j % 4 == 0:
                assert a[i, j] == 1
            else:
                assert a[i, j] == 0
