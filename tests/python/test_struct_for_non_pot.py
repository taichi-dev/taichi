import taichi as ti


def _test_1d():
    x = ti.field(ti.i32)
    sum = ti.field(ti.i32)

    n = 100

    ti.root.dense(ti.k, n).place(x)
    ti.root.place(sum)

    @ti.kernel
    def accumulate():
        for i in x:
            ti.atomic_add(sum[None], i)

    accumulate()

    for i in range(n):
        assert sum[None] == 4950


@ti.test()
def test_1d():
    _test_1d()


@ti.test(require=ti.extension.packed, packed=True)
def test_1d_packed():
    _test_1d()


def _test_2d():
    x = ti.field(ti.i32)
    sum = ti.field(ti.i32)

    n = 100
    m = 19

    ti.root.dense(ti.k, n).dense(ti.i, m).place(x)
    ti.root.place(sum)

    @ti.kernel
    def accumulate():
        for i, j in x:
            ti.atomic_add(sum[None], i + j * 2)

    gt = 0
    for i in range(n):
        for j in range(m):
            gt += i + j * 2

    accumulate()

    for i in range(n):
        assert sum[None] == gt


@ti.test()
def test_2d():
    _test_2d()


@ti.test(require=ti.extension.packed, packed=True)
def test_2d_packed():
    _test_2d()
