import taichi as ti


@ti.all_archs
def test_1d():
    x = ti.field(ti.i32)
    sum = ti.field(ti.i32)

    n = 100

    ti.root.dense(ti.k, n).place(x)
    ti.root.place(sum)

    @ti.kernel
    def accumulate():
        for i in x:
            ti.atomic_add(sum, i)

    accumulate()

    for i in range(n):
        assert sum[None] == 4950


@ti.all_archs
def test_2d():
    x = ti.field(ti.i32)
    sum = ti.field(ti.i32)

    n = 100
    m = 19

    ti.root.dense(ti.k, n).dense(ti.i, m).place(x)
    ti.root.place(sum)

    @ti.kernel
    def accumulate():
        for i, j in x:
            ti.atomic_add(sum, i + j * 2)

    gt = 0
    for i in range(n):
        for j in range(m):
            gt += i + j * 2

    accumulate()

    for i in range(n):
        assert sum[None] == gt
