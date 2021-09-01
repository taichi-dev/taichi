import taichi as ti


def ti_support_dynamic(test):
    return ti.archs_excluding(ti.opengl, ti.cc, ti.vulkan)(test)


@ti.test(exclude=[ti.opengl, ti.cc, ti.vulkan])
def test_dynamic():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32, shape=())

    n = 128

    ti.root.dynamic(ti.i, n).place(x)

    @ti.kernel
    def count():
        for i in x:
            y[None] += 1

    x[n // 3] = 1

    count()

    assert y[None] == n // 3 + 1


@ti.test(exclude=[ti.opengl, ti.cc, ti.vulkan])
def test_dense_dynamic():
    n = 128

    x = ti.field(ti.i32)

    ti.root.dense(ti.i, n).dynamic(ti.j, n, 128).place(x)

    @ti.kernel
    def append():
        for i in range(n):
            for j in range(i):
                ti.append(x.parent(), i, j * 2)

    append()

    for i in range(n):
        for j in range(i):
            assert x[i, j] == j * 2
