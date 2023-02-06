import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.sparse,
                 exclude=[ti.opengl, ti.gles, ti.cc, ti.vulkan, ti.metal])
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


@test_utils.test(require=ti.extension.sparse,
                 exclude=[ti.opengl, ti.gles, ti.cc, ti.vulkan, ti.metal])
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
