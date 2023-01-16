import taichi as ti
from tests import test_utils


@test_utils.test()
def test_1d():
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


@test_utils.test()
def test_2d():
    x = ti.field(ti.i32)
    sum = ti.field(ti.i32)

    n = 100
    m = 19

    ti.root.dense(ti.k, n).dense(ti.i, m).place(x)
    ti.root.place(sum)

    @ti.kernel
    def accumulate():
        for i, k in x:
            ti.atomic_add(sum[None], i + k * 2)

    gt = 0
    for k in range(n):
        for i in range(m):
            gt += i + k * 2

    accumulate()

    for i in range(n):
        assert sum[None] == gt


@test_utils.test(require=ti.extension.sparse)
def test_2d_pointer():
    block_size, leaf_size = 3, 8
    x = ti.field(ti.i32)
    block = ti.root.pointer(ti.ij, (block_size, block_size))
    block.dense(ti.ij, (leaf_size, leaf_size)).place(x)

    @ti.kernel
    def activate():
        x[7, 7] = 1

    activate()

    @ti.kernel
    def test() -> ti.i32:
        res = 0
        for I in ti.grouped(x):
            res += I[0] + I[1] * 2
        return res

    ans = 0
    for i in range(leaf_size):
        for j in range(leaf_size):
            ans += i + j * 2

    assert ans == test()
