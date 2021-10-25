import pytest

import taichi as ti


@ti.test(arch=[ti.cpu, ti.cuda])
def test_pointer():
    e = ti.Vector.field(2, dtype=int, shape=16)

    e[0] = ti.Vector([0, 0])

    a = ti.field(float, shape=512)
    b = ti.field(dtype=float)
    ti.root.pointer(ti.i, 32).dense(ti.i, 16).place(b)

    @ti.kernel
    def test():
        for i in a:
            a[i] = i
        for i in a:
            b[i] += a[i]

    test()
    ti.sync()

    b_np = b.to_numpy()
    for i in range(512):
        assert (b_np[i] == i)
