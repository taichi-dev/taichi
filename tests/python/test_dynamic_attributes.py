import taichi as ti
from tests import test_utils


def _test_dynamic_append_length(dt):
    n = 10
    x = ti.field(int)
    block = ti.root.dense(ti.i, n)
    pixel = block.dynamic(ti.j, n)
    pixel.place(x)

    y = ti.field(int)
    ti.root.dynamic(ti.i, n).place(y)

    @ti.kernel
    def test():
        for i in range(n):
            for j in range(i):
                x[i].append(j)

        for i in range(n):
            assert (ti.length(x.parent(), i) == i)
            for j in range(i):
                assert (x[i, j] == j)

        for i in range(n):
            x[i].deactivate()
            for j in range(n):
                assert x[i, j] == 0

        for j in range(n):
            y[j] = j

        y.deactivate()
        for j in range(n):
            assert y[j] == 0

    test()


@test_utils.test(exclude=[ti.cc, ti.opengl, ti.vulkan, ti.metal],
                 default_fp=ti.f32,
                 debug=True)
def test_dynamic_append_length_f32():
    _test_dynamic_append_length(ti.f32)
