import taichi as ti
from tests import test_utils


def _test_dynamic_append_length(dt):
    x = ti.field(int)
    block = ti.root.dense(ti.i, 10)
    pixel = block.dynamic(ti.j, 10)
    pixel.place(x)

    y = ti.field(int)
    ti.root.dynamic(ti.i, 10).place(y)

    @ti.kernel
    def test():
        for i in range(10):
            for j in range(i):
                x[i].append(j)

        for i in range(10):
            assert (ti.length(x.parent(), i) == i)
            for j in range(i):
                assert (x[i, j] == j)

        for i in range(10):
            x[i].deactivate()
            for j in range(10):
                assert x[i, j] == 0

        for j in range(10):
            y[j] = j

        y.deactivate()
        for j in range(10):
            assert y[j] == 0

    test()


@test_utils.test(exclude=[ti.cc, ti.opengl, ti.vulkan, ti.metal],
                 default_fp=ti.f32,
                 debug=True)
def test_dynamic_append_length_f32():
    _test_dynamic_append_length(ti.f32)
