import taichi as ti
from tests import test_utils


def _test_dynamic_append_length(dt):
    x = ti.field(int)
    block = ti.root.dense(ti.i, 10)
    pixel = block.dynamic(ti.j, 10)
    pixel.place(x)

    @ti.kernel
    def test():
        for i in range(10):
            for j in range(i):
                x[i].append(j)
        for i in range(10):
            assert x[i].length() == i
            for j in range(i):
                assert x[i, j] == j

    test()


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal], default_fp=ti.f32, debug=True)
def test_dynamic_append_length_f32():
    _test_dynamic_append_length(ti.f32)
