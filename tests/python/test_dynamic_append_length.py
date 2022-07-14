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

    test()


@test_utils.test(arch=ti.cpu, default_fp=ti.f32)
def test_dynamic_append_length_f32():
    _test_dynamic_append_length(ti.f32)


@test_utils.test(default_fp=ti.f64, require=ti.extension.data64)
def test_dynamic_append_length_f64():
    _test_dynamic_append_length(ti.f64)
