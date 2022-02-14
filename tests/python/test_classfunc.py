from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_classfunc():
    @ti.data_oriented
    class Foo:
        def __init__(self):
            self.val = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=3)

        @ti.func
        def add_mat(self, a, b):
            return a + b

        @ti.kernel
        def fill(self):
            self.val[0] = self.add_mat(self.val[1], self.val[2])

    foo = Foo()
    foo.fill()
