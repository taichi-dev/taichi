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


@test_utils.test(arch=get_host_arch_list())
def test_class_with_field():
    @ti.data_oriented
    class B(object):
        def __init__(self):
            self.x = ti.field(int)
            fb = ti.FieldsBuilder()
            fb.dense(ti.i, 1).place(self.x)
            self.snode_tree = fb.finalize()

        def clear(self):
            self.snode_tree.destroy()

    @ti.data_oriented
    class A(object):
        def __init__(self):
            self.n = 12345

        def init(self):
            self.b = B()
            self.x = ti.field(int)
            fb = ti.FieldsBuilder()
            fb.dense(ti.i, self.n).place(self.x)
            self.snode_tree = fb.finalize()

        def clear(self):
            self.snode_tree.destroy()
            self.b.clear()
            del self.b

        @ti.kernel
        def k(self, m: int):
            for i in range(self.n):
                self.x[i] = m * i
                self.b.x[0] += m

        def start(self):
            self.init()
            self.k(1)
            assert self.x[34] == 34
            assert self.b.x[0] == 12345
            self.clear()
            del self.x

            self.init()
            self.k(2)
            assert self.x[34] == 68
            assert self.b.x[0] == 24690
            self.clear()
            del self.x

    a = A()
    a.start()
