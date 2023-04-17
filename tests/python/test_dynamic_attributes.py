import taichi as ti
from tests import test_utils


def _test_dynamic_attributes(dt):
    n = 10
    x = ti.field(int)
    block = ti.root.dense(ti.i, n)
    pixel = block.dynamic(ti.j, n)
    pixel.place(x)

    y = ti.field(int)
    row = ti.root.dynamic(ti.i, n)
    row.place(y)

    @ti.kernel
    def test():
        # test append for depth 2 snode
        for i in range(n):
            for j in range(i):
                x[i].append(j)

        for i in range(n):
            assert x[i].length() == i
            for j in range(i):
                assert x[i, j] == j

        # test deactivate for depth 2 snode
        for i in range(n):
            x[i].deactivate()
            for j in range(n):
                assert x[i, j] == 0

        # test append for depth 1 snode in both two ways
        for j in range(n):
            y.append(j)
            assert y[j] == j

        # appending elements to fully active cells will take no effect
        y.deactivate()
        for j in range(n):
            ti.append(y.parent(), [], j * j)
            assert y[j] == j * j

        # test deactivate for depth 1 snode in both two ways
        y.deactivate()
        for j in range(n):
            assert y[j] == 0
            y[j] = j

        ti.deactivate(y.parent(), [])
        for j in range(n):
            assert y[j] == 0

    test()


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal], default_fp=ti.f32, debug=True)
def test_dynamic_attributes_f32():
    _test_dynamic_attributes(ti.f32)
