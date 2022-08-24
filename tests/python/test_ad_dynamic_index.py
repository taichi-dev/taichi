import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.dynamic_index,
                 dynamic_index=True,
                 debug=True)
def test_matrix_non_constant_index():
    m = ti.Matrix.field(2, 2, ti.f32, 5, needs_grad=True)
    n = ti.Matrix.field(2, 2, ti.f32, 5, needs_grad=True)
    loss = ti.field(ti.f32, (), needs_grad=True)

    n.fill(0)

    @ti.kernel
    def func1():
        for i in range(5):
            for j, k in ti.ndrange(2, 2):
                m[i][j, k] = (j + 1) * (k + 1) * n[i][j, k]
                loss[None] += m[i][j, k]

    loss.grad[None] = 1.0
    func1.grad()

    for i in range(5):
        for j in range(2):
            for k in range(2):
                assert n.grad[i][j, k] == (j + 1) * (k + 1)
