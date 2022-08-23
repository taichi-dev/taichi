import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.adstack, dynamic_index=True)
def test_polar_decompose_2D():
    # `polar_decompose3d` in current Taichi version (v1.1) does not support autodiff,
    # becasue it mixed usage of for-loops and statements without looping.
    dim = 2
    F_1 = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(), needs_grad=True)
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(), needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def polar_decompose_2D():
        r, s = ti.polar_decompose(F[None])
        F_1[None] += r

    with ti.ad.Tape(loss=loss):
        polar_decompose_2D()
