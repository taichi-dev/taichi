import taichi as ti
from pytest import approx


def _test_reduction_single(dtype, criterion):
    N = 1024 * 1024

    a = ti.var(dtype, shape=N)
    tot = ti.var(dtype, shape=())

    @ti.kernel
    def fill():
        for i in a:
            a[i] = i

    @ti.kernel
    def reduce():
        for i in a:
            tot[None] += a[i]

    fill()
    reduce()

    ground_truth = N * (N - 1) / 2
    assert criterion(tot[None], ground_truth)


@ti.all_archs
def test_reduction_single_i32():
    _test_reduction_single(ti.i32, lambda x, y: x % 2**32 == y % 2**32)


@ti.archs_excluding(ti.opengl)
def test_reduction_single_u32():
    _test_reduction_single(ti.u32, lambda x, y: x % 2**32 == y % 2**32)


@ti.all_archs
def test_reduction_single_f32():
    _test_reduction_single(ti.f32, lambda x, y: x == approx(y, 3e-4))


@ti.require(ti.extension.data64)
@ti.all_archs
def test_reduction_single_i64():
    _test_reduction_single(ti.i64, lambda x, y: x % 2**64 == y % 2**64)


@ti.require(ti.extension.data64)
@ti.archs_excluding(ti.opengl)  # OpenGL doesn't have u64 yet
def test_reduction_single_u64():
    _test_reduction_single(ti.u64, lambda x, y: x % 2**64 == y % 2**64)


@ti.require(ti.extension.data64)
@ti.all_archs
def test_reduction_single_f64():
    _test_reduction_single(ti.f64, lambda x, y: x == approx(y, 1e-12))


@ti.require(ti.extension.data64)
@ti.all_archs
def test_reduction_single():
    N = 1024 * 1024

    a = ti.var(ti.i32, shape=N)
    b = ti.var(ti.f64, shape=N)
    tot_a = ti.var(ti.i32, shape=())
    tot_b = ti.var(ti.f64, shape=())

    @ti.kernel
    def fill():
        for i in a:
            a[i] = i
            b[i] = i * 2

    @ti.kernel
    def reduce():
        for i in a:
            tot_a[None] += a[i]
            tot_b[None] += b[i]

    fill()
    reduce()

    ground_truth = N * (N - 1) / 2
    assert tot_a[None] % 2**32 == ground_truth % 2**32
    assert tot_b[None] / 2 == approx(ground_truth, 1e-12)
