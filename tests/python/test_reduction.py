import taichi as ti
from pytest import approx


def _test_reduction_single(dtype, criterion):
    N = 1024 * 1024
    if ti.cfg.arch == ti.opengl and dtype == ti.f32:
        # OpenGL is not capable of such large number in its float32...
        N = 1024 * 16

    a = ti.field(dtype, shape=N)
    tot = ti.field(dtype, shape=())

    @ti.kernel
    def fill():
        for i in a:
            a[i] = i

    @ti.kernel
    def reduce():
        for i in a:
            tot[None] += a[i]

    @ti.kernel
    def reduce_tmp() -> dtype:
        s = ti.zero(tot[None])
        for i in a:
            s += a[i]
        return s

    fill()
    reduce()
    tot2 = reduce_tmp()

    ground_truth = N * (N - 1) / 2
    assert criterion(tot[None], ground_truth)
    assert criterion(tot2, ground_truth)


@ti.all_archs
def test_reduction_single_i32():
    _test_reduction_single(ti.i32, lambda x, y: x % 2**32 == y % 2**32)


@ti.test(exclude=ti.opengl)
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

    a = ti.field(ti.i32, shape=N)
    b = ti.field(ti.f64, shape=N)
    tot_a = ti.field(ti.i32, shape=())
    tot_b = ti.field(ti.f64, shape=())

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


@ti.all_archs
def test_reduction_different_scale():
    @ti.kernel
    def func(n: ti.template()) -> ti.i32:
        x = 0
        for i in range(n):
            ti.atomic_add(x, 1)
        return x

    # 10 and 60 since OpenGL TLS stride size = 32
    # 1024 and 100000 since OpenGL max threads per group ~= 1792
    for n in [1, 10, 60, 1024, 100000]:
        assert n == func(n)
