import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_ad_sum_fwd():
    N = 10
    a = ti.field(ti.f32, shape=N)
    b = ti.field(ti.i32, shape=N)
    p = ti.field(ti.f32, shape=N)
    ti.root.lazy_dual()

    @ti.kernel
    def compute_sum():
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret = ret + a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    compute_sum()

    for i in range(N):
        assert p[i] == 3 * b[i] + 1

    with ti.ad.FwdMode(loss=p, parameters=a, seed=[1.0 for _ in range(N)]):
        compute_sum()

    for i in range(N):
        assert p.dual[i] == b[i]


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_ad_fibonacci_fwd():
    N = 15
    a = ti.field(ti.f32, shape=N)
    b = ti.field(ti.f32, shape=N)
    c = ti.field(ti.i32, shape=N)
    f = ti.field(ti.f32, shape=N)
    ti.root.lazy_dual()

    @ti.kernel
    def fib():
        for i in range(N):
            p = a[i]
            q = b[i]
            for j in range(c[i]):
                p, q = q, p + q
            f[i] = q

    b.fill(1)

    for i in range(N):
        c[i] = i

    fib()
    with ti.ad.FwdMode(loss=f, parameters=a, seed=[1.0 for _ in range(N)]):
        fib()

    for i in range(N):
        if i == 0:
            assert f.dual[i] == 0
        else:
            assert f.dual[i] == f[i - 1]

    with ti.ad.FwdMode(loss=f, parameters=b, seed=[1.0 for _ in range(N)]):
        fib()
    for i in range(N):
        assert f.dual[i] == f[i]
