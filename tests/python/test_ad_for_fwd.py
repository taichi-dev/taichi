import taichi as ti
from tests import test_utils


@test_utils.test()
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
        assert p[i] == a[i] * b[i] + 1

    with ti.ad.FwdMode(loss=p, parameters=a, seed=[1.0 for _ in range(N)]):
        compute_sum()

    for i in range(N):
        assert p.dual[i] == b[i]
        assert a.dual[i] == 0

    with ti.ad.FwdMode(loss=p,
                       parameters=a,
                       seed=[1.0 for _ in range(N)],
                       clear_gradients=False):
        pass

    for i in range(N):
        assert p.dual[i] == b[i]


@test_utils.test()
def test_ad_sum_local_atomic_fwd():
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
                ret += a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    with ti.ad.FwdMode(loss=p, parameters=a, seed=[1.0 for _ in range(N)]):
        compute_sum()

    for i in range(N):
        assert p[i] == a[i] * b[i] + 1

    for i in range(N):
        assert p.dual[i] == b[i]


@test_utils.test()
def test_ad_power_fwd():
    N = 10
    a = ti.field(ti.f32, shape=N)
    b = ti.field(ti.i32, shape=N)
    p = ti.field(ti.f32, shape=N)
    ti.root.lazy_dual()

    @ti.kernel
    def power():
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret = ret * a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    with ti.ad.FwdMode(loss=p, parameters=a, seed=[1.0 for _ in range(N)]):
        power()

    for i in range(N):
        assert p[i] == 3**b[i]

    for i in range(N):
        assert p.dual[i] == b[i] * 3**(b[i] - 1)


@test_utils.test()
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


@test_utils.test()
def test_ad_fibonacci_index_fwd():
    N = 5
    M = 10
    a = ti.field(ti.f32, shape=M)
    b = ti.field(ti.f32, shape=M)
    f = ti.field(ti.f32, shape=M)
    ti.root.lazy_dual()

    @ti.kernel
    def fib():
        for i in range(N):
            p = 0
            q = 1
            for j in range(5):
                p, q = q, p + q
                b[q] += a[q]

        for i in range(M):
            f[i] += b[i]

    a.fill(1)

    with ti.ad.FwdMode(loss=f, parameters=a, seed=[1.0 for _ in range(M)]):
        fib()

    for i in range(M):
        is_fib = int(i in [1, 2, 3, 5, 8])
        assert f.dual[i] == is_fib * N
        assert b[i] == is_fib * N


@test_utils.test(exclude=[ti.cc])
def test_double_for_loops():
    N = 5
    a = ti.field(ti.f32, shape=N)
    b = ti.field(ti.f32, shape=N)
    c = ti.field(ti.i32, shape=N)
    f = ti.field(ti.f32, shape=N)
    ti.root.lazy_dual()

    @ti.kernel
    def double_for():
        for i in range(N):
            weight = 1.0
            for j in range(c[i]):
                weight *= a[i]
            s = 0.0
            for j in range(c[i] * 2):
                s += weight + b[i]
            f[i] = s

    a.fill(2)
    b.fill(1)

    for i in range(N):
        c[i] = i

    with ti.ad.FwdMode(loss=f, parameters=a, seed=[1.0 for _ in range(N)]):
        double_for()

    for i in range(N):
        assert f.dual[i] == 2 * i * i * 2**(i - 1)

    with ti.ad.FwdMode(loss=f, parameters=b, seed=[1.0 for _ in range(N)]):
        double_for()

    for i in range(N):
        assert f.dual[i] == 2 * i


@test_utils.test(exclude=[ti.cc])
def test_double_for_loops_more_nests():
    N = 6
    a = ti.field(ti.f32, shape=N, needs_dual=True)
    b = ti.field(ti.f32, shape=N, needs_dual=True)
    c = ti.field(ti.i32, shape=(N, N // 2))
    f = ti.field(ti.f32, shape=(N, N // 2), needs_dual=True)

    @ti.kernel
    def double_for():
        for i in range(N):
            for k in range(N // 2):
                weight = 1.0
                for j in range(c[i, k]):
                    weight *= a[i]
                s = 0.0
                for j in range(c[i, k] * 2):
                    s += weight + b[i]
                f[i, k] = s

    a.fill(2)
    b.fill(1)

    for i in range(N):
        for k in range(N // 2):
            c[i, k] = i + k

    double_for()

    for i in range(N):
        for k in range(N // 2):
            assert f[i, k] == 2 * (i + k) * (1 + 2**(i + k))

    with ti.ad.FwdMode(loss=f, parameters=a, seed=[1.0 for _ in range(N)]):
        double_for()

    for i in range(N):
        total_grad_a = 0
        for k in range(N // 2):
            total_grad_a = 2 * (i + k)**2 * 2**(i + k - 1)
            assert f.dual[i, k] == total_grad_a

    with ti.ad.FwdMode(loss=f, parameters=b, seed=[1.0 for _ in range(N)]):
        double_for()

    for i in range(N):
        total_grad_b = 0
        for k in range(N // 2):
            total_grad_b = 2 * (i + k)
            assert f.dual[i, k] == total_grad_b


@test_utils.test(exclude=[ti.cc])
def test_complex_body():
    N = 5
    a = ti.field(ti.f32, shape=N, needs_dual=True)
    b = ti.field(ti.f32, shape=N, needs_dual=True)
    c = ti.field(ti.i32, shape=N)
    f = ti.field(ti.f32, shape=N, needs_dual=True)
    g = ti.field(ti.f32, shape=N, needs_dual=False)

    @ti.kernel
    def complex():
        for i in range(N):
            weight = 2.0
            tot = 0.0
            tot_weight = 0.0
            for j in range(c[i]):
                tot_weight += weight + 1
                tot += (weight + 1) * a[i]
                weight = weight + 1
                weight = weight * 4
                weight = ti.cast(weight, ti.f64)
                weight = ti.cast(weight, ti.f32)

            g[i] = tot_weight
            f[i] = tot

    a.fill(2)
    b.fill(1)

    for i in range(N):
        c[i] = i

    with ti.ad.FwdMode(loss=f, parameters=a, seed=[1.0 for _ in range(N)]):
        complex()

    for i in range(N):
        assert f.dual[i] == g[i]


@test_utils.test(exclude=[ti.cc])
def test_triple_for_loops_bls():
    N = 8
    M = 3
    a = ti.field(ti.f32, shape=N, needs_dual=True)
    b = ti.field(ti.f32, shape=2 * N, needs_dual=True)
    f = ti.field(ti.f32, shape=(N - M, N), needs_dual=True)

    @ti.kernel
    def triple_for():
        ti.block_local(a)
        ti.block_local(b)
        for i in range(N - M):
            for k in range(N):
                weight = 1.0
                for j in range(M):
                    weight *= a[i + j]
                s = 0.0
                for j in range(2 * M):
                    s += weight + b[2 * i + j]
                f[i, k] = s

    a.fill(2)

    for i in range(2 * N):
        b[i] = i

    triple_for()

    for i in range(N - M):
        for k in range(N):
            assert f[i, k] == 2 * M * 2**M + (4 * i + 2 * M - 1) * M

    with ti.ad.FwdMode(loss=f, parameters=a, seed=[1.0 for _ in range(N)]):
        triple_for()

    for i in range(N - M):
        for k in range(N):
            assert f.dual[i, k] == 2 * M * M * 2**(M - 1)

    with ti.ad.FwdMode(loss=f, parameters=b, seed=[1.0 for _ in range(2 * N)]):
        triple_for()

    for i in range(N - M):
        for k in range(N):
            assert f.dual[i, k] == 2 * M
