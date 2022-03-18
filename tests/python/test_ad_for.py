import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.adstack)
def test_ad_sum():
    N = 10
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.i32, shape=N)
    p = ti.field(ti.f32, shape=N, needs_grad=True)

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
        p.grad[i] = 1

    compute_sum.grad()

    for i in range(N):
        assert a.grad[i] == b[i]


@test_utils.test(require=ti.extension.adstack)
def test_ad_sum_local_atomic():
    N = 10
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.i32, shape=N)
    p = ti.field(ti.f32, shape=N, needs_grad=True)

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

    compute_sum()

    for i in range(N):
        assert p[i] == 3 * b[i] + 1
        p.grad[i] = 1

    compute_sum.grad()

    for i in range(N):
        assert a.grad[i] == b[i]


@test_utils.test(require=ti.extension.adstack)
def test_ad_power():
    N = 10
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.i32, shape=N)
    p = ti.field(ti.f32, shape=N, needs_grad=True)

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

    power()

    for i in range(N):
        assert p[i] == 3**b[i]
        p.grad[i] = 1

    power.grad()

    for i in range(N):
        assert a.grad[i] == b[i] * 3**(b[i] - 1)


@test_utils.test(require=ti.extension.adstack)
def test_ad_fibonacci():
    N = 15
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.f32, shape=N, needs_grad=True)
    c = ti.field(ti.i32, shape=N)
    f = ti.field(ti.f32, shape=N, needs_grad=True)

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

    for i in range(N):
        f.grad[i] = 1

    fib.grad()

    for i in range(N):
        print(a.grad[i], b.grad[i])
        if i == 0:
            assert a.grad[i] == 0
        else:
            assert a.grad[i] == f[i - 1]
        assert b.grad[i] == f[i]


@test_utils.test(require=ti.extension.adstack)
def test_ad_fibonacci_index():
    N = 5
    M = 10
    a = ti.field(ti.f32, shape=M, needs_grad=True)
    b = ti.field(ti.f32, shape=M, needs_grad=True)
    f = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def fib():
        for i in range(N):
            p = 0
            q = 1
            for j in range(5):
                p, q = q, p + q
                b[q] += a[q]

        for i in range(M):
            f[None] += b[i]

    f.grad[None] = 1
    a.fill(1)

    fib()
    fib.grad()

    for i in range(M):
        is_fib = int(i in [1, 2, 3, 5, 8])
        assert a.grad[i] == is_fib * N
        assert b[i] == is_fib * N


@test_utils.test(require=ti.extension.adstack)
def test_ad_global_ptr():
    N = 5
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.f32, shape=N, needs_grad=True)
    f = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def task():
        for i in range(N):
            p = 0
            for j in range(N):
                b[i] += a[p]**2
                p += 1

        for i in range(N):
            f[None] += b[i]

    f.grad[None] = 1
    for i in range(N):
        a[i] = i

    task()
    task.grad()

    for i in range(N):
        print(a.grad[i])
        assert a.grad[i] == 2 * i * N


@test_utils.test(require=ti.extension.adstack)
def test_integer_stack():
    N = 5
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.f32, shape=N, needs_grad=True)
    c = ti.field(ti.i32, shape=N)
    f = ti.field(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def int_stack():
        for i in range(N):
            weight = 1
            s = 0.0
            for j in range(c[i]):
                s += weight * a[i] + b[i]
                weight *= 10
            f[i] = s

    a.fill(1)
    b.fill(1)

    for i in range(N):
        c[i] = i

    int_stack()

    for i in range(N):
        print(f[i])
        f.grad[i] = 1

    int_stack.grad()

    t = 0
    for i in range(N):
        assert a.grad[i] == t
        assert b.grad[i] == i
        t = t * 10 + 1


@test_utils.test(require=ti.extension.adstack)
def test_double_for_loops():
    N = 5
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.f32, shape=N, needs_grad=True)
    c = ti.field(ti.i32, shape=N)
    f = ti.field(ti.f32, shape=N, needs_grad=True)

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

    double_for()

    for i in range(N):
        assert f[i] == 2 * i * (1 + 2**i)
        f.grad[i] = 1

    double_for.grad()

    for i in range(N):
        assert a.grad[i] == 2 * i * i * 2**(i - 1)
        assert b.grad[i] == 2 * i


@test_utils.test(require=ti.extension.adstack)
def test_double_for_loops_more_nests():
    N = 6
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.f32, shape=N, needs_grad=True)
    c = ti.field(ti.i32, shape=(N, N // 2))
    f = ti.field(ti.f32, shape=(N, N // 2), needs_grad=True)

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
            f.grad[i, k] = 1

    double_for.grad()

    for i in range(N):
        total_grad_a = 0
        total_grad_b = 0
        for k in range(N // 2):
            total_grad_a += 2 * (i + k)**2 * 2**(i + k - 1)
            total_grad_b += 2 * (i + k)
        assert a.grad[i] == total_grad_a
        assert b.grad[i] == total_grad_b


@test_utils.test(require=[ti.extension.adstack, ti.extension.data64])
def test_complex_body():
    N = 5
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.f32, shape=N, needs_grad=True)
    c = ti.field(ti.i32, shape=N)
    f = ti.field(ti.f32, shape=N, needs_grad=True)
    g = ti.field(ti.f32, shape=N, needs_grad=False)

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
        f.grad[i] = 1

    complex()
    complex.grad()

    for i in range(N):
        assert a.grad[i] == g[i]


@test_utils.test(require=[ti.extension.adstack, ti.extension.bls])
def test_triple_for_loops_bls():
    N = 8
    M = 3
    a = ti.field(ti.f32, shape=N, needs_grad=True)
    b = ti.field(ti.f32, shape=2 * N, needs_grad=True)
    f = ti.field(ti.f32, shape=(N - M, N), needs_grad=True)

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
            f.grad[i, k] = 1

    triple_for.grad()

    for i in range(N):
        assert a.grad[i] == 2 * M * min(min(N - i - 1, i + 1), M) * \
               2**(M - 1) * N
    for i in range(N):
        assert b.grad[i * 2] == min(min(N - i - 1, i + 1), M) * N
        assert b.grad[i * 2 + 1] == min(min(N - i - 1, i + 1), M) * N


@test_utils.test(require=ti.extension.adstack)
def test_mixed_inner_loops():
    x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    arr = ti.field(dtype=ti.f32, shape=(5))
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def mixed_inner_loops():
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(2):
                loss[None] += ti.sin(x[None]) + 1.0

    loss.grad[None] = 1.0
    x[None] = 0.0
    mixed_inner_loops()
    mixed_inner_loops.grad()

    assert loss[None] == 10.0
    assert x.grad[None] == 15.0


@test_utils.test(require=ti.extension.adstack)
def test_mixed_inner_loops_tape():
    x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    arr = ti.field(dtype=ti.f32, shape=(5))
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def mixed_inner_loops_tape():
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(2):
                loss[None] += ti.sin(x[None]) + 1.0

    x[None] = 0.0
    with ti.Tape(loss=loss):
        mixed_inner_loops_tape()

    assert loss[None] == 10.0
    assert x.grad[None] == 15.0


@test_utils.test(require=ti.extension.adstack, ad_stack_size=32)
def test_inner_loops_local_variable_fixed_stack_size_tape():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_inner_loops_local_variable():
        for i in arr:
            for j in range(3):
                s = 0.0
                t = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                    t += ti.sin(x[None])
                loss[None] += s + t

    x[None] = 0.0
    with ti.Tape(loss=loss):
        test_inner_loops_local_variable()

    assert loss[None] == 18.0
    assert x.grad[None] == 36.0


@test_utils.test(require=ti.extension.adstack, ad_stack_size=32)
def test_inner_loops_local_variable_fixed_stack_size_kernel_grad():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_inner_loops_local_variable():
        for i in arr:
            for j in range(3):
                s = 0.0
                t = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                    t += ti.sin(x[None])
                loss[None] += s + t

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_inner_loops_local_variable()
    test_inner_loops_local_variable.grad()

    assert loss[None] == 18.0
    assert x.grad[None] == 36.0


@test_utils.test(require=ti.extension.adstack, ad_stack_size=0)
def test_inner_loops_local_variable_adaptive_stack_size_tape():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_inner_loops_local_variable():
        for i in arr:
            for j in range(3):
                s = 0.0
                t = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                    t += ti.sin(x[None])
                loss[None] += s + t

    x[None] = 0.0
    with ti.Tape(loss=loss):
        test_inner_loops_local_variable()

    assert loss[None] == 18.0
    assert x.grad[None] == 36.0


@test_utils.test(require=ti.extension.adstack, ad_stack_size=0)
def test_inner_loops_local_variable_adaptive_stack_size_kernel_grad():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_inner_loops_local_variable():
        for i in arr:
            for j in range(3):
                s = 0.0
                t = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                    t += ti.sin(x[None])
                loss[None] += s + t

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_inner_loops_local_variable()
    test_inner_loops_local_variable.grad()

    assert loss[None] == 18.0
    assert x.grad[None] == 36.0


@test_utils.test(require=ti.extension.adstack, ad_stack_size=0)
def test_more_inner_loops_local_variable_adaptive_stack_size_tape():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_more_inner_loops_local_variable():
        for i in arr:
            for j in range(2):
                s = 0.0
                for k in range(3):
                    u = 0.0
                    s += ti.sin(x[None]) + 1.0
                    for l in range(2):
                        u += ti.sin(x[None])
                    loss[None] += u
                loss[None] += s

    x[None] = 0.0
    with ti.Tape(loss=loss):
        test_more_inner_loops_local_variable()

    assert loss[None] == 12.0
    assert x.grad[None] == 36.0


@test_utils.test(require=ti.extension.adstack, ad_stack_size=32)
def test_more_inner_loops_local_variable_fixed_stack_size_tape():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_more_inner_loops_local_variable():
        for i in arr:
            for j in range(2):
                s = 0.0
                for k in range(3):
                    u = 0.0
                    s += ti.sin(x[None]) + 1.0
                    for l in range(2):
                        u += ti.sin(x[None])
                    loss[None] += u
                loss[None] += s

    x[None] = 0.0
    with ti.Tape(loss=loss):
        test_more_inner_loops_local_variable()

    assert loss[None] == 12.0
    assert x.grad[None] == 36.0


@test_utils.test(require=ti.extension.adstack,
                 ad_stack_size=32,
                 arch=[ti.cpu, ti.gpu])
def test_stacked_inner_loops_local_variable_fixed_stack_size_kernel_grad():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_stacked_inner_loops_local_variable():
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_inner_loops_local_variable()
    test_stacked_inner_loops_local_variable.grad()

    assert loss[None] == 36.0
    assert x.grad[None] == 38.0


@test_utils.test(require=ti.extension.adstack,
                 ad_stack_size=32,
                 arch=[ti.cpu, ti.gpu])
def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable_fixed_stack_size_kernel_grad(
):
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable():
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(3):
                for k in range(3):
                    loss[None] += ti.sin(x[None]) + 1.0
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                for k in range(3):
                    loss[None] += ti.sin(x[None]) + 1.0

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable()
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable.grad()

    assert loss[None] == 54.0
    assert x.grad[None] == 56.0


@test_utils.test(require=ti.extension.adstack,
                 ad_stack_size=0,
                 arch=[ti.cpu, ti.gpu])
def test_stacked_inner_loops_local_variable_adaptive_stack_size_kernel_grad():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_stacked_inner_loops_local_variable():
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_inner_loops_local_variable()
    test_stacked_inner_loops_local_variable.grad()

    assert loss[None] == 36.0
    assert x.grad[None] == 38.0


@test_utils.test(require=ti.extension.adstack,
                 ad_stack_size=0,
                 arch=[ti.cpu, ti.gpu])
def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable_adaptive_stack_size_kernel_grad(
):
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable():
        for i in arr:
            loss[None] += ti.sin(x[None])
            for j in range(3):
                for k in range(3):
                    loss[None] += ti.sin(x[None]) + 1.0
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ti.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                for k in range(3):
                    loss[None] += ti.sin(x[None]) + 1.0

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable()
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable.grad()

    assert loss[None] == 54.0
    assert x.grad[None] == 56.0


@test_utils.test(require=ti.extension.adstack,
                 ad_stack_size=0,
                 arch=[ti.cpu, ti.gpu])
def test_large_for_loops_adaptive_stack_size():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_large_loop():
        for i in range(5):
            for j in range(2000):
                for k in range(1000):
                    loss[None] += ti.sin(x[None]) + 1.0

    with ti.Tape(loss=loss):
        test_large_loop()

    assert loss[None] == 1e7
    assert x.grad[None] == 1e7


@test_utils.test(require=ti.extension.adstack,
                 ad_stack_size=1,
                 arch=[ti.cpu, ti.gpu])
def test_large_for_loops_fixed_stack_size():
    x = ti.field(dtype=float, shape=(), needs_grad=True)
    arr = ti.field(dtype=float, shape=(2), needs_grad=True)
    loss = ti.field(dtype=float, shape=(), needs_grad=True)

    @ti.kernel
    def test_large_loop():
        for i in range(5):
            for j in range(2000):
                for k in range(1000):
                    loss[None] += ti.sin(x[None]) + 1.0

    with ti.Tape(loss=loss):
        test_large_loop()

    assert loss[None] == 1e7
    assert x.grad[None] == 1e7


@test_utils.test(require=ti.extension.adstack)
def test_multiple_ib():
    x = ti.field(float, (), needs_grad=True)
    y = ti.field(float, (), needs_grad=True)

    @ti.kernel
    def compute_y():
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with ti.Tape(y):
        compute_y()

    assert y[None] == 12.0
    assert x.grad[None] == 12.0


@test_utils.test(require=ti.extension.adstack)
def test_multiple_ib_multiple_outermost():
    x = ti.field(float, (), needs_grad=True)
    y = ti.field(float, (), needs_grad=True)

    @ti.kernel
    def compute_y():
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with ti.Tape(y):
        compute_y()

    assert y[None] == 24.0
    assert x.grad[None] == 24.0


@test_utils.test(require=ti.extension.adstack)
def test_multiple_ib_multiple_outermost_mixed():
    x = ti.field(float, (), needs_grad=True)
    y = ti.field(float, (), needs_grad=True)

    @ti.kernel
    def compute_y():
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
                for ii in range(3):
                    y[None] += x[None]

    x[None] = 1.0
    with ti.Tape(y):
        compute_y()

    assert y[None] == 42.0
    assert x.grad[None] == 42.0


@test_utils.test(require=ti.extension.adstack)
def test_multiple_ib_mixed():
    x = ti.field(float, (), needs_grad=True)
    y = ti.field(float, (), needs_grad=True)

    @ti.kernel
    def compute_y():
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
                for k in range(2):
                    y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with ti.Tape(y):
        compute_y()

    assert y[None] == 30.0
    assert x.grad[None] == 30.0


@test_utils.test(require=ti.extension.adstack)
def test_multiple_ib_deeper():
    x = ti.field(float, (), needs_grad=True)
    y = ti.field(float, (), needs_grad=True)

    @ti.kernel
    def compute_y():
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    for iii in range(2):
                        y[None] += x[None]

    x[None] = 1.0
    with ti.Tape(y):
        compute_y()

    assert y[None] == 42.0
    assert x.grad[None] == 42.0


@test_utils.test(require=ti.extension.adstack)
def test_multiple_ib_deeper_non_scalar():
    N = 10
    x = ti.field(float, shape=N, needs_grad=True)
    y = ti.field(float, shape=N, needs_grad=True)

    @ti.kernel
    def compute_y():
        for j in range(N):
            for i in range(j):
                y[j] += x[j]
            for i in range(3):
                for ii in range(j):
                    y[j] += x[j]
            for i in range(3):
                for ii in range(2):
                    for iii in range(j):
                        y[j] += x[j]

    x.fill(1.0)
    for i in range(N):
        y.grad[i] = 1.0
    compute_y()
    compute_y.grad()
    for i in range(N):
        assert y[i] == i * 10.0
        assert x.grad[i] == i * 10.0


@test_utils.test(require=ti.extension.adstack)
def test_multiple_ib_inner_mixed():
    x = ti.field(float, (), needs_grad=True)
    y = ti.field(float, (), needs_grad=True)

    @ti.kernel
    def compute_y():
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    y[None] += x[None]
                for iii in range(2):
                    y[None] += x[None]
                    for iiii in range(2):
                        y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    for iii in range(2):
                        y[None] += x[None]

    x[None] = 1.0
    with ti.Tape(y):
        compute_y()

    assert y[None] == 78.0
    assert x.grad[None] == 78.0
