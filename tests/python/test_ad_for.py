import taichi as ti


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_sum():
    N = 10
    a = ti.var(ti.f32, shape=N, needs_grad=True)
    b = ti.var(ti.i32, shape=N)
    p = ti.var(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def comptue_sum():
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret = ret + a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    comptue_sum()

    for i in range(N):
        assert p[i] == 3 * b[i] + 1
        p.grad[i] = 1

    comptue_sum.grad()

    for i in range(N):
        assert a.grad[i] == b[i]


# @ti.require(ti.extension.adstack)
# @ti.all_archs
def test_ad_sum_local_atomic():
    ti.init(print_ir=True)
    N = 10
    a = ti.var(ti.f32, shape=N, needs_grad=True)
    b = ti.var(ti.i32, shape=N)
    p = ti.var(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def comptue_sum():
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret += a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    comptue_sum()

    for i in range(N):
        assert p[i] == 3 * b[i] + 1
        p.grad[i] = 1

    comptue_sum.grad()

    for i in range(N):
        assert a.grad[i] == b[i]


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_power():
    N = 10
    a = ti.var(ti.f32, shape=N, needs_grad=True)
    b = ti.var(ti.i32, shape=N)
    p = ti.var(ti.f32, shape=N, needs_grad=True)

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


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_fibonacci():
    N = 15
    a = ti.var(ti.f32, shape=N, needs_grad=True)
    b = ti.var(ti.f32, shape=N, needs_grad=True)
    c = ti.var(ti.i32, shape=N)
    f = ti.var(ti.f32, shape=N, needs_grad=True)

    @ti.kernel
    def fib():
        for i in range(N):
            p = a[i]
            q = b[i]
            for j in range(c[i]):
                new_p = q
                new_q = p + q
                p, q = new_p, new_q
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


# TODO: test integer stack (primal without adjoint)

# test_ad_fibonacci()
test_ad_sum_local_atomic()
