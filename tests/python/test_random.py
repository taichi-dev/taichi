import taichi as ti
from taichi import approx


def archs_support_random(func):
    return ti.archs_excluding(ti.metal)(func)


@archs_support_random
def test_random_float():
    for precision in [ti.f32, ti.f64]:
        ti.init()
        n = 1024
        x = ti.var(ti.f32, shape=(n, n))

        @ti.kernel
        def fill():
            for i in range(n):
                for j in range(n):
                    x[i, j] = ti.random(precision)

        fill()
        X = x.to_numpy()
        for i in range(4):
            assert (X**i).mean() == approx(1 / (i + 1), rel=1e-2)


@archs_support_random
def test_random_int():
    for precision in [ti.i32, ti.i64]:
        ti.init()
        n = 1024
        x = ti.var(ti.f32, shape=(n, n))

        @ti.kernel
        def fill():
            for i in range(n):
                for j in range(n):
                    v = ti.random(precision)
                    if precision == ti.i32:
                        x[i, j] = (float(v) + float(2**31)) / float(2**32)
                    else:
                        x[i, j] = (float(v) + float(2**63)) / float(2**64)

        fill()
        X = x.to_numpy()
        for i in range(4):
            assert (X**i).mean() == approx(1 / (i + 1), rel=1e-2)


@archs_support_random
def test_random_independent_product():
    n = 1024
    x = ti.var(ti.f32, shape=n * n)

    @ti.kernel
    def fill():
        for i in range(n * n):
            a = ti.random()
            b = ti.random()
            x[i] = a * b

    fill()
    X = x.to_numpy()
    for i in range(4):
        assert X.mean() == approx(1 / 4, rel=1e-2)


@archs_support_random
def test_random_2d_dist():
    n = 8192

    x = ti.Vector(2, dt=ti.f32, shape=n)

    @ti.kernel
    def gen():
        for i in range(n):
            x[i] = [ti.random(), ti.random()]

    gen()

    X = x.to_numpy()
    counters = [0 for _ in range(4)]
    for i in range(n):
        c = int(X[i, 0] < 0.5) * 2 + int(X[i, 1] < 0.5)
        counters[c] += 1

    for c in range(4):
        assert counters[c] / n == approx(1 / 4, rel=0.2)
