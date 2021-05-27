import taichi as ti
from taichi import approx


def archs_support_random(func):
    return ti.archs_excluding(ti.metal)(func)


@archs_support_random
def test_random_float():
    for precision in [ti.f32, ti.f64]:
        ti.init()
        n = 1024
        x = ti.field(ti.f32, shape=(n, n))

        @ti.kernel
        def fill():
            for i in range(n):
                for j in range(n):
                    x[i, j] = ti.random(precision)

        fill()
        X = x.to_numpy()
        for i in range(1, 4):
            assert (X**i).mean() == approx(1 / (i + 1), rel=1e-2)


@archs_support_random
def test_random_int():
    for precision in [ti.i32, ti.i64]:
        ti.init()
        n = 1024
        x = ti.field(ti.f32, shape=(n, n))

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
        for i in range(1, 4):
            assert (X**i).mean() == approx(1 / (i + 1), rel=1e-2)


@archs_support_random
def test_random_independent_product():
    n = 1024
    x = ti.field(ti.f32, shape=n * n)

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

    x = ti.Vector.field(2, dtype=ti.f32, shape=n)

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


@archs_support_random
def test_random_seed_per_launch():
    n = 10
    x = ti.field(ti.f32, shape=n)

    @ti.kernel
    def gen(i: ti.i32):
        x[i] = ti.random()

    count = 0
    gen(0)
    for i in range(1, n):
        gen(i)
        count += 1 if x[i] == x[i - 1] else 0

    assert count <= n * 0.15


@ti.test(arch=[ti.cpu, ti.cuda])
def test_random_seed_per_program():
    import numpy as np
    n = 10
    result = []
    for s in [0, 1]:
        ti.init(random_seed=s)
        x = ti.field(ti.f32, shape=n)

        @ti.kernel
        def gen():
            for i in x:
                x[i] = ti.random()

        gen()
        result.append(x.to_numpy())
        ti.reset()

    assert not np.allclose(result[0], result[1])


@ti.test(arch=[ti.cpu, ti.cuda])
def test_random_f64():
    '''
    Tests the granularity of float64 random numbers.
    See https://github.com/taichi-dev/taichi/issues/2251 for an explanation.
    '''
    import numpy as np
    n = int(2**23)
    x = ti.field(ti.f64, shape=n)

    @ti.kernel
    def foo() -> ti.f64:
        for i in x:
            x[i] = ti.random(dtype=ti.f64)

    foo()
    frac, _ = np.modf(x.to_numpy() * 4294967296)
    assert np.max(frac) > 0


@archs_support_random
def test_randn():
    '''
    Tests the generation of Gaussian random numbers.
    '''
    for precision in [ti.f32, ti.f64]:
        ti.init()
        n = 1024
        x = ti.field(ti.f32, shape=(n, n))

        @ti.kernel
        def fill():
            for i in range(n):
                for j in range(n):
                    x[i, j] = ti.randn(precision)

        fill()
        X = x.to_numpy()

        # https://en.wikipedia.org/wiki/Normal_distribution#Moments
        moments = [0.0, 1.0, 0.0, 3.0]
        for i in range(4):
            assert (X**(i + 1)).mean() == approx(moments[i], abs=3e-2)
