import taichi as ti


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_if_simple():
    x = ti.field(ti.f32, shape=())
    y = ti.field(ti.f32, shape=())

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        if x[None] > 0.:
            y[None] = x[None]

    x[None] = 1
    y.grad[None] = 1

    func()
    func.grad()

    assert x.grad[None] == 1


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_if():
    x = ti.field(ti.f32, shape=2)
    y = ti.field(ti.f32, shape=2)

    ti.root.lazy_grad()

    @ti.kernel
    def func(i: ti.i32):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = 2 * x[i]

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0)
    func.grad(0)
    func(1)
    func.grad(1)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_if_nested():
    n = 20
    x = ti.field(ti.f32, shape=n)
    y = ti.field(ti.f32, shape=n)
    z = ti.field(ti.f32, shape=n)

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in x:
            if x[i] < 2:
                if x[i] == 0:
                    y[i] = 0
                else:
                    y[i] = z[i] * 1
            else:
                if x[i] == 2:
                    y[i] = z[i] * 2
                else:
                    y[i] = z[i] * 3

    z.fill(1)

    for i in range(n):
        x[i] = i % 4

    func()
    for i in range(n):
        assert y[i] == i % 4
        y.grad[i] = 1
    func.grad()

    for i in range(n):
        assert z.grad[i] == i % 4


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_if_mutable():
    x = ti.field(ti.f32, shape=2)
    y = ti.field(ti.f32, shape=2)

    ti.root.lazy_grad()

    @ti.kernel
    def func(i: ti.i32):
        t = x[i]
        if t > 0:
            y[i] = t
        else:
            y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0)
    func.grad(0)
    func(1)
    func.grad(1)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_if_parallel():
    x = ti.field(ti.f32, shape=2)
    y = ti.field(ti.f32, shape=2)

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@ti.require(ti.extension.adstack, ti.extension.data64)
@ti.all_archs_with(default_fp=ti.f64)
def test_ad_if_parallel_f64():
    x = ti.field(ti.f64, shape=2)
    y = ti.field(ti.f64, shape=2)

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@ti.require(ti.extension.adstack)
@ti.all_archs
def test_ad_if_parallel_complex():
    x = ti.field(ti.f32, shape=2)
    y = ti.field(ti.f32, shape=2)

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        ti.parallelize(1)
        for i in range(2):
            t = 0.0
            if x[i] > 0:
                t = 1 / x[i]
            y[i] = t

    x[0] = 0
    x[1] = 2
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 0
    assert x.grad[1] == -0.25


@ti.test(require=[ti.extension.adstack, ti.extension.data64],
         default_fp=ti.f64)
def test_ad_if_parallel_complex_f64():
    x = ti.field(ti.f64, shape=2)
    y = ti.field(ti.f64, shape=2)

    ti.root.lazy_grad()

    @ti.kernel
    def func():
        ti.parallelize(1)
        for i in range(2):
            t = 0.0
            if x[i] > 0:
                t = 1 / x[i]
            y[i] = t

    x[0] = 0
    x[1] = 2
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 0
    assert x.grad[1] == -0.25


@ti.host_arch_only
def test_stack():
    @ti.kernel
    def func():
        ti.call_internal("test_stack")

    func()
