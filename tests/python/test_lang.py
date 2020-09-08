import taichi as ti
import pytest


@ti.all_archs
def test_nested_subscript():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    ti.root.dense(ti.i, 1).place(x)
    ti.root.dense(ti.i, 1).place(y)

    x[0] = 0

    @ti.kernel
    def inc():
        for i in range(1):
            x[x[i]] += 1

    inc()

    assert x[0] == 1


@ti.all_archs
def test_norm():
    val = ti.field(ti.i32)
    f = ti.field(ti.f32)

    n = 1024

    ti.root.dense(ti.i, n).dense(ti.i, 2).place(val, f)

    @ti.kernel
    def test():
        for i in range(n):
            s = 0
            for j in range(10):
                s += j
            a = ti.Vector([0.4, 0.3])
            val[i] = s + ti.cast(a.norm() * 100, ti.i32) + i

    test()

    @ti.kernel
    def test2():
        for i in range(n):
            val[i] += 1

    test2()

    for i in range(n):
        assert val[i] == 96 + i


@ti.all_archs
def test_simple2():
    val = ti.field(ti.i32)
    f = ti.field(ti.f32)

    n = 16

    ti.root.dense(ti.i, n).place(val, f)

    @ti.kernel
    def test():
        for i in range(n):
            val[i] = i * 2

    test()

    @ti.kernel
    def test2():
        for i in range(n):
            val[i] += 1

    test2()

    for i in range(n):
        assert val[i] == 1 + i * 2


@ti.all_archs
def test_recreate():
    @ti.kernel
    def test():
        a = 0
        a, b = 1, 2

    test()


@ti.all_archs
def test_local_atomics():
    n = 32
    val = ti.field(ti.i32, shape=n)

    @ti.kernel
    def test():
        for i in range(n):
            s = 0
            s += 45
            print(s)
            val[i] = s + i
            print(val[i])

    test()

    for i in range(n):
        assert val[i] == i + 45


@ti.must_throw(UnboundLocalError)
@ti.host_arch_only
def test_loop_var_life():
    @ti.kernel
    def test():
        for i in ti.static(range(8)):
            pass
        print(i)

    test()


@ti.must_throw(UnboundLocalError)
@ti.host_arch_only
def test_loop_var_life_double_iters():
    @ti.kernel
    def test():
        for i, v in ti.static(enumerate(range(8))):
            pass
        print(i)

    test()


@ti.test(require=ti.extension.data64)
@pytest.mark.parametrize('dt', [ti.i32, ti.f32, ti.i64, ti.f64])
def test_meta_zero(dt):
    @ti.kernel
    def func(x: dt) -> dt:
        return ti.zero(x)

    for x in [-1, -2.3, 1.2, 2, 3]:
        if ti.core.is_integral(dt):
            x = int(x)
        assert func(x) == 0  # intentionally not using ti.approx here


@ti.test(require=ti.extension.data64)
@pytest.mark.parametrize('dt', [ti.i32, ti.f32, ti.i64, ti.f64])
def test_meta_one(dt):
    @ti.kernel
    def func(x: dt) -> dt:
        return ti.one(x)

    for x in [-1, -2.3, 1.2, 2, 3]:
        if ti.core.is_integral(dt):
            x = int(x)
        assert func(x) == 1
