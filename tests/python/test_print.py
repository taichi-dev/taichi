import pytest

import taichi as ti


# Not really testable..
# Just making sure it does not crash
# Metal doesn't support print() or 64-bit data
# While OpenGL does support print, but not 64-bit data
@pytest.mark.parametrize('dt', [ti.i32, ti.f32, ti.i64, ti.f64])
@ti.test(exclude=[ti.metal, ti.opengl])
def test_print(dt):
    @ti.kernel
    def func():
        print(ti.cast(1234.5, dt))

    func()
    # Discussion: https://github.com/taichi-dev/taichi/issues/1063#issuecomment-636421904
    # Synchronize to prevent cross-test failure of print:
    ti.sync()


# TODO: As described by @k-ye above, what we want to ensure
#       is that, the content shows on console is *correct*.
@ti.test()
def test_multi_print():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        print(x, 1234.5, y)

    func(666, 233.3)
    ti.sync()


@ti.test()
def test_print_string():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        # make sure `%` doesn't break vprintf:
        print('hello, world! %s %d %f', 233, y)
        print('cool', x, 'well', y)

    func(666, 233.3)
    ti.sync()


@ti.test()
def test_print_matrix():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print('hello', x[None], 'world!')
        print(y[2] * k, x[None] / k, y[2])

    func(233.3)
    ti.sync()


@ti.test()
def test_print_sep_end():
    @ti.kernel
    def func():
        # hello 42 world!
        print('hello', 42, 'world!')
        # hello 42 Taichi 233 world!
        print('hello', 42, 'Tai', end='')
        print('chi', 233, 'world!')
        # hello42world!
        print('hello', 42, 'world!', sep='')
        # '  ' (with no newline)
        print('  ', end='')
        # 'helloaswd42qwer'
        print('  ', 42, sep='aswd', end='qwer')

    func()
    ti.sync()


@ti.test()
def test_print_multiple_threads():
    x = ti.field(dtype=ti.f32, shape=(128, ))

    @ti.kernel
    def func(k: ti.f32):
        for i in x:
            x[i] = i * k
            print('x[', i, ']=', x[i])

    func(0.1)
    ti.sync()
    func(10.0)
    ti.sync()


@ti.test()
def test_print_list():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=(2, 3))
    y = ti.Vector.field(3, dtype=ti.f32, shape=())

    @ti.kernel
    def func(k: ti.f32):
        w = [k, x.shape]
        print(w + [y.n])  # [233.3, [2, 3], 3]
        print(x.shape)  # [2, 3]
        print(y.shape)  # []
        z = (1, )
        print([1, k**2, k + 1])  # [1, 233.3, 234.3]
        print(z)  # [1]
        print([y[None], z])  # [[0, 0, 0], [1]]
        print([])  # []

    func(233.3)
    ti.sync()


@ti.test(arch=ti.cpu)
def test_python_scope_print_field():
    x = ti.Matrix.field(2, 3, dtype=ti.f32, shape=())
    y = ti.Vector.field(3, dtype=ti.f32, shape=3)
    z = ti.field(dtype=ti.f32, shape=3)

    print(x)
    print(y)
    print(z)
