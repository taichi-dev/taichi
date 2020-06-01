import taichi as ti
import pytest


# Not really testable..
# Just making sure it does not crash
# Metal doesn't support ti.print() or 64-bit data
# So does OpenGL..
@pytest.mark.parametrize('dt', [ti.i32, ti.f32, ti.i64, ti.f64])
@ti.archs_excluding(ti.metal, ti.opengl)
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
@ti.archs_excluding(ti.metal, ti.opengl)
def test_multi_print():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        print(x, 1234.5, y)

    func(666, 233.3)
    ti.sync()


@ti.archs_excluding(ti.metal, ti.opengl)
def test_print_string():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        # make sure `%` doesn't break vprintf:
        print('hello, world! %s %d %f', 233, y)
        print('cool', x, 'well', y)

    func(666, 233.3)
    ti.sync()


@ti.archs_excluding(ti.metal, ti.opengl)
def test_print_matrix():
    x = ti.Matrix(2, 3, dt=ti.f32, shape=())
    y = ti.Vector(3, dt=ti.f32, shape=3)

    @ti.kernel
    def func(k: ti.f32):
        print('hello', x[None], 'world!')
        print(y[2] * k, x[None] / k, y[2])

    func(233.3)
    ti.sync()
