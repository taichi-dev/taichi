# TODO: make test_element_wise slim (#1055)
import taichi as ti
from taichi import approx
from random import random, randint, seed
import operator as ops
import math


def _c_mod(a, b):
    return a - b * int(float(a) / b)


def rand(dtype):
    if ti.core.is_integral(dtype):
        return randint(1, 5)
    else:
        # Prevent integer operands in pow and floordiv in GLSL
        # Discussion: https://github.com/taichi-dev/taichi/pull/943#discussion_r423177941
        return float(randint(1, 5)) / 5 - 0.01


@ti.host_arch_only
def _test_matrix_element_wise_unary(dtype, n, m, ti_func, math_func):
    a = ti.Matrix(n, m, dt=dtype, shape=())
    u = ti.Matrix(n, m, dt=dtype, shape=())

    w = []
    for i in range(n * m):
        w.append(rand(dtype))

    for i in range(n):
        for j in range(m):
            u[None][i, j] = w[i * m + j]

    @ti.kernel
    def func():
        a[None] = ti_func(u[None])

    func()

    for i in range(n):
        for j in range(m):
            expected = math_func(w[i * m + j])
            assert a[None][i, j] == approx(expected)


@ti.host_arch_only
def _test_matrix_element_wise_binary(dtype, n, m, ti_func, math_func):
    a = ti.Matrix(n, m, dt=dtype, shape=())
    b = ti.Matrix(n, m, dt=dtype, shape=())
    c = ti.Matrix(n, m, dt=dtype, shape=())
    u1 = ti.Matrix(n, m, dt=dtype, shape=())
    u2 = ti.Matrix(n, m, dt=dtype, shape=())
    u3 = ti.var(dt=dtype, shape=())

    w1 = []
    w2 = []
    for i in range(n * m):
        w1.append(rand(dtype))
        w2.append(rand(dtype))

    w3 = rand(dtype)

    for i in range(n):
        for j in range(m):
            u1[None][i, j] = w1[i * m + j]
            u2[None][i, j] = w2[i * m + j]

    u3[None] = w3

    @ti.kernel
    def func():
        a[None] = ti_func(u1[None], u2[None])
        b[None] = ti_func(u1[None], u3[None])
        c[None] = ti_func(u3[None], u1[None])

    func()

    for i in range(n):
        for j in range(m):
            expected = math_func(w1[i * m + j], w2[i * m + j])
            assert a[None][i, j] == approx(expected)
            expected = math_func(w1[i * m + j], w3)
            assert b[None][i, j] == approx(expected)
            expected = math_func(w3, w1[i * m + j])
            assert c[None][i, j] == approx(expected)


@ti.host_arch_only
def _test_matrix_element_wise_writeback_binary(dtype,
                                               n,
                                               m,
                                               ti_func,
                                               math_func,
                                               is_atomic=True):
    a = ti.Matrix(n, m, dt=dtype, shape=())
    b = ti.Matrix(n, m, dt=dtype, shape=())
    u1 = ti.Matrix(n, m, dt=dtype, shape=())
    u2 = ti.Matrix(n, m, dt=dtype, shape=())
    u3 = ti.var(dt=dtype, shape=())

    w1 = []
    w2 = []
    for i in range(n * m):
        w1.append(rand(dtype))
        w2.append(rand(dtype))

    w3 = rand(dtype)

    for i in range(n):
        for j in range(m):
            u1[None][i, j] = w1[i * m + j]
            u2[None][i, j] = w2[i * m + j]

    u3[None] = w3

    @ti.kernel
    def func():
        a[None] = ti_func(u1[None], u2[None])
        # Suggested by @k-ye, we forbid `matrix = scalar`, and use `matrix.fill(scalar)` instead for filling a matrix with a same scalar in Taichi-scope.
        # Discussion: https://github.com/taichi-dev/taichi/pull/1062#pullrequestreview-421364614
        if ti.static(is_atomic):
            b[None] = ti_func(u2[None], u3[None])
        else:
            b[None] = u2[None].fill(u3[None])

    func()

    for i in range(n):
        for j in range(m):
            expected = math_func(w1[i * m + j], w2[i * m + j])
            assert u1[None][i, j] == approx(expected)

            # ti.atomic_* returns the orignal value of lhs,
            # while ti.assign returns the value-after-assign of lhs.
            # Discussion: https://github.com/taichi-dev/taichi/pull/1062#issuecomment-635090596
            if is_atomic:
                expected = w1[i * m + j]
            assert a[None][i, j] == approx(expected)

            expected = math_func(w2[i * m + j], w3)
            assert u2[None][i, j] == approx(expected)

            if is_atomic:
                expected = w2[i * m + j]
            assert b[None][i, j] == approx(expected)


def test_matrix_element_wise_unary_infix():
    seed(5156)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_unary(ti.i32, n, m, ti.logical_not, ops.not_)
        _test_matrix_element_wise_unary(ti.i32, n, m, ops.invert, ops.invert)
        _test_matrix_element_wise_unary(ti.i32, n, m, ops.neg, ops.neg)
        _test_matrix_element_wise_unary(ti.f32, n, m, ops.neg, ops.neg)


def test_matrix_element_wise_binary_infix_f32_1():
    seed(4399)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.add, ops.add)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.sub, ops.sub)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.mul, ops.mul)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.mod, ops.mod)


def test_matrix_element_wise_binary_infix_f32_2():
    seed(4401)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.truediv,
                                         ops.truediv)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.floordiv,
                                         ops.floordiv)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.pow, ops.pow)


def test_matrix_element_wise_binary_infix_f32_cmp():
    seed(4400)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.lt, ops.lt)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.le, ops.le)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.gt, ops.gt)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.ge, ops.ge)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.eq, ops.eq)
        _test_matrix_element_wise_binary(ti.f32, n, m, ops.ne, ops.ne)


def test_matrix_element_wise_binary_infix_i32():
    seed(6174)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.add, ops.add)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.sub, ops.sub)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.mul, ops.mul)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.mod, ops.mod)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.pow, ops.pow)
        # TODO: add pow(f32, i32) test


def test_matrix_element_wise_binary_infix_i32_cmp():
    seed(6175)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.lt, ops.lt)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.le, ops.le)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.gt, ops.gt)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.ge, ops.ge)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.eq, ops.eq)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.ne, ops.ne)


def test_matrix_element_wise_binary_f32():
    seed(666)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_binary(ti.f32, n, m, ti.atan2, math.atan2)
        _test_matrix_element_wise_binary(ti.f32, n, m, ti.min, min)
        _test_matrix_element_wise_binary(ti.f32, n, m, ti.max, max)
        _test_matrix_element_wise_binary(ti.f32, n, m, ti.pow, pow)


def test_matrix_element_wise_binary_i32():
    seed(985)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_binary(ti.i32, n, m, ti.min, min)
        _test_matrix_element_wise_binary(ti.i32, n, m, ti.max, max)
        _test_matrix_element_wise_binary(ti.i32, n, m, ti.pow, pow)
        _test_matrix_element_wise_binary(ti.i32, n, m, ti.raw_mod, _c_mod)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.or_, ops.or_)
        _test_matrix_element_wise_binary(ti.i32, n, m, ops.and_, ops.and_)
        # TODO: add ti.raw_div


def test_matrix_element_wise_unary_1():
    seed(233)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.sin, math.sin)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.cos, math.cos)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.tan, math.tan)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.asin, math.asin)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.acos, math.acos)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.tanh, math.tanh)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.sqrt, math.sqrt)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.exp, math.exp)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.log, math.log)


def test_matrix_element_wise_unary_2():
    seed(211)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.ceil, math.ceil)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.floor, math.floor)
        _test_matrix_element_wise_unary(ti.f32, n, m, ti.abs, abs)
        _test_matrix_element_wise_unary(ti.i32, n, m, ti.abs, abs)
        # TODO(archibate): why ti.inv, ti.sqr doesn't work?


def test_matrix_element_wise_writeback_binary_i32():
    seed(986)
    for n, m in [(5, 4), (3, 1)]:
        # We support `matrix += scalar` syntax in Taichi-scope.
        # Discussion: https://github.com/taichi-dev/taichi/pull/1062#pullrequestreview-419402587
        _test_matrix_element_wise_writeback_binary(ti.i32, n, m, ti.atomic_add,
                                                   ops.add)
        _test_matrix_element_wise_writeback_binary(ti.i32, n, m, ti.atomic_sub,
                                                   ops.sub)
        _test_matrix_element_wise_writeback_binary(ti.i32, n, m, ti.atomic_and,
                                                   ops.and_)
        _test_matrix_element_wise_writeback_binary(ti.i32, n, m, ti.atomic_xor,
                                                   ops.xor)
        _test_matrix_element_wise_writeback_binary(ti.i32, n, m, ti.atomic_or,
                                                   ops.or_)
        _test_matrix_element_wise_writeback_binary(ti.i32, n, m, ti.atomic_max,
                                                   max)
        _test_matrix_element_wise_writeback_binary(ti.i32, n, m, ti.atomic_min,
                                                   min)
        # We also support `matrix = scalar` syntax in Taichi-scope.
        # Discussion: https://github.com/taichi-dev/taichi/pull/1062#issuecomment-635089253
        _test_matrix_element_wise_writeback_binary(ti.i32, n, m, ti.assign,
                                                   lambda x, y: y, False)


def test_matrix_element_wise_writeback_binary_f32():
    seed(987)
    for n, m in [(5, 4), (3, 1)]:
        _test_matrix_element_wise_writeback_binary(ti.f32, n, m, ti.atomic_add,
                                                   ops.add)
        _test_matrix_element_wise_writeback_binary(ti.f32, n, m, ti.atomic_sub,
                                                   ops.sub)
        _test_matrix_element_wise_writeback_binary(ti.f32, n, m, ti.atomic_max,
                                                   max)
        _test_matrix_element_wise_writeback_binary(ti.f32, n, m, ti.atomic_min,
                                                   min)
        _test_matrix_element_wise_writeback_binary(ti.f32, n, m, ti.assign,
                                                   lambda x, y: y, False)
