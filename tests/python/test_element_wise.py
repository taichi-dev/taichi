import taichi as ti
from taichi import approx
from random import random, randint
import math


@ti.host_arch_only
def _test_matrix_element_wise_unary(ti_func, math_func):
    n, m = randint(1, 8), randint(1, 8)

    a = ti.Matrix(n, m, dt=ti.f32, shape=())
    u = ti.Matrix(n, m, dt=ti.f32, shape=())

    w = []
    for i in range(n * m):
        w.append(random())

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
def _test_matrix_element_wise_binary(ti_func, math_func):
    n, m = randint(1, 8), randint(1, 8)

    a = ti.Matrix(n, m, dt=ti.f32, shape=())
    b = ti.Matrix(n, m, dt=ti.f32, shape=())
    u1 = ti.Matrix(n, m, dt=ti.f32, shape=())
    u2 = ti.Matrix(n, m, dt=ti.f32, shape=())
    u3 = ti.var(dt=ti.f32, shape=())

    w1 = []
    w2 = []
    for i in range(n * m):
        w1.append(random())
        w2.append(random())

    w3 = random()

    for i in range(n):
        for j in range(m):
            u1[None][i, j] = w1[i * m + j]
            u2[None][i, j] = w2[i * m + j]

    u3[None] = w3

    @ti.kernel
    def func():
        a[None] = ti_func(u1[None], u2[None])
        b[None] = ti_func(u1[None], u3[None])

    func()

    for i in range(n):
        for j in range(m):
            expected = math_func(w1[i * m + j], w2[i * m + j])
            assert a[None][i, j] == approx(expected)
            expected = math_func(w1[i * m + j], w3)
            assert b[None][i, j] == approx(expected)


def test_matrix_element_wise_binary():
    _test_matrix_element_wise_binary(ti.atan2, math.atan2)
    _test_matrix_element_wise_binary(ti.min, min)
    _test_matrix_element_wise_binary(ti.max, max)
    # TODO(archibate): revert ti.pow
    #_test_matrix_element_wise_binary(ti.pow, pow)



def test_matrix_element_wise_unary():
    _test_matrix_element_wise_unary(ti.sin, math.sin)
    _test_matrix_element_wise_unary(ti.cos, math.cos)
    _test_matrix_element_wise_unary(ti.tan, math.tan)
    _test_matrix_element_wise_unary(ti.asin, math.asin)
    _test_matrix_element_wise_unary(ti.acos, math.acos)
    _test_matrix_element_wise_unary(ti.tanh, math.tanh)
    _test_matrix_element_wise_unary(ti.sqrt, math.sqrt)
    _test_matrix_element_wise_unary(ti.exp, math.exp)
    _test_matrix_element_wise_unary(ti.log, math.log)
    # ASK(yuanming-hu): why we don't have taichi_core.expr_ceil?
    #_test_matrix_element_wise_unary(ti.ceil, math.ceil)
    _test_matrix_element_wise_unary(ti.floor, math.floor)
    _test_matrix_element_wise_unary(ti.abs, abs)
