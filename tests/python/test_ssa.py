'''
SSA violation edge-case regression test.
1. Ensure working well when computation result is assigned to self.
2. Prevent duplicate-evaluation on expression with side-effect like random.
'''
import math

import numpy as np

import taichi as ti
from taichi import approx


@ti.all_archs
def test_matrix_self_assign():
    a = ti.Vector.field(2, ti.f32, ())
    b = ti.Matrix.field(2, 2, ti.f32, ())
    c = ti.Vector.field(2, ti.f32, ())

    @ti.kernel
    def func():
        a[None] = a[None].normalized()
        b[None] = b[None].transpose()
        c[None] = ti.Vector([c[None][1], c[None][0]])

    inv_sqrt2 = 1 / math.sqrt(2)

    a[None] = [1, 1]
    b[None] = [[1, 2], [3, 4]]
    c[None] = [2, 3]
    func()
    assert a[None].value == ti.Vector([inv_sqrt2, inv_sqrt2])
    assert b[None].value == ti.Matrix([[1, 3], [2, 4]])
    assert c[None].value == ti.Vector([3, 2])


@ti.all_archs
def test_random_vector_dup_eval():
    a = ti.Vector.field(2, ti.f32, ())

    @ti.kernel
    def func():
        a[None] = ti.Vector([ti.random(), 1]).normalized()

    for i in range(4):
        func()
        assert a[None].value.norm_sqr() == approx(1)


@ti.all_archs
def test_func_argument_dup_eval():
    @ti.func
    def func(a, t):
        return a * t - a

    @ti.kernel
    def kern(t: ti.f32) -> ti.f32:
        return func(ti.random(), t)

    for i in range(4):
        assert kern(1.0) == 0.0


@ti.all_archs
def test_func_random_argument_dup_eval():
    @ti.func
    def func(a):
        return ti.Vector([ti.cos(a), ti.sin(a)])

    @ti.kernel
    def kern() -> ti.f32:
        return func(ti.random()).norm_sqr()

    for i in range(4):
        assert kern() == approx(1.0)
