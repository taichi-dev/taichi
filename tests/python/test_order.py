import pytest
from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_order_scalar():
    X = 16
    Y = 8
    Z = 4
    S = 4

    a = ti.field(ti.i32, shape=(X, Y, Z), order='ijk')
    b = ti.field(ti.i32, shape=(X, Y, Z), order='ikj')
    c = ti.field(ti.i32, shape=(X, Y, Z), order='jik')
    d = ti.field(ti.i32, shape=(X, Y, Z), order='jki')
    e = ti.field(ti.i32, shape=(X, Y, Z), order='kij')
    f = ti.field(ti.i32, shape=(X, Y, Z), order='kji')

    @ti.kernel
    def fill():
        for i, j, k in b:
            a[i, j, k] = i * j * k
            b[i, j, k] = i * j * k
            c[i, j, k] = i * j * k
            d[i, j, k] = i * j * k
            e[i, j, k] = i * j * k
            f[i, j, k] = i * j * k

    @ti.kernel
    def get_field_addr(a: ti.template(), i: ti.i32, j: ti.i32,
                       k: ti.i32) -> ti.u64:
        return ti.get_addr(a, [i, j, k])

    fill()

    a_addr = get_field_addr(a, 0, 0, 0)
    b_addr = get_field_addr(b, 0, 0, 0)
    c_addr = get_field_addr(c, 0, 0, 0)
    d_addr = get_field_addr(d, 0, 0, 0)
    e_addr = get_field_addr(e, 0, 0, 0)
    f_addr = get_field_addr(f, 0, 0, 0)
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                assert a[i, j, k] == b[i, j, k] == c[i, j, k] == i * j * k
                assert d[i, j, k] == e[i, j, k] == f[i, j, k] == i * j * k
                assert a_addr + (i * (Y * Z) + j * Z + k) * S == \
                       get_field_addr(a, i, j, k)
                assert b_addr + (i * (Z * Y) + k * Y + j) * S == \
                       get_field_addr(b, i, j, k)
                assert c_addr + (j * (X * Z) + i * Z + k) * S == \
                       get_field_addr(c, i, j, k)
                assert d_addr + (j * (Z * X) + k * X + i) * S == \
                       get_field_addr(d, i, j, k)
                assert e_addr + (k * (X * Y) + i * Y + j) * S == \
                       get_field_addr(e, i, j, k)
                assert f_addr + (k * (Y * X) + j * X + i) * S == \
                       get_field_addr(f, i, j, k)


@test_utils.test(arch=get_host_arch_list())
def test_order_vector():
    X = 4
    Y = 2
    Z = 2
    S = 4

    a = ti.Vector.field(Z,
                        ti.i32,
                        shape=(X, Y),
                        order='ij',
                        layout=ti.Layout.AOS)
    b = ti.Vector.field(Z,
                        ti.i32,
                        shape=(X, Y),
                        order='ji',
                        layout=ti.Layout.AOS)
    c = ti.Vector.field(Z,
                        ti.i32,
                        shape=(X, Y),
                        order='ij',
                        layout=ti.Layout.SOA)
    d = ti.Vector.field(Z,
                        ti.i32,
                        shape=(X, Y),
                        order='ji',
                        layout=ti.Layout.SOA)

    @ti.kernel
    def fill():
        for i, j in b:
            a[i, j] = [i, j]
            b[i, j] = [i, j]
            c[i, j] = [i, j]
            d[i, j] = [i, j]

    @ti.kernel
    def get_field_addr(a: ti.template(), i: ti.i32, j: ti.i32) -> ti.u64:
        return ti.get_addr(a, [i, j])

    fill()

    a_addr = get_field_addr(a, 0, 0)
    b_addr = get_field_addr(b, 0, 0)
    c_addr = get_field_addr(c, 0, 0)
    d_addr = get_field_addr(d, 0, 0)
    for i in range(X):
        for j in range(Y):
            assert a[i, j] == b[i, j] == c[i, j] == d[i, j] == [i, j]
            for k in range(Z):
                assert a_addr + (i * (Y * Z) + j * Z + k) * S == \
                       get_field_addr(a.get_scalar_field(k), i, j)
                assert b_addr + (j * (X * Z) + i * Z + k) * S == \
                       get_field_addr(b.get_scalar_field(k), i, j)
                assert c_addr + (k * (X * Y) + i * Y + j) * S == \
                       get_field_addr(c.get_scalar_field(k), i, j)
                assert d_addr + (k * (Y * X) + j * X + i) * S == \
                       get_field_addr(d.get_scalar_field(k), i, j)


@test_utils.test(arch=get_host_arch_list())
def test_order_must_throw_scalar():
    with pytest.raises(
            ti.TaichiCompilationError,
            match='The dimensionality of shape and order must be the same'):
        a = ti.field(dtype=ti.f32, shape=3, order='ij')
    with pytest.raises(ti.TaichiCompilationError,
                       match='shape cannot be None when order is set'):
        b = ti.field(dtype=ti.f32, shape=None, order='i')
    with pytest.raises(ti.TaichiCompilationError,
                       match='The axes in order must be different'):
        c = ti.field(dtype=ti.f32, shape=(3, 4, 3), order='iji')
    with pytest.raises(ti.TaichiCompilationError, match='Invalid axis'):
        d = ti.field(dtype=ti.f32, shape=(3, 4, 3), order='ijl')


@test_utils.test(arch=get_host_arch_list())
def test_order_must_throw_vector():
    with pytest.raises(
            ti.TaichiCompilationError,
            match='The dimensionality of shape and order must be the same'):
        a = ti.Vector.field(3, dtype=ti.f32, shape=3, order='ij')
    with pytest.raises(ti.TaichiCompilationError,
                       match='shape cannot be None when order is set'):
        b = ti.Vector.field(3, dtype=ti.f32, shape=None, order='i')
    with pytest.raises(ti.TaichiCompilationError,
                       match='The axes in order must be different'):
        c = ti.Vector.field(3, dtype=ti.f32, shape=(3, 4, 3), order='iii')
    with pytest.raises(ti.TaichiCompilationError, match='Invalid axis'):
        d = ti.Vector.field(3, dtype=ti.f32, shape=(3, 4, 3), order='ihj')
