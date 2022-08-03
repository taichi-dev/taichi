import pytest
from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_accessor():
    a = ti.field(dtype=ti.i32)

    ti.root.dense(ti.ijk, 128).place(a, offset=(1024, 2048, 2100))

    a[1029, 2100, 2200] = 1
    assert a[1029, 2100, 2200] == 1


@test_utils.test()
def test_struct_for_huge_offsets():
    a = ti.field(dtype=ti.i32)

    offset = 1024, 2048, 2100, 2200
    ti.root.dense(ti.ijkl, 4).place(a, offset=offset)

    @ti.kernel
    def test():
        for i, j, k, l in a:
            a[i, j, k, l] = i + j * 10 + k * 100 + l * 1000

    test()

    for i in range(offset[0], offset[0] + 4):
        for j in range(offset[1], offset[1] + 4):
            for k in range(offset[2], offset[2] + 4):
                for l in range(offset[3], offset[3] + 4):
                    assert a[i, j, k, l] == i + j * 10 + k * 100 + l * 1000


@test_utils.test()
def test_struct_for_negative():
    a = ti.field(dtype=ti.i32)

    offset = 16, -16
    ti.root.dense(ti.ij, 32).place(a, offset=offset)

    @ti.kernel
    def test():
        for i, j in a:
            a[i, j] = i + j * 10

    test()

    for i in range(16, 48):
        for j in range(-16, 16):
            assert a[i, j] == i + j * 10


@test_utils.test()
def test_offset_for_var():
    a = ti.field(dtype=ti.i32, shape=16, offset=-48)
    b = ti.field(dtype=ti.i32, shape=(16, ), offset=(16, ))
    c = ti.field(dtype=ti.i32, shape=(16, 64), offset=(-16, -64))
    d = ti.field(dtype=ti.i32, shape=(16, 64), offset=None)

    offset = 4, -4
    shape = 16, 16
    e = ti.field(dtype=ti.i32, shape=shape, offset=offset)

    @ti.kernel
    def test():
        for i, j in e:
            e[i, j] = i * j

    test()
    for i in range(4, 20):
        for j in range(-4, 12):
            assert e[i, j] == i * j


@test_utils.test()
def test_offset_for_vector():
    a = ti.field(dtype=ti.i32, shape=16, offset=-48)
    b = ti.field(dtype=ti.i32, shape=16, offset=None)

    offset = 16
    shape = 16
    c = ti.Vector.field(n=1, dtype=ti.i32, shape=shape, offset=offset)

    @ti.kernel
    def test():
        for i in c:
            c[i][0] = 2 * i

    test()
    for i in range(offset, offset + shape, 1):
        assert c[i][0] == 2 * i


@test_utils.test()
def test_offset_for_matrix():
    a = ti.Matrix.field(3,
                        3,
                        shape=(16, 16),
                        offset=(-16, 16),
                        dtype=ti.float32)

    @ti.kernel
    def test():
        for i, j in a:
            for m in range(3):
                a[i, j][0, 0] = i + j

    test()

    for i in range(-16, 0):
        for j in range(16, 32):
            assert a[i, j][0, 0] == i + j


@test_utils.test(arch=get_host_arch_list())
def test_offset_must_throw_scalar():
    with pytest.raises(
            ti.TaichiCompilationError,
            match='The dimensionality of shape and offset must be the same'):
        a = ti.field(dtype=ti.f32, shape=3, offset=(3, 4))
    with pytest.raises(ti.TaichiCompilationError,
                       match='shape cannot be None when offset is set'):
        b = ti.field(dtype=ti.f32, shape=None, offset=(3, 4))


@test_utils.test(arch=get_host_arch_list())
def test_offset_must_throw_vector():
    with pytest.raises(
            ti.TaichiCompilationError,
            match='The dimensionality of shape and offset must be the same'):
        a = ti.Vector.field(3, dtype=ti.f32, shape=3, offset=(3, 4))
    with pytest.raises(ti.TaichiCompilationError,
                       match='shape cannot be None when offset is set'):
        b = ti.Vector.field(3, dtype=ti.f32, shape=None, offset=(3, ))


@test_utils.test(arch=get_host_arch_list())
def test_offset_must_throw_matrix():
    with pytest.raises(
            ti.TaichiCompilationError,
            match='The dimensionality of shape and offset must be the same'):
        a = ti.Matrix.field(3,
                            3,
                            dtype=ti.i32,
                            shape=(32, 16, 8),
                            offset=(32, 16))
    with pytest.raises(ti.TaichiCompilationError,
                       match='shape cannot be None when offset is set'):
        b = ti.Matrix.field(3, 3, dtype=ti.i32, shape=None, offset=(32, 16))
