import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(debug=True)
def test_kernel_keyword_args():
    @ti.kernel
    def foo(a: ti.i32, b: ti.i32):
        assert a == 1
        assert b == 2

    foo(1, b=2)


@test_utils.test(debug=True)
def test_kernel_keyword_args_missing():
    @ti.kernel
    def foo(a: ti.i32, b: ti.i32):
        assert a == 1
        assert b == 2

    with pytest.raises(ti.TaichiSyntaxError, match="Parameter 'a' missing"):
        foo(b=2)


@test_utils.test(debug=True)
def test_kernel_keyword_args_not_found():
    @ti.kernel
    def foo(a: ti.i32, b: ti.i32):
        assert a == 1
        assert b == 2

    with pytest.raises(ti.TaichiSyntaxError, match="Unexpected argument 'c'"):
        foo(1, 2, c=2)


@test_utils.test(debug=True)
def test_kernel_too_many():
    @ti.kernel
    def foo(a: ti.i32, b: ti.i32):
        assert a == 1
        assert b == 2

    with pytest.raises(ti.TaichiSyntaxError, match="Too many arguments"):
        foo(1, 2, 3)


@test_utils.test(debug=True)
def test_function_keyword_args():
    @ti.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @ti.func
    def bar(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 4

    @ti.kernel
    def baz():
        foo(1, b=2)
        bar(b=2, a=1, c=4)

    baz()


@test_utils.test(debug=True)
def test_function_keyword_args_missing():
    @ti.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @ti.kernel
    def missing():
        foo(1, c=3)

    with pytest.raises(ti.TaichiSyntaxError, match="Parameter 'b' missing"):
        missing()


@test_utils.test(debug=True)
def test_function_keyword_args_not_found():
    @ti.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @ti.kernel
    def not_found():
        foo(1, 2, 3, d=3)

    with pytest.raises(ti.TaichiSyntaxError, match="Unexpected argument 'd'"):
        not_found()


@test_utils.test(debug=True)
def test_function_too_many():
    @ti.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @ti.kernel
    def many():
        foo(1, 2, 3, 4)

    with pytest.raises(ti.TaichiSyntaxError, match="Too many arguments"):
        many()


@test_utils.test(debug=True)
def test_function_keyword_args_duplicate():
    @ti.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @ti.kernel
    def duplicate():
        foo(1, a=3, b=3)

    with pytest.raises(ti.TaichiSyntaxError, match="Multiple values for argument 'a'"):
        duplicate()


@test_utils.test()
def test_args_with_many_ndarrays():
    particle_num = 0
    cluster_num = 0
    permu_num = 0

    particlePosition = ti.Vector.ndarray(3, ti.f32, shape=10)
    outClusterPosition = ti.Vector.ndarray(3, ti.f32, shape=10)
    outClusterOffsets = ti.ndarray(ti.i32, shape=10)
    outClusterSizes = ti.ndarray(ti.i32, shape=10)
    outClusterIndices = ti.ndarray(ti.i32, shape=10)

    particle_pos = ti.Vector.ndarray(3, ti.f32, shape=20)
    particle_prev_pos = ti.Vector.ndarray(3, ti.f32, shape=20)
    particle_rest_pos = ti.Vector.ndarray(3, ti.f32, shape=20)
    particle_index = ti.ndarray(ti.i32, shape=20)

    cluster_rest_mass_center = ti.Vector.ndarray(3, ti.f32, shape=20)
    cluster_begin = ti.ndarray(ti.i32, shape=20)

    @ti.kernel
    def ti_import_cluster_data(
        center: ti.types.vector(3, ti.f32),
        particle_num: int,
        cluster_num: int,
        permu_num: int,
        particlePosition: ti.types.ndarray(ndim=1),
        outClusterPosition: ti.types.ndarray(ndim=1),
        outClusterOffsets: ti.types.ndarray(ndim=1),
        outClusterSizes: ti.types.ndarray(ndim=1),
        outClusterIndices: ti.types.ndarray(ndim=1),
        particle_pos: ti.types.ndarray(ndim=1),
        particle_prev_pos: ti.types.ndarray(ndim=1),
        particle_rest_pos: ti.types.ndarray(ndim=1),
        cluster_rest_mass_center: ti.types.ndarray(ndim=1),
        cluster_begin: ti.types.ndarray(ndim=1),
        particle_index: ti.types.ndarray(ndim=1),
    ):
        added_permu_num = outClusterIndices.shape[0]

        for i in range(added_permu_num):
            particle_index[i] = 1.0

    center = ti.math.vec3(0, 0, 0)
    ti_import_cluster_data(
        center,
        particle_num,
        cluster_num,
        permu_num,
        particlePosition,
        outClusterPosition,
        outClusterOffsets,
        outClusterSizes,
        outClusterIndices,
        particle_pos,
        particle_prev_pos,
        particle_rest_pos,
        cluster_rest_mass_center,
        cluster_begin,
        particle_index,
    )


@test_utils.test()
def test_struct_arg():
    s0 = ti.types.struct(a=ti.i16, b=ti.f32)
    s1 = ti.types.struct(a=ti.f32, b=s0)

    @ti.kernel
    def foo(a: s1) -> ti.f32:
        return a.a + a.b.a + a.b.b

    ret = foo(s1(a=1, b=s0(a=65537, b=123)))
    assert ret == pytest.approx(125)


@test_utils.test()
def test_struct_arg_with_matrix():
    mat = ti.types.matrix(3, 2, ti.f32)
    s0 = ti.types.struct(a=mat, b=ti.f32)
    s1 = ti.types.struct(a=ti.i32, b=s0)

    @ti.kernel
    def foo(a: s1) -> ti.i32:
        ret = a.a + a.b.b
        for i in range(3):
            for j in range(2):
                ret += a.b.a[i, j] * (i + 1) * (j + 2)
        return ret

    arg = s1(a=1, b=s0(a=mat(1, 2, 3, 4, 5, 6), b=123))
    ret_std = 1 + 123

    for i in range(3):
        for j in range(2):
            ret_std += (i + 1) * (j + 2) * (i * 2 + j + 1)

    ret = foo(arg)
    assert ret == ret_std
