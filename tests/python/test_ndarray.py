import copy

import numpy as np
import pytest
from taichi.lang import impl
from taichi.lang.misc import get_host_arch_list
from taichi.lang.util import has_pytorch

import taichi as ti
from tests import test_utils

# properties

data_types = [ti.i32, ti.f32, ti.i64, ti.f64]
ndarray_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]
supported_archs_taichi_ndarray = [ti.cpu, ti.cuda, ti.opengl, ti.vulkan]


def _test_scalar_ndarray(dtype, shape):
    x = ti.ndarray(dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )
    assert x.element_shape == ()

    assert x.dtype == dtype


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=get_host_arch_list(), ndarray_use_torch=True)
def test_scalar_ndarray_torch(dtype, shape):
    _test_scalar_ndarray(dtype, shape)


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalar_ndarray(dtype, shape):
    _test_scalar_ndarray(dtype, shape)


def _test_vector_ndarray(n, dtype, shape):
    x = ti.Vector.ndarray(n, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )
    assert x.element_shape == (n, )

    assert x.dtype == dtype
    assert x.n == n


@pytest.mark.parametrize('n', vector_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=get_host_arch_list(), ndarray_use_torch=True)
def test_vector_ndarray_torch(n, dtype, shape):
    _test_vector_ndarray(n, dtype, shape)


@pytest.mark.parametrize('n', vector_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_vector_ndarray(n, dtype, shape):
    _test_vector_ndarray(n, dtype, shape)


def _test_matrix_ndarray(n, m, dtype, shape):
    x = ti.Matrix.ndarray(n, m, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )
    assert x.element_shape == (n, m)

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@pytest.mark.parametrize('n,m', matrix_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=get_host_arch_list(), ndarray_use_torch=True)
def test_matrix_ndarray_torch(n, m, dtype, shape):
    _test_matrix_ndarray(n, m, dtype, shape)


@pytest.mark.parametrize('n,m', matrix_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_matrix_ndarray(n, m, dtype, shape):
    _test_matrix_ndarray(n, m, dtype, shape)


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
def test_default_fp_ndarray_torch(dtype):
    ti.init(default_fp=dtype, ndarray_use_torch=True)

    x = ti.Vector.ndarray(2, float, ())

    assert x.dtype == impl.get_runtime().default_fp


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
def test_default_fp_ndarray(dtype):
    ti.init(arch=supported_archs_taichi_ndarray, default_fp=dtype)

    x = ti.Vector.ndarray(2, float, ())

    assert x.dtype == impl.get_runtime().default_fp


@pytest.mark.parametrize('dtype', [ti.i32, ti.i64])
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
def test_default_ip_ndarray_torch(dtype):
    ti.init(default_ip=dtype, ndarray_use_torch=True)

    x = ti.Vector.ndarray(2, int, ())

    assert x.dtype == impl.get_runtime().default_ip


@pytest.mark.parametrize('dtype', [ti.i32, ti.i64])
def test_default_ip_ndarray(dtype):
    ti.init(arch=supported_archs_taichi_ndarray, default_ip=dtype)

    x = ti.Vector.ndarray(2, int, ())

    assert x.dtype == impl.get_runtime().default_ip


# access

layouts = [ti.Layout.SOA, ti.Layout.AOS]


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_1d():
    n = 4

    @ti.kernel
    def run(x: ti.any_arr(), y: ti.any_arr()):
        for i in range(n):
            x[i] += i + y[i]

    a = ti.ndarray(ti.i32, shape=(n, ))
    for i in range(n):
        a[i] = i * i
    b = np.ones((n, ), dtype=np.int32)
    run(a, b)
    for i in range(n):
        assert a[i] == i * i + i + 1
    run(b, a)
    for i in range(n):
        assert b[i] == i * i + (i + 1) * 2


def _test_ndarray_2d():
    n = 4
    m = 7

    @ti.kernel
    def run(x: ti.any_arr(), y: ti.any_arr()):
        for i in range(n):
            for j in range(m):
                x[i, j] += i + j + y[i, j]

    a = ti.ndarray(ti.i32, shape=(n, m))
    for i in range(n):
        for j in range(m):
            a[i, j] = i * j
    b = np.ones((n, m), dtype=np.int32)
    run(a, b)
    for i in range(n):
        for j in range(m):
            assert a[i, j] == i * j + i + j + 1
    run(b, a)
    for i in range(n):
        for j in range(m):
            assert b[i, j] == i * j + (i + j + 1) * 2


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_ndarray_2d_torch():
    _test_ndarray_2d()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_2d():
    _test_ndarray_2d()


def _test_ndarray_copy_from_ndarray():
    n = 16
    a = ti.ndarray(ti.i32, shape=n)
    b = ti.ndarray(ti.i32, shape=n)
    a[0] = 1
    a[4] = 2
    b[0] = 4
    b[4] = 5

    a.copy_from(b)

    assert a[0] == 4
    assert a[4] == 5

    x = ti.Vector.ndarray(10, ti.i32, 5, layout=ti.Layout.SOA)
    y = ti.Vector.ndarray(10, ti.i32, 5, layout=ti.Layout.SOA)
    x[1][0] = 1
    x[2][4] = 2
    y[1][0] = 4
    y[2][4] = 5

    x.copy_from(y)

    assert x[1][0] == 4
    assert x[2][4] == 5

    x = ti.Matrix.ndarray(2, 2, ti.i32, 5, layout=ti.Layout.AOS)
    y = ti.Matrix.ndarray(2, 2, ti.i32, 5, layout=ti.Layout.AOS)
    x[0][0, 0] = 1
    x[4][1, 0] = 3
    y[0][0, 0] = 4
    y[4][1, 0] = 6

    x.copy_from(y)

    assert x[0][0, 0] == 4
    assert x[4][1, 0] == 6


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_ndarray_copy_from_ndarray_torch():
    _test_ndarray_copy_from_ndarray()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_copy_from_ndarray():
    _test_ndarray_copy_from_ndarray()


def _test_ndarray_deepcopy():
    n = 16
    x = ti.ndarray(ti.i32, shape=n)
    x[0] = 1
    x[4] = 2

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y[0] == 1
    assert y[4] == 2
    x[0] = 4
    x[4] = 5
    assert y[0] == 1
    assert y[4] == 2

    x = ti.Vector.ndarray(10, ti.i32, 5, layout=ti.Layout.SOA)
    x[1][0] = 4
    x[2][4] = 5

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.n == x.n
    assert y.layout == x.layout
    assert y[1][0] == 4
    assert y[2][4] == 5
    x[1][0] = 1
    x[2][4] = 2
    assert y[1][0] == 4
    assert y[2][4] == 5

    x = ti.Matrix.ndarray(2, 2, ti.i32, 5, layout=ti.Layout.AOS)
    x[0][0, 0] = 7
    x[4][1, 0] = 9

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.m == x.m
    assert y.n == x.n
    assert y.layout == x.layout
    assert y[0][0, 0] == 7
    assert y[4][1, 0] == 9
    x[0][0, 0] = 3
    x[4][1, 0] = 5
    assert y[0][0, 0] == 7
    assert y[4][1, 0] == 9


def test_ndarray_cuda_caching_allocator():
    ti.init(arch=ti.cuda, ndarray_use_cached_allocator=True)
    n = 8
    a = ti.ndarray(ti.i32, shape=(n))
    a.fill(2)
    a = 1
    b = ti.ndarray(ti.i32, shape=(n))
    b.fill(2)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_fill():
    n = 8
    a = ti.ndarray(ti.i32, shape=(n))
    anp = np.ones((n, ), dtype=np.int32)
    a.fill(2)
    anp.fill(2)
    assert (a.to_numpy() == anp).all()

    b = ti.Vector.ndarray(4, ti.f32, shape=(n))
    bnp = np.ones(shape=b.arr.shape, dtype=np.float32)
    b.fill(2.5)
    bnp.fill(2.5)
    assert (b.to_numpy() == bnp).all()

    c = ti.Matrix.ndarray(4, 4, ti.f32, shape=(n))
    cnp = np.ones(shape=c.arr.shape, dtype=np.float32)
    c.fill(1.5)
    cnp.fill(1.5)
    assert (c.to_numpy() == cnp).all()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_rw_cache():
    a = ti.Vector.ndarray(3, ti.f32, ())
    b = ti.Vector.ndarray(3, ti.f32, 12)

    n = 1000
    for i in range(n):
        c_a = copy.deepcopy(a)
        c_b = copy.deepcopy(b)
        c_a[None] = c_b[10]


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_ndarray_deepcopy_torch():
    _test_ndarray_deepcopy()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_deepcopy():
    _test_ndarray_deepcopy()


def _test_ndarray_numpy_io():
    n = 7
    m = 4
    a = ti.ndarray(ti.i32, shape=(n, m))
    a.fill(2)
    b = ti.ndarray(ti.i32, shape=(n, m))
    b.from_numpy(np.ones((n, m), dtype=np.int32) * 2)
    assert (a.to_numpy() == b.to_numpy()).all()

    d = 2
    p = 4
    x = ti.Vector.ndarray(d, ti.f32, p)
    x.fill(2)
    y = ti.Vector.ndarray(d, ti.f32, p)
    y.from_numpy(np.ones((p, d), dtype=np.int32) * 2)
    assert (x.to_numpy() == y.to_numpy()).all()

    c = 2
    d = 2
    p = 4
    x = ti.Matrix.ndarray(c, d, ti.f32, p)
    x.fill(2)
    y = ti.Matrix.ndarray(c, d, ti.f32, p)
    y.from_numpy(np.ones((p, c, d), dtype=np.int32) * 2)
    assert (x.to_numpy() == y.to_numpy()).all()


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=supported_archs_taichi_ndarray, ndarray_use_torch=True)
def test_ndarray_numpy_io_torch():
    _test_ndarray_numpy_io()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_numpy_io():
    _test_ndarray_numpy_io()


def _test_matrix_ndarray_python_scope(layout):
    a = ti.Matrix.ndarray(2, 2, ti.i32, 5, layout=layout)
    for i in range(5):
        for j, k in ti.ndrange(2, 2):
            a[i][j, k] = j * j + k * k
    assert a[0][0, 0] == 0
    assert a[1][0, 1] == 1
    assert a[2][1, 0] == 1
    assert a[3][1, 1] == 2
    assert a[4][0, 1] == 1


@pytest.mark.parametrize('layout', layouts)
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=supported_archs_taichi_ndarray, ndarray_use_torch=True)
def test_matrix_ndarray_python_scope_torch(layout):
    _test_matrix_ndarray_python_scope(layout)


@pytest.mark.parametrize('layout', layouts)
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_python_scope(layout):
    _test_matrix_ndarray_python_scope(layout)


def _test_matrix_ndarray_taichi_scope(layout):
    @ti.kernel
    def func(a: ti.any_arr()):
        for i in range(5):
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = ti.Matrix.ndarray(2, 2, ti.i32, 5, layout=layout)
    func(m)
    assert m[0][0, 0] == 0
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1


@pytest.mark.parametrize('layout', layouts)
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_matrix_ndarray_taichi_scope_torch(layout):
    _test_matrix_ndarray_taichi_scope(layout)


@pytest.mark.parametrize('layout', layouts)
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_taichi_scope(layout):
    _test_matrix_ndarray_taichi_scope(layout)


def _test_matrix_ndarray_taichi_scope_struct_for(layout):
    @ti.kernel
    def func(a: ti.any_arr()):
        for i in a:
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = ti.Matrix.ndarray(2, 2, ti.i32, 5, layout=layout)
    func(m)
    assert m[0][0, 0] == 0
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1


@pytest.mark.parametrize('layout', layouts)
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_matrix_ndarray_taichi_scope_struct_for_torch(layout):
    _test_matrix_ndarray_taichi_scope_struct_for(layout)


@pytest.mark.parametrize('layout', layouts)
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_taichi_scope_struct_for(layout):
    _test_matrix_ndarray_taichi_scope_struct_for(layout)


@pytest.mark.parametrize('layout', layouts)
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_vector_ndarray_python_scope_torch(layout):
    a = ti.Vector.ndarray(10, ti.i32, 5, layout=layout)
    for i in range(5):
        for j in range(4):
            a[i][j * j] = j * j
    assert a[0][6] == 0  # torch memory initialized to zero
    assert a[1][0] == 0
    assert a[2][1] == 1
    assert a[3][4] == 4
    assert a[4][9] == 9


@pytest.mark.parametrize('layout', layouts)
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_vector_ndarray_python_scope(layout):
    a = ti.Vector.ndarray(10, ti.i32, 5, layout=layout)
    for i in range(5):
        for j in range(4):
            a[i][j * j] = j * j
    assert a[0][9] == 9
    assert a[1][0] == 0
    assert a[2][1] == 1
    assert a[3][4] == 4
    assert a[4][9] == 9


@pytest.mark.parametrize('layout', layouts)
@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_vector_ndarray_taichi_scope_torch(layout):
    @ti.kernel
    def func(a: ti.any_arr()):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5, layout=layout)
    func(v)
    assert v[0][6] == 0  # torch memory initialized to zero
    assert v[1][0] == 0
    assert v[2][1] == 1
    assert v[3][4] == 4
    assert v[4][9] == 9


@pytest.mark.parametrize('layout', layouts)
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_vector_ndarray_taichi_scope(layout):
    @ti.kernel
    def func(a: ti.any_arr()):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5, layout=layout)
    func(v)
    assert v[0][9] == 9
    assert v[1][0] == 0
    assert v[2][1] == 1
    assert v[3][4] == 4
    assert v[4][9] == 9


# number of compiled functions


def _test_compiled_functions():
    @ti.kernel
    def func(a: ti.any_arr(element_dim=1)):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 1
    v = np.zeros((6, 10), dtype=np.int32)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 1
    import torch
    v = torch.zeros((6, 11), dtype=torch.int32)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 2
    v = ti.Vector.ndarray(10, ti.i32, 5, layout=ti.Layout.SOA)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 3


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_compiled_functions_torch():
    _test_compiled_functions()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_compiled_functions():
    _test_compiled_functions()


# annotation compatibility


def _test_arg_not_match():
    @ti.kernel
    def func1(a: ti.any_arr(element_dim=1)):
        pass

    x = ti.Matrix.ndarray(2, 3, ti.i32, shape=(4, 7))
    with pytest.raises(
            ValueError,
            match=
            r'Invalid argument into ti\.any_arr\(\) - required element_dim=1, but .* is provided'
    ):
        func1(x)

    @ti.kernel
    def func2(a: ti.any_arr(element_dim=2)):
        pass

    x = ti.Vector.ndarray(2, ti.i32, shape=(4, 7))
    with pytest.raises(
            ValueError,
            match=
            r'Invalid argument into ti\.any_arr\(\) - required element_dim=2, but .* is provided'
    ):
        func2(x)

    @ti.kernel
    def func3(a: ti.any_arr(layout=ti.Layout.AOS)):
        pass

    x = ti.Matrix.ndarray(2, 3, ti.i32, shape=(4, 7), layout=ti.Layout.SOA)
    with pytest.raises(
            ValueError,
            match=
            r'Invalid argument into ti\.any_arr\(\) - required layout=Layout\.AOS, but .* is provided'
    ):
        func3(x)

    @ti.kernel
    def func4(a: ti.any_arr(layout=ti.Layout.SOA)):
        pass

    x = ti.Vector.ndarray(2, ti.i32, shape=(4, 7))
    with pytest.raises(
            ValueError,
            match=
            r'Invalid argument into ti\.any_arr\(\) - required layout=Layout\.SOA, but .* is provided'
    ):
        func4(x)

    @ti.kernel
    def func5(a: ti.any_arr(element_shape=(2, 3))):
        pass

    x = ti.Vector.ndarray(2, ti.i32, shape=(4, 7))
    with pytest.raises(
            ValueError,
            match=
            r'Invalid argument into ti\.any_arr\(\) - required element_dim'):
        func5(x)

    with pytest.raises(
            ValueError,
            match=r'Both element_shape and element_dim are specified'):

        @ti.kernel
        def func6(a: ti.any_arr(element_dim=1, element_shape=(2, 3))):
            pass

    @ti.kernel
    def func7(a: ti.any_arr(field_dim=2)):
        pass

    x = ti.ndarray(ti.i32, shape=(3, ))
    with pytest.raises(
            ValueError,
            match=r'Invalid argument into ti\.any_arr\(\) - required field_dim'
    ):
        func7(x)


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=get_host_arch_list(), ndarray_use_torch=True)
def test_arg_not_match_torch():
    _test_arg_not_match()


@test_utils.test(arch=get_host_arch_list())
def test_arg_not_match():
    _test_arg_not_match()


def _test_size_in_bytes():
    a = ti.ndarray(ti.i32, 8)
    assert a.get_element_size() == 4
    assert a.get_nelement() == 8

    b = ti.Vector.ndarray(10, ti.f64, 5)
    assert b.get_element_size() == 8
    assert b.get_nelement() == 50


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@test_utils.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=True)
def test_size_in_bytes_torch():
    _test_size_in_bytes()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_size_in_bytes():
    _test_size_in_bytes()


@test_utils.test(arch=ti.vulkan, ndarray_use_torch=True)
def test_torch_based_ndarray_not_supported():
    x = ti.ndarray(ti.f32, shape=(4, 2))

    @ti.kernel
    def init(x: ti.any_arr()):
        for i, j in x:
            x[i, j] = i + j

    with pytest.raises(
            AssertionError,
            match=
            r'Torch-based ndarray is only supported on taichi x64/arm64/cuda backend'
    ):
        init(x)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_different_shape():
    n1 = 4
    x = ti.ndarray(dtype=ti.f32, shape=(n1, n1))

    @ti.kernel
    def init(d: ti.i32, arr: ti.any_arr()):
        for i, j in arr:
            arr[i, j] = d

    init(2, x)
    assert (x.to_numpy() == (np.ones(shape=(n1, n1)) * 2)).all()
    n2 = 8
    y = ti.ndarray(dtype=ti.f32, shape=(n2, n2))
    init(3, y)
    assert (y.to_numpy() == (np.ones(shape=(n2, n2)) * 3)).all()
