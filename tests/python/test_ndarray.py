import numpy as np
import pytest

import taichi as ti

# properties

data_types = [ti.i32, ti.f32, ti.i64, ti.f64]
ndarray_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]


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
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=ti.get_host_arch_list())
def test_scalar_ndarray_torch(dtype, shape):
    _test_scalar_ndarray(dtype, shape)


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@ti.test(arch=ti.get_host_arch_list(), ndarray_use_torch=False)
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
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=ti.get_host_arch_list())
def test_vector_ndarray_torch(n, dtype, shape):
    _test_vector_ndarray(n, dtype, shape)


@pytest.mark.parametrize('n', vector_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@ti.test(arch=ti.get_host_arch_list(), ndarray_use_torch=False)
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
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=ti.get_host_arch_list())
def test_matrix_ndarray_torch(n, m, dtype, shape):
    _test_matrix_ndarray(n, m, dtype, shape)


@pytest.mark.parametrize('n,m', matrix_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@ti.test(arch=ti.get_host_arch_list(), ndarray_use_torch=False)
def test_matrix_ndarray(n, m, dtype, shape):
    _test_matrix_ndarray(n, m, dtype, shape)


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
def test_default_fp_ndarray_torch(dtype):
    ti.init(default_fp=dtype)

    x = ti.Vector.ndarray(2, float, ())

    assert x.dtype == ti.get_runtime().default_fp


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
def test_default_fp_ndarray(dtype):
    ti.init(arch=[ti.cpu, ti.cuda], default_fp=dtype, ndarray_use_torch=False)

    x = ti.Vector.ndarray(2, float, ())

    assert x.dtype == ti.get_runtime().default_fp


@pytest.mark.parametrize('dtype', [ti.i32, ti.i64])
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
def test_default_ip_ndarray_torch(dtype):
    ti.init(default_ip=dtype)

    x = ti.Vector.ndarray(2, int, ())

    assert x.dtype == ti.get_runtime().default_ip


@pytest.mark.parametrize('dtype', [ti.i32, ti.i64])
def test_default_ip_ndarray(dtype):
    ti.init(arch=[ti.cpu, ti.cuda], default_ip=dtype, ndarray_use_torch=False)

    x = ti.Vector.ndarray(2, int, ())

    assert x.dtype == ti.get_runtime().default_ip


# access

layouts = [ti.Layout.SOA, ti.Layout.AOS]


@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
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


@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_ndarray_2d_torch():
    _test_ndarray_2d()


@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
def test_ndarray_2d():
    _test_ndarray_2d()


def _test_ndarray_numpy_io():
    n = 7
    m = 4
    a = ti.ndarray(ti.i32, shape=(n, m))
    a.fill(2)
    b = ti.ndarray(ti.i32, shape=(n, m))
    b.from_numpy(np.ones((n, m), dtype=np.int32) * 2)
    assert (a.to_numpy() == b.to_numpy()).all()


@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_ndarray_numpy_io_torch():
    _test_ndarray_numpy_io()


@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
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
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_matrix_ndarray_python_scope_torch(layout):
    _test_matrix_ndarray_python_scope(layout)


@pytest.mark.parametrize('layout', layouts)
@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
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
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_matrix_ndarray_taichi_scope_torch(layout):
    _test_matrix_ndarray_taichi_scope(layout)


@pytest.mark.parametrize('layout', layouts)
@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
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
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_matrix_ndarray_taichi_scope_struct_for_torch(layout):
    _test_matrix_ndarray_taichi_scope_struct_for(layout)


@pytest.mark.parametrize('layout', layouts)
@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
def test_matrix_ndarray_taichi_scope_struct_for(layout):
    _test_matrix_ndarray_taichi_scope_struct_for(layout)


def _test_vector_ndarray_python_scope(layout):
    a = ti.Vector.ndarray(10, ti.i32, 5, layout=layout)
    for i in range(5):
        for j in range(4):
            a[i][j * j] = j * j
    assert a[0][6] == 0
    assert a[1][0] == 0
    assert a[2][1] == 1
    assert a[3][4] == 4
    assert a[4][9] == 9


@pytest.mark.parametrize('layout', layouts)
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_vector_ndarray_python_scope_torch(layout):
    _test_vector_ndarray_python_scope(layout)


@pytest.mark.parametrize('layout', layouts)
@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
def test_vector_ndarray_python_scope(layout):
    _test_vector_ndarray_python_scope(layout)


def _test_vector_ndarray_taichi_scope(layout):
    @ti.kernel
    def func(a: ti.any_arr()):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5, layout=layout)
    func(v)
    assert v[0][6] == 0
    assert v[1][0] == 0
    assert v[2][1] == 1
    assert v[3][4] == 4
    assert v[4][9] == 9


@pytest.mark.parametrize('layout', layouts)
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_vector_ndarray_taichi_scope_torch(layout):
    _test_vector_ndarray_taichi_scope(layout)


@pytest.mark.parametrize('layout', layouts)
@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
def test_vector_ndarray_taichi_scope(layout):
    _test_vector_ndarray_taichi_scope(layout)


# number of compiled functions


def _test_compiled_functions():
    @ti.kernel
    def func(a: ti.any_arr(element_dim=1)):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5)
    func(v)
    assert ti.get_runtime().get_num_compiled_functions() == 1
    v = np.zeros((6, 10), dtype=np.int32)
    func(v)
    assert ti.get_runtime().get_num_compiled_functions() == 1
    import torch
    v = torch.zeros((6, 11), dtype=torch.int32)
    func(v)
    assert ti.get_runtime().get_num_compiled_functions() == 2
    v = ti.Vector.ndarray(10, ti.i32, 5, layout=ti.Layout.SOA)
    func(v)
    assert ti.get_runtime().get_num_compiled_functions() == 3


@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_compiled_functions_torch():
    _test_compiled_functions()


@ti.test(arch=[ti.cpu, ti.cuda], ndarray_use_torch=False)
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


@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=ti.get_host_arch_list())
def test_arg_not_match_torch():
    _test_arg_not_match()


@ti.test(arch=ti.get_host_arch_list(), ndarray_use_torch=False)
def test_arg_not_match():
    _test_arg_not_match()
