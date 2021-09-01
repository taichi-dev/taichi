import numpy as np
import pytest

import taichi as ti

# properties

data_types = [ti.i32, ti.f32, ti.i64, ti.f64]
ndarray_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=ti.get_host_arch_list())
def test_scalar_ndarray(dtype, shape):
    x = ti.ndarray(dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )

    assert x.dtype == dtype


@pytest.mark.parametrize('n', vector_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=ti.get_host_arch_list())
def test_vector_ndarray(n, dtype, shape):
    x = ti.Vector.ndarray(n, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )

    assert x.dtype == dtype
    assert x.n == n


@pytest.mark.parametrize('n,m', matrix_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', ndarray_shapes)
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=ti.get_host_arch_list())
def test_matrix_ndarray(n, m, dtype, shape):
    x = ti.Matrix.ndarray(n, m, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
def test_default_fp_ndarray(dtype):
    ti.init(default_fp=dtype)

    x = ti.Vector.ndarray(2, float, ())

    assert x.dtype == ti.get_runtime().default_fp


@pytest.mark.parametrize('dtype', [ti.i32, ti.i64])
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
def test_default_ip_ndarray(dtype):
    ti.init(default_ip=dtype)

    x = ti.Vector.ndarray(2, int, ())

    assert x.dtype == ti.get_runtime().default_ip


# access

layouts = [ti.Layout.SOA, ti.Layout.AOS]


@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_ndarray_2d():
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


@pytest.mark.parametrize('layout', layouts)
@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_matrix_ndarray_python_scope(layout):
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
def test_matrix_ndarray_taichi_scope(layout):
    @ti.kernel
    def func(a: ti.any_arr(element_shape=(2, 2), layout=layout)):
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
def test_matrix_ndarray_taichi_scope_struct_for(layout):
    @ti.kernel
    def func(a: ti.any_arr(element_shape=(2, 2), layout=layout)):
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
def test_vector_ndarray_python_scope(layout):
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
def test_vector_ndarray_taichi_scope(layout):
    @ti.kernel
    def func(a: ti.any_arr(element_shape=(10, ), layout=layout)):
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


# number of compiled functions


@pytest.mark.skipif(not ti.has_pytorch(), reason='Pytorch not installed.')
@ti.test(exclude=ti.opengl)
def test_compiled_functions():
    @ti.kernel
    def func(a: ti.any_arr(element_shape=(10, ))):
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
    v = torch.zeros((7, 10), dtype=torch.int32)
    func(v)
    assert ti.get_runtime().get_num_compiled_functions() == 1
