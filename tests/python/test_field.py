'''
To test our new `ti.field` API is functional (#1500)
'''

import pytest

import taichi as ti

data_types = [ti.i32, ti.f32, ti.i64, ti.f64]
field_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', field_shapes)
@ti.test(arch=ti.get_host_arch_list())
def test_scalar_field(dtype, shape):
    x = ti.field(dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )

    assert x.dtype == dtype


@pytest.mark.parametrize('n', vector_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', field_shapes)
@ti.test(arch=ti.get_host_arch_list())
def test_vector_field(n, dtype, shape):
    x = ti.Vector.field(n, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == 1


@pytest.mark.parametrize('n,m', matrix_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', field_shapes)
@ti.test(arch=ti.get_host_arch_list())
def test_matrix_field(n, m, dtype, shape):
    x = ti.Matrix.field(n, m, dtype=dtype, shape=shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', field_shapes)
@ti.test(arch=ti.get_host_arch_list())
def test_scalr_field_from_numpy(dtype, shape):
    import numpy as np
    x = ti.field(dtype, shape)
    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        ti.i32: np.int32,
        ti.f32: np.float32,
        ti.f64: np.float64,
        ti.i64: np.int64,
    }
    arr = np.empty(shape, dtype=numpy_dtypes[dtype])
    x.from_numpy(arr)


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', field_shapes)
@ti.test(arch=ti.get_host_arch_list())
def test_scalr_field_from_numpy_with_mismatch_shape(dtype, shape):
    import numpy as np
    x = ti.field(dtype, shape)
    numpy_dtypes = {
        ti.i32: np.int32,
        ti.f32: np.float32,
        ti.f64: np.float64,
        ti.i64: np.int64,
    }
    # compose the mismatch shape for every ti.field.
    # set the shape to (2, 3) by default, if the ti.field shape is a tuple, set it to 1.
    mismatch_shape = (2, 3)
    if isinstance(shape, tuple):
        mismatch_shape = 1
    arr = np.empty(mismatch_shape, dtype=numpy_dtypes[dtype])
    with pytest.raises(ValueError):
        x.from_numpy(arr)


@ti.test(arch=ti.get_host_arch_list())
def test_field_needs_grad():
    # Just make sure the usage doesn't crash, see #1545
    n = 8
    m1 = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
    m2 = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
    gr = ti.field(dtype=ti.f32, shape=n)

    @ti.kernel
    def func():
        for i in range(n):
            gr[i] = m1.grad[i] + m2.grad[i]

    func()


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
def test_default_fp(dtype):
    ti.init(default_fp=dtype)

    x = ti.Vector.field(2, float, ())

    assert x.dtype == ti.get_runtime().default_fp


@pytest.mark.parametrize('dtype', [ti.i32, ti.i64])
def test_default_ip(dtype):
    ti.init(default_ip=dtype)

    x = ti.Vector.field(2, int, ())

    assert x.dtype == ti.get_runtime().default_ip


@ti.test()
def test_field_name():
    a = ti.field(dtype=ti.f32, shape=(2, 3), name='a')
    b = ti.Vector.field(3, dtype=ti.f32, shape=(2, 3), name='b')
    c = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(5, 4), name='c')
    assert a.name == 'a'
    assert b.name == 'b'
    assert c.name == 'c'
    assert b.snode.name == 'b'
    d = []
    for i in range(10):
        d.append(ti.field(dtype=ti.f32, shape=(2, 3), name=f'd{i}'))
        assert d[i].name == f'd{i}'
