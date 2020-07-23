'''
To test our new `ti.field` API is functional (#1500)
'''

import taichi as ti
import pytest

# TODO(archibate): add 64-bit dtypes after #1462
data_types = [ti.i32, ti.f32]
field_shapes = [(), 8, (8, ), (6, 12)]
vector_dims = [2, 3]
matrix_dims = [(1, 2), (3, 1), (4, 3), (2, 2)]


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', field_shapes)
@ti.host_arch_only
def test_scalar_field(dtype, shape):
    x = ti.field(dtype=dtype, shape=shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )

    assert x.dtype == dtype


@pytest.mark.parametrize('n', vector_dims)
@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', field_shapes)
@ti.host_arch_only
def test_vector_field(n, dtype, shape):
    x = ti.Vector.field(n, dtype=dtype, shape=shape)

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
@ti.host_arch_only
def test_matrix_field(n, m, dtype, shape):
    x = ti.Matrix.field(n, m, dtype=dtype, shape=shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape, )

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@ti.host_arch_only
def test_field_needs_grad():
    # Just make sure the usage doesn't crash, see https://github.com/taichi-dev/taichi/pull/1545
    n = 8
    m1 = ti.field(ti.f32, n, needs_grad=True)
    m2 = ti.field(ti.f32, n, needs_grad=True)
    gr = ti.field(ti.f32, n)

    @ti.kernel
    def func():
        for i in range(n):
            gr[i] = m1.grad[i] + m2.grad[i]

    func()
