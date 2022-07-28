'''
To test our new `ti.field` API is functional (#1500)
'''

import pytest
from taichi.lang import impl
from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils

data_types = [ti.i32, ti.f32, ti.i64, ti.f64]
field_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]


@pytest.mark.parametrize('dtype', data_types)
@pytest.mark.parametrize('shape', field_shapes)
@test_utils.test(arch=get_host_arch_list())
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
@test_utils.test(arch=get_host_arch_list())
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
@test_utils.test(arch=get_host_arch_list())
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
@test_utils.test(arch=get_host_arch_list())
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
@test_utils.test(arch=get_host_arch_list())
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


@test_utils.test(arch=get_host_arch_list())
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


@test_utils.test()
def test_field_needs_grad_dtype():
    with pytest.raises(
            RuntimeError,
            match=
            r".* is not supported for field with `needs_grad=True` or `needs_dual=True`."
    ):
        a = ti.field(int, shape=1, needs_grad=True)
    with pytest.raises(
            RuntimeError,
            match=
            r".* is not supported for field with `needs_grad=True` or `needs_dual=True`."
    ):
        b = ti.Vector.field(3, int, shape=1, needs_grad=True)
    with pytest.raises(
            RuntimeError,
            match=
            r".* is not supported for field with `needs_grad=True` or `needs_dual=True`."
    ):
        c = ti.Matrix.field(2, 3, int, shape=1, needs_grad=True)
    with pytest.raises(
            RuntimeError,
            match=
            r".* is not supported for field with `needs_grad=True` or `needs_dual=True`."
    ):
        d = ti.Struct.field(
            {
                "pos": ti.types.vector(3, int),
                "vel": ti.types.vector(3, float),
                "acc": ti.types.vector(3, float),
                "mass": ti.f32,
            },
            shape=1,
            needs_grad=True)


@test_utils.test()
def test_field_needs_dual_dtype():
    with pytest.raises(
            RuntimeError,
            match=
            r".* is not supported for field with `needs_grad=True` or `needs_dual=True`."
    ):
        a = ti.field(int, shape=1, needs_dual=True)
    with pytest.raises(
            RuntimeError,
            match=
            r".* is not supported for field with `needs_grad=True` or `needs_dual=True`."
    ):
        b = ti.Vector.field(3, int, shape=1, needs_dual=True)
    with pytest.raises(
            RuntimeError,
            match=
            r".* is not supported for field with `needs_grad=True` or `needs_dual=True`."
    ):
        c = ti.Matrix.field(2, 3, int, shape=1, needs_dual=True)
    with pytest.raises(
            RuntimeError,
            match=
            r".* is not supported for field with `needs_grad=True` or `needs_dual=True`."
    ):
        d = ti.Struct.field(
            {
                "pos": ti.types.vector(3, int),
                "vel": ti.types.vector(3, float),
                "acc": ti.types.vector(3, float),
                "mass": ti.f32,
            },
            shape=1,
            needs_dual=True)


@pytest.mark.parametrize('dtype', [ti.f32, ti.f64])
def test_default_fp(dtype):
    ti.init(default_fp=dtype)

    x = ti.Vector.field(2, float, ())

    assert x.dtype == impl.get_runtime().default_fp


@pytest.mark.parametrize('dtype', [ti.i32, ti.i64])
def test_default_ip(dtype):
    ti.init(default_ip=dtype)

    x = ti.Vector.field(2, int, ())

    assert x.dtype == impl.get_runtime().default_ip


@test_utils.test()
def test_field_name():
    a = ti.field(dtype=ti.f32, shape=(2, 3), name='a')
    b = ti.Vector.field(3, dtype=ti.f32, shape=(2, 3), name='b')
    c = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(5, 4), name='c')
    assert a._name == 'a'
    assert b._name == 'b'
    assert c._name == 'c'
    assert b.snode._name == 'b'
    d = []
    for i in range(10):
        d.append(ti.field(dtype=ti.f32, shape=(2, 3), name=f'd{i}'))
        assert d[i]._name == f'd{i}'


@test_utils.test()
@pytest.mark.parametrize('shape', field_shapes)
@pytest.mark.parametrize('dtype', [ti.i32, ti.f32])
def test_field_copy_from(shape, dtype):
    x = ti.field(dtype=ti.f32, shape=shape)
    other = ti.field(dtype=dtype, shape=shape)
    other.fill(1)
    x.copy_from(other)
    convert = lambda arr: arr[0] if len(arr) == 1 else arr
    assert (convert(x.shape) == shape)
    assert (x.dtype == ti.f32)
    assert ((x.to_numpy() == 1).all())


@test_utils.test()
def test_field_copy_from_with_mismatch_shape():
    x = ti.field(dtype=ti.f32, shape=(2, 3))
    for other_shape in [(2, ), (2, 2), (2, 3, 4)]:
        other = ti.field(dtype=ti.f16, shape=other_shape)
        with pytest.raises(ValueError):
            x.copy_from(other)


@test_utils.test()
def test_field_copy_from_with_non_filed_object():
    import numpy as np
    x = ti.field(dtype=ti.f32, shape=(2, 3))
    other = np.zeros((2, 3))
    with pytest.raises(TypeError):
        x.copy_from(other)


@test_utils.test()
def test_field_shape_0():
    with pytest.raises(
            ti._lib.core.TaichiRuntimeError,
            match="Every dimension of a Taichi field should be positive"):
        x = ti.field(dtype=ti.f32, shape=0)


@test_utils.test()
def test_index_mismatch():
    with pytest.raises(AssertionError,
                       match="Slicing is not supported on ti.field"):
        val = ti.field(ti.i32, shape=(1, 2, 3))
        val[0, 0] = 1


@test_utils.test()
def test_invalid_slicing():
    with pytest.raises(
            TypeError,
            match=
            "Detected illegal element of type: .*?\. Please be aware that slicing a ti.field is not supported so far."
    ):
        val = ti.field(ti.i32, shape=(2, 2))
        val[0, :]


@test_utils.test(exclude=[ti.cc], debug=True)
def test_field_fill():
    x = ti.field(int, shape=(3, 3))
    x.fill(2)

    y = ti.field(float, shape=(3, 3))
    y.fill(2.0)

    z = ti.Vector.field(3, float, shape=(3, 3))
    z.fill([1, 2, 3])

    @ti.kernel
    def test():
        x.fill(3)
        y.fill(3.0)
        z.fill([4, 5, 6])

    test()


@test_utils.test()
def test_python_for_in():
    x = ti.field(int, shape=3)
    with pytest.raises(NotImplementedError,
                       match="Struct for is only available in Taichi scope"):
        for i in x:
            pass
