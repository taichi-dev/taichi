import copy

import numpy as np
import pytest
from taichi.lang import impl
from taichi.lang.exception import TaichiIndexError
from taichi.lang.misc import get_host_arch_list
from taichi.lang.util import has_pytorch

import taichi as ti
from tests import test_utils

if has_pytorch():
    import torch

# properties

data_types = [ti.i32, ti.f32, ti.i64, ti.f64]
ndarray_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]
supported_archs_taichi_ndarray = [
    ti.cpu,
    ti.cuda,
    ti.opengl,
    ti.vulkan,
    ti.metal,
    ti.amdgpu,
]


def _test_scalar_ndarray(dtype, shape):
    x = ti.ndarray(dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == ()

    assert x.dtype == dtype


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalar_ndarray(dtype, shape):
    _test_scalar_ndarray(dtype, shape)


def _test_vector_ndarray(n, dtype, shape):
    x = ti.Vector.ndarray(n, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == (n,)

    assert x.dtype == dtype
    assert x.n == n


@pytest.mark.parametrize("n", vector_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_vector_ndarray(n, dtype, shape):
    _test_vector_ndarray(n, dtype, shape)


def _test_matrix_ndarray(n, m, dtype, shape):
    x = ti.Matrix.ndarray(n, m, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == (n, m)

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@pytest.mark.parametrize("n,m", matrix_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_matrix_ndarray(n, m, dtype, shape):
    _test_matrix_ndarray(n, m, dtype, shape)


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_default_fp_ndarray(dtype):
    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch, default_fp=dtype)

    x = ti.Vector.ndarray(2, float, ())

    assert x.dtype == impl.get_runtime().default_fp


@pytest.mark.parametrize("dtype", [ti.i32, ti.i64])
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_default_ip_ndarray(dtype):
    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch, default_ip=dtype)

    x = ti.Vector.ndarray(2, int, ())

    assert x.dtype == impl.get_runtime().default_ip


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_1d():
    n = 4

    @ti.kernel
    def run(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in range(n):
            x[i] += i + y[i]

    a = ti.ndarray(ti.i32, shape=(n,))
    for i in range(n):
        a[i] = i * i
    b = np.ones((n,), dtype=np.int32)
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
    def run(x: ti.types.ndarray(), y: ti.types.ndarray()):
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


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_2d():
    _test_ndarray_2d()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_compound_element():
    n = 10
    a = ti.ndarray(ti.i32, shape=(n,))

    vec3 = ti.types.vector(3, ti.i32)
    b = ti.ndarray(vec3, shape=(n, n))
    assert isinstance(b, ti.VectorNdarray)
    assert b.shape == (n, n)
    assert b.element_type.element_type() == ti.i32
    assert b.element_type.shape() == (3,)

    matrix34 = ti.types.matrix(3, 4, float)
    c = ti.ndarray(matrix34, shape=(n, n + 1))
    assert isinstance(c, ti.MatrixNdarray)
    assert c.shape == (n, n + 1)
    assert c.element_type.element_type() == ti.f32
    assert c.element_type.shape() == (3, 4)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_copy_from_ndarray():
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

    x = ti.Vector.ndarray(10, ti.i32, 5)
    y = ti.Vector.ndarray(10, ti.i32, 5)
    x[1][0] = 1
    x[2][4] = 2
    y[1][0] = 4
    y[2][4] = 5

    x.copy_from(y)

    assert x[1][0] == 4
    assert x[2][4] == 5

    x = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    y = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    x[0][0, 0] = 1
    x[4][1, 0] = 3
    y[0][0, 0] = 4
    y[4][1, 0] = 6

    x.copy_from(y)

    assert x[0][0, 0] == 4
    assert x[4][1, 0] == 6


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_deepcopy():
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

    x = ti.Vector.ndarray(10, ti.i32, 5)
    x[1][0] = 4
    x[2][4] = 5

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.n == x.n
    assert y[1][0] == 4
    assert y[2][4] == 5
    x[1][0] = 1
    x[2][4] = 2
    assert y[1][0] == 4
    assert y[2][4] == 5

    x = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    x[0][0, 0] = 7
    x[4][1, 0] = 9

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.m == x.m
    assert y.n == x.n
    assert y[0][0, 0] == 7
    assert y[4][1, 0] == 9
    x[0][0, 0] = 3
    x[4][1, 0] = 5
    assert y[0][0, 0] == 7
    assert y[4][1, 0] == 9


@test_utils.test(arch=[ti.cuda])
def test_ndarray_caching_allocator():
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
    anp = np.ones((n,), dtype=np.int32)
    a.fill(2)
    anp.fill(2)
    assert (a.to_numpy() == anp).all()

    b = ti.Vector.ndarray(4, ti.f32, shape=(n))
    bnp = np.ones(shape=b.arr.total_shape(), dtype=np.float32)
    b.fill(2.5)
    bnp.fill(2.5)
    assert (b.to_numpy() == bnp).all()

    c = ti.Matrix.ndarray(4, 4, ti.f32, shape=(n))
    cnp = np.ones(shape=c.arr.total_shape(), dtype=np.float32)
    c.fill(1.5)
    cnp.fill(1.5)
    assert (c.to_numpy() == cnp).all()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_rw_cache():
    a = ti.Vector.ndarray(3, ti.f32, ())
    b = ti.Vector.ndarray(3, ti.f32, 12)

    n = 100
    for i in range(n):
        c_a = copy.deepcopy(a)
        c_b = copy.deepcopy(b)
        c_a[None] = c_b[10]


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


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_numpy_io():
    _test_ndarray_numpy_io()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_matrix_numpy_io():
    n = 5
    m = 2

    x = ti.Vector.ndarray(n, ti.i32, (m,))
    x_np = 1 + np.arange(n * m).reshape(m, n).astype(np.int32)
    x.from_numpy(x_np)
    assert (x_np.flatten() == x.to_numpy().flatten()).all()

    k = 2
    x = ti.Matrix.ndarray(m, k, ti.i32, n)
    x_np = 1 + np.arange(m * k * n).reshape(n, m, k).astype(np.int32)
    x.from_numpy(x_np)
    assert (x_np.flatten() == x.to_numpy().flatten()).all()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_python_scope():
    a = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    for i in range(5):
        for j, k in ti.ndrange(2, 2):
            a[i][j, k] = j * j + k * k
    assert a[0][0, 0] == 0
    assert a[1][0, 1] == 1
    assert a[2][1, 0] == 1
    assert a[3][1, 1] == 2
    assert a[4][0, 1] == 1


def _test_matrix_ndarray_taichi_scope():
    @ti.kernel
    def func(a: ti.types.ndarray()):
        for i in range(5):
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    func(m)
    assert m[0][0, 0] == 0
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_taichi_scope():
    _test_matrix_ndarray_taichi_scope()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_matrix_ndarray_taichi_scope_real_matrix():
    _test_matrix_ndarray_taichi_scope()


def _test_matrix_ndarray_taichi_scope_struct_for():
    @ti.kernel
    def func(a: ti.types.ndarray()):
        for i in a:
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    func(m)
    assert m[0][0, 0] == 0
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_taichi_scope_struct_for():
    _test_matrix_ndarray_taichi_scope_struct_for()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_matrix_ndarray_taichi_scope_struct_for_real_matrix():
    _test_matrix_ndarray_taichi_scope_struct_for()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_vector_ndarray_python_scope():
    a = ti.Vector.ndarray(10, ti.i32, 5)
    for i in range(5):
        for j in range(4):
            a[i][j * j] = j * j
    assert a[0][9] == 9
    assert a[1][0] == 0
    assert a[2][1] == 1
    assert a[3][4] == 4
    assert a[4][9] == 9


def _test_vector_ndarray_taichi_scope():
    @ti.kernel
    def func(a: ti.types.ndarray()):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5)
    func(v)
    assert v[0][9] == 9
    assert v[1][0] == 0
    assert v[2][1] == 1
    assert v[3][4] == 4
    assert v[4][9] == 9


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_vector_ndarray_taichi_scope():
    _test_vector_ndarray_taichi_scope()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_vector_ndarray_taichi_scope_real_matrix():
    _test_vector_ndarray_taichi_scope()


# number of compiled functions
def _test_compiled_functions():
    @ti.kernel
    def func(a: ti.types.ndarray(ti.types.vector(n=10, dtype=ti.i32))):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 1
    v = np.zeros((6, 10), dtype=np.int32)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 1


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_compiled_functions():
    _test_compiled_functions()


# annotation compatibility


def _test_arg_not_match():
    @ti.kernel
    def func1(a: ti.types.ndarray(dtype=ti.types.vector(2, ti.i32))):
        pass

    x = ti.Matrix.ndarray(2, 3, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid argument into ti\.types\.ndarray\(\) - required element_dim=1, but .* is provided",
    ):
        func1(x)

    @ti.kernel
    def func2(a: ti.types.ndarray(dtype=ti.types.matrix(2, 2, ti.i32))):
        pass

    x = ti.Vector.ndarray(2, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid argument into ti\.types\.ndarray\(\) - required element_dim=2, but .* is provided",
    ):
        func2(x)

    @ti.kernel
    def func5(a: ti.types.ndarray(dtype=ti.types.matrix(2, 3, dtype=ti.i32))):
        pass

    x = ti.Vector.ndarray(2, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid argument into ti\.types\.ndarray\(\) - required element_dim",
    ):
        func5(x)

    @ti.kernel
    def func7(a: ti.types.ndarray(ndim=2)):
        pass

    x = ti.ndarray(ti.i32, shape=(3,))
    with pytest.raises(
        ValueError,
        match=r"Invalid argument into ti\.types\.ndarray\(\) - required ndim",
    ):
        func7(x)

    @ti.kernel
    def func8(x: ti.types.ndarray(dtype=ti.f32)):
        pass

    x = ti.ndarray(dtype=ti.i32, shape=(16, 16))
    with pytest.raises(TypeError, match=r"Expect element type .* for Ndarray, but get .*"):
        func8(x)


@test_utils.test(arch=get_host_arch_list())
def test_arg_not_match():
    _test_arg_not_match()


def _test_size_in_bytes():
    a = ti.ndarray(ti.i32, 8)
    assert a._get_element_size() == 4
    assert a._get_nelement() == 8

    b = ti.Vector.ndarray(10, ti.f64, 5)
    assert b._get_element_size() == 80
    assert b._get_nelement() == 5


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_size_in_bytes():
    _test_size_in_bytes()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_different_shape():
    n1 = 4
    x = ti.ndarray(dtype=ti.f32, shape=(n1, n1))

    @ti.kernel
    def init(d: ti.i32, arr: ti.types.ndarray()):
        for i, j in arr:
            arr[i, j] = d

    init(2, x)
    assert (x.to_numpy() == (np.ones(shape=(n1, n1)) * 2)).all()
    n2 = 8
    y = ti.ndarray(dtype=ti.f32, shape=(n2, n2))
    init(3, y)
    assert (y.to_numpy() == (np.ones(shape=(n2, n2)) * 3)).all()


def _test_ndarray_grouped():
    @ti.kernel
    def func(a: ti.types.ndarray()):
        for i in ti.grouped(a):
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j

    a1 = ti.Matrix.ndarray(2, 2, ti.i32, shape=5)
    func(a1)
    for i in range(5):
        for j in range(2):
            for k in range(2):
                assert a1[i][j, k] == j * j

    a2 = ti.Matrix.ndarray(2, 2, ti.i32, shape=(3, 3))
    func(a2)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                for p in range(2):
                    assert a2[i, j][k, p] == k * k


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_grouped():
    _test_ndarray_grouped()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_ndarray_grouped_real_matrix():
    _test_ndarray_grouped()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_as_template():
    @ti.kernel
    def func(arr_src: ti.template(), arr_dst: ti.template()):
        for i, j in ti.ndrange(*arr_src.shape):
            arr_dst[i, j] = arr_src[i, j]

    arr_0 = ti.ndarray(ti.f32, shape=(5, 10))
    arr_1 = ti.ndarray(ti.f32, shape=(5, 10))
    with pytest.raises(ti.TaichiRuntimeTypeError, match=r"Ndarray shouldn't be passed in via"):
        func(arr_0, arr_1)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_gaussian_kernel():
    M_PI = 3.14159265358979323846

    @ti.func
    def gaussian(x, sigma):
        return ti.exp(-0.5 * ti.pow(x / sigma, 2)) / (sigma * ti.sqrt(2.0 * M_PI))

    @ti.kernel
    def fill_gaussian_kernel(ker: ti.types.ndarray(ti.f32, ndim=1), N: ti.i32):
        sum = 0.0
        for i in range(2 * N + 1):
            ker[i] = gaussian(i - N, ti.sqrt(N))
            sum += ker[i]
        for i in range(2 * N + 1):
            ker[i] = ker[i] / sum

    N = 4
    arr = ti.ndarray(dtype=ti.f32, shape=(20))
    fill_gaussian_kernel(arr, N)
    res = arr.to_numpy()

    np_arr = np.zeros(20, dtype=np.float32)
    fill_gaussian_kernel(np_arr, N)

    assert test_utils.allclose(res, np_arr)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_numpy_matrix():
    boundary_box_np = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    boundary_box = ti.Vector.ndarray(3, ti.f32, shape=2)
    boundary_box.from_numpy(boundary_box_np)
    ref_numpy = boundary_box.to_numpy()

    assert (boundary_box_np == ref_numpy).all()


@pytest.mark.parametrize("dtype", [ti.i64, ti.u64, ti.f64])
@test_utils.test(arch=supported_archs_taichi_ndarray, require=ti.extension.data64)
def test_ndarray_python_scope_read_64bit(dtype):
    @ti.kernel
    def run(x: ti.types.ndarray()):
        for i in x:
            x[i] = i + ti.i64(2**40)

    n = 4
    a = ti.ndarray(dtype, shape=(n,))
    run(a)
    for i in range(n):
        assert a[i] == i + 2**40


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_init_as_zero():
    a = ti.ndarray(dtype=ti.f32, shape=(6, 10))
    v = np.zeros((6, 10), dtype=np.float32)
    assert test_utils.allclose(a.to_numpy(), v)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_reset():
    n = 8
    c = ti.Matrix.ndarray(4, 4, ti.f32, shape=(n))
    del c
    d = ti.Matrix.ndarray(4, 4, ti.f32, shape=(n))
    ti.reset()


@pytest.mark.run_in_serial
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_in_python_func():
    def test():
        z = ti.ndarray(float, (8192, 8192))

    for i in range(300):
        test()


@test_utils.test(arch=[ti.cpu, ti.cuda], exclude=[ti.amdgpu])
def test_ndarray_with_fp16():
    half2 = ti.types.vector(n=2, dtype=ti.f16)

    @ti.kernel
    def init(x: ti.types.ndarray(dtype=half2, ndim=1)):
        for i in x:
            x[i] = half2(2.0)

    @ti.kernel
    def test(table: ti.types.ndarray(dtype=half2, ndim=1)):
        tmp = ti.Vector([ti.f16(0.0), ti.f16(0.0)])
        for i in ti.static(range(2)):
            tmp = tmp + 4.0 * table[i]

        table[0] = tmp

    acc = ti.ndarray(dtype=half2, shape=(40))
    table = ti.ndarray(dtype=half2, shape=(40))

    init(table)
    test(table)

    assert (table.to_numpy()[0] == 16.0).all()


@test_utils.test(
    arch=supported_archs_taichi_ndarray,
    require=ti.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_scalar_ndarray_oob():
    @ti.kernel
    def access_arr(input: ti.types.ndarray(), x: ti.i32) -> ti.f32:
        return input[x]

    input = np.random.randn(4)

    # Works
    access_arr(input, 1)

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 4)

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, -1)


# SOA layout for ndarray is deprecated so no need to test
@test_utils.test(
    arch=supported_archs_taichi_ndarray,
    require=ti.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_matrix_ndarray_oob():
    @ti.kernel
    def access_arr(input: ti.types.ndarray(), p: ti.i32, q: ti.i32, x: ti.i32, y: ti.i32) -> ti.f32:
        return input[p, q][x, y]

    input = ti.ndarray(dtype=ti.math.mat2, shape=(4, 5))

    # Works
    access_arr(input, 2, 3, 0, 1)

    # element_shape
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 2, 3, 2, 1)
    # field_shape[0]
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 4, 4, 0, 1)
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, -3, 4, 1, 1)
    # field_shape[1]
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 3, 5, 0, 1)
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 2, -10, 1, 1)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_mismatched_index_python_scope():
    x = ti.ndarray(dtype=ti.f32, shape=(4, 4))
    with pytest.raises(TaichiIndexError, match=r"2d ndarray indexed with 1d indices"):
        x[0]

    with pytest.raises(TaichiIndexError, match=r"2d ndarray indexed with 3d indices"):
        x[0, 0, 0]


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_0dim_ndarray_read_write_python_scope():
    x = ti.ndarray(dtype=ti.f32, shape=())

    x[()] = 1.0
    assert x[None] == 1.0

    y = ti.ndarray(dtype=ti.math.vec2, shape=())
    y[()] = [1.0, 2.0]
    assert y[None] == [1.0, 2.0]


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_0dim_ndarray_read_write_taichi_scope():
    x = ti.ndarray(dtype=ti.f32, shape=())

    @ti.kernel
    def write(x: ti.types.ndarray()):
        a = x[()] + 1
        x[None] = 2 * a

    write(x)
    assert x[None] == 2.0

    y = ti.ndarray(dtype=ti.math.vec2, shape=())
    write(y)
    assert y[None] == [2.0, 2.0]


@test_utils.test(arch=supported_archs_taichi_ndarray, require=ti.extension.data64)
def test_read_write_f64_python_scope():
    x = ti.ndarray(dtype=ti.f64, shape=2)

    x[0] = 1.0
    assert x[0] == 1.0

    y = ti.ndarray(dtype=ti.math.vec2, shape=2)
    y[0] = [1.0, 2.0]
    assert y[0] == [1.0, 2.0]


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_fill():
    vec2 = ti.types.vector(2, ti.f32)
    x_vec = ti.ndarray(vec2, (512, 512))
    x_vec.fill(1.0)
    assert (x_vec[2, 2] == [1.0, 1.0]).all()

    x_vec.fill(vec2(2.0, 4.0))
    assert (x_vec[3, 3] == [2.0, 4.0]).all()

    mat2x2 = ti.types.matrix(2, 2, ti.f32)
    x_mat = ti.ndarray(mat2x2, (512, 512))
    x_mat.fill(2.0)
    assert (x_mat[2, 2] == [[2.0, 2.0], [2.0, 2.0]]).all()

    x_mat.fill(mat2x2([[2.0, 4.0], [1.0, 3.0]]))
    assert (x_mat[3, 3] == [[2.0, 4.0], [1.0, 3.0]]).all()
