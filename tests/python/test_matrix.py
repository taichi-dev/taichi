import math
import operator

import numpy as np
import pytest
from taichi.lang import impl
from taichi.lang.exception import TaichiCompilationError
from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils

operation_types = [operator.add, operator.sub, operator.matmul]
test_matrix_arrays = [
    np.array([[1, 2], [3, 4]]),
    np.array([[5, 6], [7, 8]]),
    np.array([[2, 8], [-1, 3]])
]

vector_operation_types = [operator.add, operator.sub]
test_vector_arrays = [
    np.array([42, 42]),
    np.array([24, 24]),
    np.array([83, 12])
]


@test_utils.test(arch=get_host_arch_list())
def test_python_scope_vector_operations():
    for ops in vector_operation_types:
        a, b = test_vector_arrays[:2]
        m1, m2 = ti.Vector(a), ti.Vector(b)
        c = ops(m1, m2)
        assert np.allclose(c.to_numpy(), ops(a, b))


@test_utils.test(arch=get_host_arch_list())
def test_python_scope_matrix_operations():
    for ops in operation_types:
        a, b = test_matrix_arrays[:2]
        m1, m2 = ti.Matrix(a), ti.Matrix(b)
        c = ops(m1, m2)
        assert np.allclose(c.to_numpy(), ops(a, b))


# TODO: Loops inside the function will cause AssertionError:
# No new variables can be declared after kernel invocations
# or Python-scope field accesses.
# ideally we should use pytest.fixture to parameterize the tests
# over explicit loops
@pytest.mark.parametrize('ops', vector_operation_types)
@test_utils.test(arch=get_host_arch_list())
def test_python_scope_vector_field(ops):
    t1 = ti.Vector.field(2, dtype=ti.i32, shape=())
    t2 = ti.Vector.field(2, dtype=ti.i32, shape=())
    a, b = test_vector_arrays[:2]
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None], t2[None])
    assert np.allclose(c.to_numpy(), ops(a, b))


@pytest.mark.parametrize('ops', vector_operation_types)
@test_utils.test(arch=get_host_arch_list())
def test_python_scope_matrix_field(ops):
    t1 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    t2 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    a, b = test_matrix_arrays[:2]
    # ndarray not supported here
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None], t2[None])
    print(c)

    assert np.allclose(c.to_numpy(), ops(a, b))


@test_utils.test(arch=get_host_arch_list())
def test_constant_matrices():
    assert ti.cos(math.pi / 3) == test_utils.approx(0.5)
    assert np.allclose((-ti.Vector([2, 3])).to_numpy(), np.array([-2, -3]))
    assert ti.cos(ti.Vector([2, 3])).to_numpy() == test_utils.approx(
        np.cos(np.array([2, 3])))
    assert ti.max(2, 3) == 3
    res = ti.max(4, ti.Vector([3, 4, 5]))
    assert np.allclose(res.to_numpy(), np.array([4, 4, 5]))
    res = ti.Vector([2, 3]) + ti.Vector([3, 4])
    assert np.allclose(res.to_numpy(), np.array([5, 7]))
    res = ti.atan2(ti.Vector([2, 3]), ti.Vector([3, 4]))
    assert res.to_numpy() == test_utils.approx(
        np.arctan2(np.array([2, 3]), np.array([3, 4])))
    res = ti.Matrix([[2, 3], [4, 5]]) @ ti.Vector([2, 3])
    assert np.allclose(res.to_numpy(), np.array([13, 23]))
    v = ti.Vector([3, 4])
    w = ti.Vector([5, -12])
    r = ti.Vector([1, 2, 3, 4])
    s = ti.Matrix([[1, 2], [3, 4]])
    assert v.normalized().to_numpy() == test_utils.approx(np.array([0.6, 0.8]))
    assert v.cross(w) == test_utils.approx(-12 * 3 - 4 * 5)
    w.y = v.x * w[0]
    r.x = r.y
    r.y = r.z
    r.z = r.w
    r.w = r.x
    assert np.allclose(w.to_numpy(), np.array([5, 15]))
    assert ti.select(ti.Vector([1, 0]), ti.Vector([2, 3]),
                     ti.Vector([4, 5])) == ti.Vector([2, 5])
    s[0, 1] = 2
    assert s[0, 1] == 2

    @ti.kernel
    def func(t: ti.i32):
        m = ti.Matrix([[2, 3], [4, t]])
        print(m @ ti.Vector([2, 3]))
        m += ti.Matrix([[3, 4], [5, t]])
        print(m @ v)
        print(r.x, r.y, r.z, r.w)
        s = w @ m
        print(s)
        print(m)

    func(5)


@pytest.mark.parametrize('ops', vector_operation_types)
@test_utils.test(arch=get_host_arch_list())
def test_taichi_scope_vector_operations_with_global_vectors(ops):
    a, b, c = test_vector_arrays[:3]
    m1, m2 = ti.Vector(a), ti.Vector(b)
    r1 = ti.Vector.field(2, dtype=ti.i32, shape=())
    r2 = ti.Vector.field(2, dtype=ti.i32, shape=())
    m3 = ti.Vector.field(2, dtype=ti.i32, shape=())
    m3.from_numpy(c)

    @ti.kernel
    def run():
        r1[None] = ops(m1, m2)
        r2[None] = ops(m1, m3[None])

    run()

    assert np.allclose(r1[None].to_numpy(), ops(a, b))
    assert np.allclose(r2[None].to_numpy(), ops(a, c))


@pytest.mark.parametrize('ops', vector_operation_types)
@test_utils.test(arch=get_host_arch_list())
def test_taichi_scope_matrix_operations_with_global_matrices(ops):
    a, b, c = test_matrix_arrays[:3]
    m1, m2 = ti.Matrix(a), ti.Matrix(b)
    r1 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    r2 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    m3 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    m3.from_numpy(c)

    @ti.kernel
    def run():
        r1[None] = ops(m1, m2)
        r2[None] = ops(m1, m3[None])

    run()

    assert np.allclose(r1[None].to_numpy(), ops(a, b))
    assert np.allclose(r2[None].to_numpy(), ops(a, c))


def _test_local_matrix_non_constant_index():
    @ti.kernel
    def func1():
        tmp = ti.Vector([1, 2, 3])
        for i in range(3):
            vec = ti.Vector([4, 5, 6])
            for j in range(3):
                vec[tmp[i] % 3] += vec[j]
            tmp[i] = vec[tmp[i] % 3]
        assert tmp[0] == 24
        assert tmp[1] == 30
        assert tmp[2] == 19

    func1()

    @ti.kernel
    def func2(i: ti.i32, j: ti.i32, k: ti.i32):
        tmp = ti.Matrix([[k, k * 2], [k * 2, k * 3]])
        assert tmp[i, j] == k * (i + j + 1)

    for i in range(2):
        for j in range(2):
            func2(i, j, 10)


@test_utils.test(require=ti.extension.dynamic_index,
                 dynamic_index=True,
                 debug=True)
def test_local_matrix_non_constant_index():
    _test_local_matrix_non_constant_index()


@test_utils.test(require=ti.extension.dynamic_index,
                 real_matrix=True,
                 debug=True)
def test_local_matrix_non_constant_index_real_matrix():
    _test_local_matrix_non_constant_index()


@test_utils.test(require=ti.extension.dynamic_index,
                 dynamic_index=True,
                 real_matrix=True,
                 real_matrix_scalarize=True,
                 debug=True)
def test_local_matrix_non_constant_index_real_matrix_scalarize():
    _test_local_matrix_non_constant_index()


@test_utils.test(exclude=[ti.cc])
def test_matrix_ndarray_non_constant_index():
    @ti.kernel
    def func1(a: ti.types.ndarray(element_dim=2)):
        for i in range(5):
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = np.empty((5, 2, 2), dtype=np.int32)
    func1(m)
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1

    @ti.kernel
    def func2(b: ti.types.ndarray(element_dim=1)):
        for i in range(5):
            for j in range(4):
                b[i][j * j] = j * j

    v = np.empty((5, 10), dtype=np.int32)
    func2(v)
    assert v[0][0] == 0
    assert v[1][1] == 1
    assert v[2][4] == 4
    assert v[3][9] == 9


def _test_matrix_field_non_constant_index():
    m = ti.Matrix.field(2, 2, ti.i32, 5)
    v = ti.Vector.field(10, ti.i32, 5)

    @ti.kernel
    def func1():
        for i in range(5):
            for j, k in ti.ndrange(2, 2):
                m[i][j, k] = j * j + k * k

    func1()
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1

    @ti.kernel
    def func2():
        for i in range(5):
            for j in range(4):
                v[i][j * j] = j * j

    func2()
    assert v[1][0] == 0
    assert v[1][1] == 1
    assert v[1][4] == 4
    assert v[1][9] == 9


@test_utils.test(require=ti.extension.dynamic_index, dynamic_index=True)
def test_matrix_field_non_constant_index():
    _test_matrix_field_non_constant_index()


@test_utils.test(require=ti.extension.dynamic_index, real_matrix=True)
def test_matrix_field_non_constant_index_real_matrix():
    _test_matrix_field_non_constant_index()


def _test_matrix_field_constant_index():
    m = ti.Matrix.field(2, 2, ti.i32, 5)

    @ti.kernel
    def func():
        for i in range(5):
            for j, k in ti.static(ti.ndrange(2, 2)):
                m[i][j, k] = 12

    func()

    assert np.allclose(m.to_numpy(), np.ones((5, 2, 2), np.int32) * 12)


@test_utils.test()
def test_matrix_field_constant_index():
    _test_matrix_field_constant_index()


@test_utils.test(real_matrix=True)
def test_matrix_field_constant_index_real_matrix():
    _test_matrix_field_constant_index()


@test_utils.test(arch=ti.cpu)
def test_vector_to_list():
    a = ti.Vector.field(2, float, ())

    data = [2, 3]
    b = ti.Vector(data)
    assert list(b) == data
    assert len(b) == len(data)

    a[None] = b
    assert all(a[None] == ti.Vector(data))


@test_utils.test(arch=ti.cpu)
def test_matrix_to_list():
    a = ti.Matrix.field(2, 3, float, ())

    data = [[2, 3, 4], [5, 6, 7]]
    b = ti.Matrix(data)
    assert list(b) == data
    assert len(b) == len(data)

    a[None] = b
    assert all(a[None] == ti.Matrix(data))


@test_utils.test()
def test_matrix_needs_grad():
    # Just make sure the usage doesn't crash, see https://github.com/taichi-dev/taichi/pull/1545
    n = 8
    m1 = ti.Matrix.field(2, 2, ti.f32, n, needs_grad=True)
    m2 = ti.Matrix.field(2, 2, ti.f32, n, needs_grad=True)
    gr = ti.Matrix.field(2, 2, ti.f32, n)

    @ti.kernel
    def func():
        for i in range(n):
            gr[i] = m1.grad[i] + m2.grad[i]

    func()


@test_utils.test(debug=True)
def test_copy_python_scope_matrix_to_taichi_scope():
    a = ti.Vector([1, 2, 3])

    @ti.kernel
    def test():
        b = a
        assert b[0] == 1
        assert b[1] == 2
        assert b[2] == 3
        b = ti.Vector([4, 5, 6])
        assert b[0] == 4
        assert b[1] == 5
        assert b[2] == 6

    test()


@test_utils.test(exclude=[ti.cc], debug=True)
def test_copy_matrix_field_element_to_taichi_scope():
    a = ti.Vector.field(3, ti.i32, shape=())
    a[None] = ti.Vector([1, 2, 3])

    @ti.kernel
    def test():
        b = a[None]
        assert b[0] == 1
        assert b[1] == 2
        assert b[2] == 3
        b[0] = 5
        b[1] = 9
        b[2] = 7
        assert b[0] == 5
        assert b[1] == 9
        assert b[2] == 7
        assert a[None][0] == 1
        assert a[None][1] == 2
        assert a[None][2] == 3

    test()


@test_utils.test(debug=True)
def test_copy_matrix_in_taichi_scope():
    @ti.kernel
    def test():
        a = ti.Vector([1, 2, 3])
        b = a
        assert b[0] == 1
        assert b[1] == 2
        assert b[2] == 3
        b[0] = 5
        b[1] = 9
        b[2] = 7
        assert b[0] == 5
        assert b[1] == 9
        assert b[2] == 7
        assert a[0] == 1
        assert a[1] == 2
        assert a[2] == 3

    test()


@test_utils.test(arch=[ti.cpu, ti.cuda], dynamic_index=True, debug=True)
def test_matrix_field_dynamic_index_stride():
    # placeholders
    temp_a = ti.field(ti.f32)
    temp_b = ti.field(ti.f32)
    temp_c = ti.field(ti.f32)
    # target
    v = ti.Vector.field(3, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)
    z = v.get_scalar_field(2)

    S0 = ti.root
    S1 = S0.pointer(ti.i, 4)
    S2 = S1.dense(ti.i, 2)
    S3 = S2.pointer(ti.i, 8)
    S3.place(temp_a)
    S4 = S2.dense(ti.i, 16)
    S4.place(x)
    S5 = S1.dense(ti.i, 2)
    S6 = S5.pointer(ti.i, 8)
    S6.place(temp_b)
    S7 = S5.dense(ti.i, 16)
    S7.place(y)
    S8 = S1.dense(ti.i, 2)
    S9 = S8.dense(ti.i, 32)
    S9.place(temp_c)
    S10 = S8.dense(ti.i, 16)
    S10.place(z)

    @ti.kernel
    def check_stride():
        for i in range(128):
            assert ti.get_addr(y, i) - ti.get_addr(
                x, i) == v._get_dynamic_index_stride()
            assert ti.get_addr(z, i) - ti.get_addr(
                y, i) == v._get_dynamic_index_stride()

    check_stride()

    @ti.kernel
    def run():
        for i in range(128):
            for j in range(3):
                v[i][j] = i * j

    run()
    for i in range(128):
        for j in range(3):
            assert v[i][j] == i * j


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_matrix_field_dynamic_index_different_path_length():
    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 8).place(x)
    ti.root.dense(ti.i, 2).dense(ti.i, 4).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_matrix_field_dynamic_index_not_pure_dense():
    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 2).pointer(ti.i, 4).place(x)
    ti.root.dense(ti.i, 2).dense(ti.i, 4).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_matrix_field_dynamic_index_different_cell_size_bytes():
    temp = ti.field(ti.f32)

    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 8).place(x, temp)
    ti.root.dense(ti.i, 8).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_matrix_field_dynamic_index_different_offset_bytes_in_parent_cell():
    temp_a = ti.field(ti.f32)
    temp_b = ti.field(ti.f32)

    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 8).place(temp_a, x)
    ti.root.dense(ti.i, 8).place(y, temp_b)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_matrix_field_dynamic_index_different_stride():
    temp = ti.field(ti.f32)

    v = ti.Vector.field(3, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)
    z = v.get_scalar_field(2)

    ti.root.dense(ti.i, 8).place(x, y, temp, z)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test(arch=[ti.cpu, ti.cuda], dynamic_index=True)
def test_matrix_field_dynamic_index_multiple_materialize():
    @ti.kernel
    def empty():
        pass

    empty()

    n = 5
    a = ti.Vector.field(3, dtype=ti.i32, shape=n)

    @ti.kernel
    def func():
        for i in a:
            a[i][i % 3] = i

    func()
    for i in range(n):
        for j in range(3):
            assert a[i][j] == (i if j == i % 3 else 0)


@test_utils.test(arch=[ti.cpu, ti.cuda], dynamic_index=True, debug=True)
def test_local_vector_initialized_in_a_loop():
    @ti.kernel
    def foo():
        for c in range(10):
            p = ti.Vector([c, c * 2])
            for i in range(2):
                assert p[i] == c * (i + 1)

    foo()


@test_utils.test(debug=True)
def test_vector_dtype():
    @ti.kernel
    def foo():
        a = ti.Vector([1, 2, 3], ti.f32)
        a /= 2
        assert all(abs(a - (0.5, 1., 1.5)) < 1e-6)
        b = ti.Vector([1.5, 2.5, 3.5], ti.i32)
        assert all(b == (1, 2, 3))

    foo()


@test_utils.test(debug=True)
def test_matrix_dtype():
    @ti.kernel
    def foo():
        a = ti.Matrix([[1, 2], [3, 4]], ti.f32)
        a /= 2
        assert all(abs(a - ((0.5, 1.), (1.5, 2.))) < 1e-6)
        b = ti.Matrix([[1.5, 2.5], [3.5, 4.5]], ti.i32)
        assert all(b == ((1, 2), (3, 4)))

    foo()


inplace_operation_types = [
    operator.iadd, operator.isub, operator.imul, operator.ifloordiv,
    operator.imod, operator.ilshift, operator.irshift, operator.ior,
    operator.ixor, operator.iand
]


@test_utils.test()
def test_python_scope_inplace_operator():
    for ops in inplace_operation_types:
        a, b = test_matrix_arrays[:2]
        m1, m2 = ti.Matrix(a), ti.Matrix(b)
        m1 = ops(m1, m2)
        assert np.allclose(m1.to_numpy(), ops(a, b))


@test_utils.test()
def test_slice_assign_basic():
    @ti.kernel
    def foo():
        m = ti.Matrix([[0., 0., 0., 0.] for _ in range(3)])
        vec = ti.Vector([1., 2., 3., 4.])
        m[0, :] = vec.transpose()
        ref = ti.Matrix([[1., 2., 3., 4.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
        assert all(m == ref)

        m[1, 1:3] = ti.Vector([1., 2.]).transpose()
        ref = ti.Matrix([[1., 2., 3., 4.], [0., 1., 2., 0.], [0., 0., 0., 0.]])
        assert all(m == ref)

        m1 = ti.Matrix([[1., 1., 1., 1.] for _ in range(2)])
        m[:2, :] = m1
        ref = ti.Matrix([[1., 1., 1., 1.], [1., 1., 1., 1.], [0., 0., 0., 0.]])
        assert all(m == ref)

    foo()


@test_utils.test(dynamic_index=True)
def test_slice_assign_dynamic_index():
    @ti.kernel
    def foo(i: ti.i32, ref: ti.template()):
        m = ti.Matrix([[0., 0., 0., 0.] for _ in range(3)])
        vec = ti.Vector([1., 2., 3., 4.])
        m[i, :] = vec.transpose()
        assert all(m == ref)

    for i in range(3):
        foo(
            i,
            ti.Matrix([[1., 2., 3., 4.] if j == i else [0., 0., 0., 0.]
                       for j in range(3)]))


@test_utils.test()
def test_indexing():
    @ti.kernel
    def foo():
        m = ti.Matrix([[0., 0., 0., 0.] for _ in range(4)])
        print(m[0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 2 indices, got 1'):
        foo()

    @ti.kernel
    def bar():
        vec = ti.Vector([1, 2, 3, 4])
        print(vec[0, 0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 1 indices, got 2'):
        bar()


@test_utils.test()
def test_indexing_in_fields():
    f = ti.Matrix.field(3, 3, ti.f32, shape=())

    @ti.kernel
    def foo():
        f[None][0, 0] = 1.0
        print(f[None][0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 2 indices, got 1'):
        foo()

    g = ti.Vector.field(3, ti.f32, shape=())

    @ti.kernel
    def bar():
        g[None][0] = 1.0
        print(g[None][0, 0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 1 indices, got 2'):
        bar()


@test_utils.test()
def test_indexing_in_struct():
    @ti.kernel
    def foo():
        s = ti.Struct(a=ti.Vector([0, 0, 0]), b=2)
        print(s.a[0, 0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 1 indices, got 2'):
        foo()

    @ti.kernel
    def bar():
        s = ti.Struct(m=ti.Matrix([[0, 0, 0], [0, 0, 0]]), n=2)
        print(s.m[0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 2 indices, got 1'):
        bar()


@test_utils.test()
def test_indexing_in_struct_field():

    s = ti.Struct.field(
        {
            'v': ti.types.vector(3, ti.f32),
            'm': ti.types.matrix(3, 3, ti.f32)
        },
        shape=())

    @ti.kernel
    def foo():
        print(s[None].v[0, 0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 1 indices, got 2'):
        foo()

    @ti.kernel
    def bar():
        print(s[None].m[0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 2 indices, got 1'):
        bar()


@test_utils.test(arch=get_host_arch_list(), debug=True)
def test_matrix_vector_multiplication():
    mat = ti.math.mat3(1)
    vec = ti.math.vec3(3)
    r = mat @ vec
    for i in range(3):
        assert r[i] == 9

    @ti.kernel
    def foo():
        mat = ti.math.mat3(1)
        vec = ti.math.vec3(3)
        r = mat @ vec
        assert r[0] == r[1] == r[2] == 9

    foo()


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix=True)
def test_local_matrix_read():

    s = ti.field(ti.i32, shape=())

    @ti.kernel
    def get_index(i: ti.i32, j: ti.i32):
        mat = ti.Matrix([[x * 3 + y for y in range(3)] for x in range(3)])
        s[None] = mat[i, j]

    for i in range(3):
        for j in range(3):
            get_index(i, j)
            assert s[None] == i * 3 + j


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix=True)
def test_local_matrix_read_without_assign():
    @ti.kernel
    def local_vector_read(i: ti.i32) -> ti.i32:
        return ti.Vector([0, 1, 2])[i]

    for i in range(3):
        assert local_vector_read(i) == i


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix=True)
def test_local_matrix_indexing_in_loop():
    s = ti.field(ti.i32, shape=(3, 3))

    @ti.kernel
    def test():
        mat = ti.Matrix([[x * 3 + y for y in range(3)] for x in range(3)])
        for i in range(3):
            for j in range(3):
                s[i, j] = mat[i, j] + 1

    test()
    for i in range(3):
        for j in range(3):
            assert s[i, j] == i * 3 + j + 1


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix=True)
def test_local_matrix_indexing_ops():
    @ti.kernel
    def element_write() -> ti.i32:
        mat = ti.Matrix([[x * 3 + y for y in range(3)] for x in range(3)])
        s = 0
        for i in range(3):
            for j in range(3):
                mat[i, j] = 10
                s += mat[i, j]
        return s

    f = ti.field(ti.i32, shape=(3, 3))

    @ti.kernel
    def assign_from_index():
        mat = ti.Matrix([[x * 3 + y for y in range(3)] for x in range(3)])
        result = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # TODO: fix parallelization
        ti.loop_config(serialize=True)
        for i in range(3):
            for j in range(3):
                result[i, j] = mat[j, i]
        for i in range(3):
            for j in range(3):
                f[i, j] = result[i, j]

    assert element_write() == 90
    assign_from_index()
    xs = [[x * 3 + y for y in range(3)] for x in range(3)]
    for i in range(3):
        for j in range(3):
            assert f[i, j] == xs[j][i]


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix=True)
def test_local_matrix_index_check():
    @ti.kernel
    def foo():
        mat = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        print(mat[0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 2 indices, but got 1'):
        foo()

    @ti.kernel
    def bar():
        vec = ti.Vector([1, 2, 3, 4])
        print(vec[0, 0])

    with pytest.raises(TaichiCompilationError,
                       match=r'Expected 1 indices, but got 2'):
        bar()


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix=True, debug=True)
def test_elementwise_ops():
    @ti.kernel
    def test():
        # TODO: fix parallelization
        x = ti.Matrix([[1, 2], [3, 4]])
        # Unify rhs
        t1 = x + 10
        ti.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t1[i, j] == x[i, j] + 10
        t2 = x * 2
        ti.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t2[i, j] == x[i, j] * 2
        # elementwise-add
        t3 = t1 + t2
        ti.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t3[i, j] == t1[i, j] + t2[i, j]
        # Unify lhs
        t4 = 1 / t1
        # these should be *exactly* equals
        ti.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t4[i, j] == 1 / t1[i, j]
        t5 = 1 << x
        ti.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t5[i, j] == 1 << x[i, j]
        t6 = 1 + (x // 2)
        ti.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t6[i, j] == 1 + (x[i, j] // 2)

        # test floordiv
        y = ti.Matrix([[1, 2], [3, 4]], dt=ti.i32)
        z = y * 2
        factors = z // y
        ti.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert factors[i, j] == 2

        y1 = ti.Matrix([[1, 2], [3, 4]], dt=ti.f32)
        z1 = y1 * 2
        factors1 = z1 // y1
        ti.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert factors1[i, j] == 2

    test()


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_local_matrix_scalarize():
    @ti.kernel
    def func():
        x = ti.Matrix([[1, 2], [3, 4]], ti.f32)

        # Store
        x[0, 0] = 100.

        # Load + Store
        x[0, 1] = x[0, 0]

        # Binary
        x[1, 0] = x[0, 1] + x[0, 1]

        # Unary
        x[1, 1] = ti.sqrt(x[1, 0])

        # TODO: test for dynamic indexing

        assert (x[0, 0] == 100.)
        assert (x[0, 1] == 200.)
        assert (x[1, 0] == 200.)
        assert (x[1, 1] < 14.14214)
        assert (x[1, 1] > 14.14213)

    func()


@test_utils.test()
def test_vector_vector_t():
    @ti.kernel
    def foo() -> ti.types.matrix(2, 2, ti.f32):
        a = ti.Vector([1.0, 2.0])
        b = ti.Vector([1.0, 2.0])
        return a @ b.transpose()

    assert foo() == [[1.0, 2.0], [2.0, 4.0]]


def _test_field_and_ndarray(field, ndarray, func, verify):
    @ti.kernel
    def kern_field(a: ti.template()):
        func(a)

    @ti.kernel
    def kern_ndarray(a: ti.types.ndarray()):
        func(a)

    kern_field(field)
    verify(field)
    kern_ndarray(ndarray)
    verify(ndarray)


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_store_scalarize():
    @ti.func
    def func(a: ti.template()):
        for i in range(5):
            a[i] = [[i, i + 1], [i + 2, i + 3]]

    def verify(x):
        assert (x[0] == [[0, 1], [2, 3]]).all()
        assert (x[1] == [[1, 2], [3, 4]]).all()
        assert (x[2] == [[2, 3], [4, 5]]).all()
        assert (x[3] == [[3, 4], [5, 6]]).all()
        assert (x[4] == [[4, 5], [6, 7]]).all()

    field = ti.Matrix.field(2, 2, ti.i32, shape=5)
    ndarray = ti.Matrix.ndarray(2, 2, ti.i32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_load_store_scalarize():
    @ti.func
    def func(a: ti.template()):
        for i in range(3):
            a[i] = [[i, i + 1], [i + 2, i + 3]]

        a[3] = a[1]
        a[4] = a[2]

    def verify(x):
        assert (x[3] == [[1, 2], [3, 4]]).all()
        assert (x[4] == [[2, 3], [4, 5]]).all()

    field = ti.Matrix.field(2, 2, ti.i32, shape=5)
    ndarray = ti.Matrix.ndarray(2, 2, ti.i32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_unary_op_scalarize():
    @ti.func
    def func(a: ti.template()):
        a[0] = [[0, 1], [2, 3]]
        a[1] = [[3, 4], [5, 6]]
        a[2] = -a[0]
        a[3] = ti.exp(a[1])
        a[4] = ti.sqrt(a[3])

    def verify(x):
        assert (x[0] == [[0., 1.], [2., 3.]]).all()
        assert (x[1] == [[3., 4.], [5., 6.]]).all()
        assert (x[2] == [[-0., -1.], [-2., -3.]]).all()
        assert (x[3] < [[20.086, 54.60], [148.42, 403.43]]).all()
        assert (x[3] > [[20.085, 54.59], [148.41, 403.42]]).all()
        assert (x[4] < [[4.49, 7.39], [12.19, 20.09]]).all()
        assert (x[4] > [[4.48, 7.38], [12.18, 20.08]]).all()

    field = ti.Matrix.field(2, 2, ti.f32, shape=5)
    ndarray = ti.Matrix.ndarray(2, 2, ti.f32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_binary_op_scalarize():
    @ti.func
    def func(a: ti.template()):
        a[0] = [[0., 1.], [2., 3.]]
        a[1] = [[3., 4.], [5., 6.]]
        a[2] = a[0] + a[0]
        a[3] = a[1] * a[1]
        a[4] = ti.max(a[2], a[3])

    def verify(x):
        assert (x[2] == [[0., 2.], [4., 6.]]).all()
        assert (x[3] == [[9., 16.], [25., 36.]]).all()
        assert (x[4] == [[9., 16.], [25., 36.]]).all()

    field = ti.Matrix.field(2, 2, ti.f32, shape=5)
    ndarray = ti.Matrix.ndarray(2, 2, ti.f32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_trace_op():
    @ti.kernel
    def test_fun() -> ti.f32:
        x = ti.Matrix([[.1, 3.], [5., 7.]])
        return x.trace()

    assert np.abs(test_fun() - 7.1) < 1e-6

    x = ti.Matrix([[.1, 3.], [5., 7.]])
    assert np.abs(x.trace() - 7.1) < 1e-6

    with pytest.raises(TaichiCompilationError,
                       match=r"not a square matrix: \(3, 2\)"):
        x = ti.Matrix([[.1, 3.], [5., 7.], [1., 2.]])
        print(x.trace())

    @ti.kernel
    def failed_func():
        x = ti.Matrix([[.1, 3.], [5., 7.], [1., 2.]])
        print(x.trace())

    with pytest.raises(TaichiCompilationError,
                       match=r"not a square matrix: \(3, 2\)"):
        failed_func()


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True,
                 debug=True)
def test_ternary_op_scalarize():
    @ti.kernel
    def test():
        cond = ti.Vector([1, 0, 1])
        x = ti.Vector([3, 3, 3])
        y = ti.Vector([5, 5, 5])

        z = ti.select(cond, x, y)

        assert z[0] == 3
        assert z[1] == 5
        assert z[2] == 3

    test()


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True,
                 debug=True)
def test_fill_op():
    @ti.kernel
    def test_fun():
        x = ti.Matrix([[0.0 for _ in range(4)] for _ in range(5)])
        y = x.fill(1.14)
        for i in ti.static(range(5)):
            for j in ti.static(range(4)):
                assert y[i, j] == x[i, j] == 1.14

    test_fun()


@test_utils.test(arch=[ti.cuda, ti.cpu],
                 real_matrix=True,
                 real_matrix_scalarize=True,
                 debug=True)
def test_atomic_op_scalarize():
    @ti.func
    def func(x: ti.template()):
        x[0] = [1., 2., 3.]
        tmp = ti.Vector([3., 2., 1.])
        z = ti.atomic_add(x[0], tmp)
        assert z[0] == 1.
        assert z[1] == 2.
        assert z[2] == 3.

        # Broadcasting
        x[1] = [1., 1., 1.]
        g = ti.atomic_add(x[1], 2)
        assert g[0] == 1.
        assert g[1] == 1.
        assert g[2] == 1.

    def verify(x):
        assert (x[0] == [4., 4., 4.]).all()
        assert (x[1] == [3., 3., 3.]).all()

    field = ti.Vector.field(n=3, dtype=ti.f32, shape=10)
    ndarray = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=(10))
    _test_field_and_ndarray(field, ndarray, func, verify)
