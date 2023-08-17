import math
import operator

import numpy as np
import pytest
from pytest import approx
from taichi.lang import impl
from taichi.lang.exception import TaichiCompilationError, TaichiTypeError
from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils

matrix_operation_types = [operator.add, operator.sub, operator.matmul]
test_matrix_arrays = [
    np.array([[1, 2], [3, 4]]),
    np.array([[5, 6], [7, 8]]),
    np.array([[2, 8], [-1, 3]]),
]

vector_operation_types = [operator.add, operator.sub]
test_vector_arrays = [np.array([42, 42]), np.array([24, 24]), np.array([83, 12])]


@test_utils.test(arch=get_host_arch_list())
def test_python_scope_vector_operations():
    for ops in vector_operation_types:
        a, b = test_vector_arrays[:2]
        m1, m2 = ti.Vector(a), ti.Vector(b)
        c = ops(m1, m2)
        assert np.allclose(c.to_numpy(), ops(a, b))


@test_utils.test(arch=get_host_arch_list())
def test_python_scope_matrix_operations():
    for ops in matrix_operation_types:
        a, b = test_matrix_arrays[:2]
        m1, m2 = ti.Matrix(a), ti.Matrix(b)
        c = ops(m1, m2)
        assert np.allclose(c.to_numpy(), ops(a, b))


# TODO: Loops inside the function will cause AssertionError:
# No new variables can be declared after kernel invocations
# or Python-scope field accesses.
# ideally we should use pytest.fixture to parameterize the tests
# over explicit loops
@pytest.mark.parametrize("ops", vector_operation_types)
@test_utils.test(arch=get_host_arch_list())
def test_python_scope_vector_field(ops):
    t1 = ti.Vector.field(2, dtype=ti.i32, shape=())
    t2 = ti.Vector.field(2, dtype=ti.i32, shape=())
    a, b = test_vector_arrays[:2]
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None], t2[None])
    assert np.allclose(c.to_numpy(), ops(a, b))


@pytest.mark.parametrize("ops", matrix_operation_types)
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
    assert ti.cos(ti.Vector([2, 3])).to_numpy() == test_utils.approx(np.cos(np.array([2, 3])))
    assert ti.max(2, 3) == 3
    res = ti.max(4, ti.Vector([3, 4, 5]))
    assert np.allclose(res.to_numpy(), np.array([4, 4, 5]))
    res = ti.Vector([2, 3]) + ti.Vector([3, 4])
    assert np.allclose(res.to_numpy(), np.array([5, 7]))
    res = ti.atan2(ti.Vector([2, 3]), ti.Vector([3, 4]))
    assert res.to_numpy() == test_utils.approx(np.arctan2(np.array([2, 3]), np.array([3, 4])))
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
    assert ti.select(ti.Vector([1, 0]), ti.Vector([2, 3]), ti.Vector([4, 5])) == ti.Vector([2, 5])
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


@test_utils.test(arch=get_host_arch_list())
def test_taichi_scope_vector_operations_with_global_vectors():
    for ops in vector_operation_types:
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


@test_utils.test(arch=get_host_arch_list())
def test_taichi_scope_matrix_operations_with_global_matrices():
    for ops in matrix_operation_types:
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
    def func1() -> ti.types.vector(3, ti.i32):
        tmp = ti.Vector([1, 2, 3])
        for i in range(3):
            vec = ti.Vector([4, 5, 6])
            for j in range(3):
                vec[tmp[i] % 3] += vec[j]
            tmp[i] = vec[tmp[i] % 3]
        return tmp

    assert (func1() == ti.Vector([24, 30, 19])).all()

    @ti.kernel
    def func2(i: ti.i32, j: ti.i32, k: ti.i32) -> ti.i32:
        tmp = ti.Matrix([[k, k * 2], [k * 2, k * 3]])
        return tmp[i, j]

    for i in range(2):
        for j in range(2):
            assert func2(i, j, 10) == 10 * (i + j + 1)


@test_utils.test()
def test_local_matrix_non_constant_index():
    _test_local_matrix_non_constant_index()


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix_scalarize=False)
def test_local_matrix_non_constant_index_real_matrix():
    _test_local_matrix_non_constant_index()


@test_utils.test()
def test_matrix_ndarray_non_constant_index():
    @ti.kernel
    def func1(a: ti.types.ndarray(dtype=ti.math.mat2)):
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
    def func2(b: ti.types.ndarray(dtype=ti.types.vector(n=10, dtype=ti.i32))):
        for i in range(5):
            for j in range(4):
                b[i][j * j] = j * j

    v = np.empty((5, 10), dtype=np.int32)
    func2(v)
    assert v[0][0] == 0
    assert v[1][1] == 1
    assert v[2][4] == 4
    assert v[3][9] == 9


@test_utils.test()
def test_matrix_field_non_constant_index():
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


@test_utils.test()
def test_matrix_field_constant_index():
    m = ti.Matrix.field(2, 2, ti.i32, 5)

    @ti.kernel
    def func():
        for i in range(5):
            for j, k in ti.static(ti.ndrange(2, 2)):
                m[i][j, k] = 12

    func()

    assert np.allclose(m.to_numpy(), np.ones((5, 2, 2), np.int32) * 12)


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


@test_utils.test(debug=True)
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


@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
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
            assert ti.get_addr(y, i) - ti.get_addr(x, i) == v._get_dynamic_index_stride()
            assert ti.get_addr(z, i) - ti.get_addr(y, i) == v._get_dynamic_index_stride()

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


@test_utils.test()
def test_matrix_field_dynamic_index_different_path_length():
    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 8).place(x)
    ti.root.dense(ti.i, 2).dense(ti.i, 4).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test(require=ti.extension.sparse, exclude=[ti.metal])
def test_matrix_field_dynamic_index_not_pure_dense():
    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 2).pointer(ti.i, 4).place(x)
    ti.root.dense(ti.i, 2).dense(ti.i, 4).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test()
def test_matrix_field_dynamic_index_different_cell_size_bytes():
    temp = ti.field(ti.f32)

    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 8).place(x, temp)
    ti.root.dense(ti.i, 8).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test()
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


@test_utils.test()
def test_matrix_field_dynamic_index_different_stride():
    temp = ti.field(ti.f32)

    v = ti.Vector.field(3, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)
    z = v.get_scalar_field(2)

    ti.root.dense(ti.i, 8).place(x, y, temp, z)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test()
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


@test_utils.test(debug=True)
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
        assert all(abs(a - (0.5, 1.0, 1.5)) < 1e-6)
        b = ti.Vector([1.5, 2.5, 3.5], ti.i32)
        assert all(b == (1, 2, 3))

    foo()


@test_utils.test(debug=True)
def test_matrix_dtype():
    @ti.kernel
    def foo():
        a = ti.Matrix([[1, 2], [3, 4]], ti.f32)
        a /= 2
        assert all(abs(a - ((0.5, 1.0), (1.5, 2.0))) < 1e-6)
        b = ti.Matrix([[1.5, 2.5], [3.5, 4.5]], ti.i32)
        assert all(b == ((1, 2), (3, 4)))

    foo()


inplace_operation_types = [
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.ifloordiv,
    operator.imod,
    operator.ilshift,
    operator.irshift,
    operator.ior,
    operator.ixor,
    operator.iand,
]


@test_utils.test()
def test_python_scope_inplace_operator():
    for ops in inplace_operation_types:
        a, b = test_matrix_arrays[:2]
        m1, m2 = ti.Matrix(a), ti.Matrix(b)
        m1 = ops(m1, m2)
        assert np.allclose(m1.to_numpy(), ops(a, b))


@test_utils.test()
def test_indexing():
    @ti.kernel
    def foo():
        m = ti.Matrix([[0.0, 0.0, 0.0, 0.0] for _ in range(4)])
        print(m[0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 2 indices, got 1"):
        foo()

    @ti.kernel
    def bar():
        vec = ti.Vector([1, 2, 3, 4])
        print(vec[0, 0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 1 indices, got 2"):
        bar()


@test_utils.test()
def test_indexing_in_fields():
    f = ti.Matrix.field(3, 3, ti.f32, shape=())

    @ti.kernel
    def foo():
        f[None][0, 0] = 1.0
        print(f[None][0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 2 indices, got 1"):
        foo()

    g = ti.Vector.field(3, ti.f32, shape=())

    @ti.kernel
    def bar():
        g[None][0] = 1.0
        print(g[None][0, 0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 1 indices, got 2"):
        bar()


@test_utils.test()
def test_indexing_in_struct():
    @ti.kernel
    def foo():
        s = ti.Struct(a=ti.Vector([0, 0, 0]), b=2)
        print(s.a[0, 0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 1 indices, got 2"):
        foo()

    @ti.kernel
    def bar():
        s = ti.Struct(m=ti.Matrix([[0, 0, 0], [0, 0, 0]]), n=2)
        print(s.m[0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 2 indices, got 1"):
        bar()


@test_utils.test()
def test_indexing_in_struct_field():
    s = ti.Struct.field({"v": ti.types.vector(3, ti.f32), "m": ti.types.matrix(3, 3, ti.f32)}, shape=())

    @ti.kernel
    def foo():
        print(s[None].v[0, 0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 1 indices, got 2"):
        foo()

    @ti.kernel
    def bar():
        print(s[None].m[0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 2 indices, got 1"):
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


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix_scalarize=False)
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


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix_scalarize=False)
def test_local_matrix_read_without_assign():
    @ti.kernel
    def local_vector_read(i: ti.i32) -> ti.i32:
        return ti.Vector([0, 1, 2])[i]

    for i in range(3):
        assert local_vector_read(i) == i


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix_scalarize=False)
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


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix_scalarize=False)
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


@test_utils.test()
def test_local_matrix_index_check():
    @ti.kernel
    def foo():
        mat = ti.Matrix([[1, 2, 3], [4, 5, 6]])
        print(mat[0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 2 indices, got 1"):
        foo()

    @ti.kernel
    def bar():
        vec = ti.Vector([1, 2, 3, 4])
        print(vec[0, 0])

    with pytest.raises(TaichiCompilationError, match=r"Expected 1 indices, got 2"):
        bar()


@test_utils.test(arch=[ti.cuda, ti.cpu], real_matrix_scalarize=False, debug=True)
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


@test_utils.test(debug=True)
def test_local_matrix_scalarize():
    @ti.kernel
    def func():
        x = ti.Matrix([[1, 2], [3, 4]], ti.f32)

        # Store
        x[0, 0] = 100.0

        # Load + Store
        x[0, 1] = x[0, 0]

        # Binary
        x[1, 0] = x[0, 1] + x[0, 1]

        # Unary
        x[1, 1] = ti.sqrt(x[1, 0])

        assert x[0, 0] == 100.0
        assert x[0, 1] == 100.0
        assert x[1, 0] == 200.0
        assert x[1, 1] < 14.14214
        assert x[1, 1] > 14.14213

    func()


@test_utils.test()
def test_vector_vector_t():
    @ti.kernel
    def foo() -> ti.f32:
        a = ti.Vector([1.0, 2.0])
        b = ti.Vector([1.0, 2.0])
        return a @ b

    assert foo() == test_utils.approx(5.0)


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


@test_utils.test()
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


@test_utils.test()
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


@test_utils.test()
def test_load_broadcast():
    @ti.func
    def func(a: ti.template()):
        for i in ti.grouped(a):
            a[i] = 42

    def verify(x):
        for i in range(5):
            assert (x[i] == [[42, 42], [42, 42]]).all()

    field = ti.Matrix.field(2, 2, ti.i32, shape=5)
    ndarray = ti.Matrix.ndarray(2, 2, ti.i32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test()
def test_unary_op_scalarize():
    @ti.func
    def func(a: ti.template()):
        a[0] = [[0, 1], [2, 3]]
        a[1] = [[3, 4], [5, 6]]
        a[2] = -a[0]
        a[3] = ti.exp(a[1])
        a[4] = ti.sqrt(a[3])

    def verify(x):
        assert (x[0] == [[0.0, 1.0], [2.0, 3.0]]).all()
        assert (x[1] == [[3.0, 4.0], [5.0, 6.0]]).all()
        assert (x[2] == [[-0.0, -1.0], [-2.0, -3.0]]).all()
        assert (x[3] < [[20.086, 54.60], [148.42, 403.43]]).all()
        assert (x[3] > [[20.085, 54.59], [148.41, 403.42]]).all()
        assert (x[4] < [[4.49, 7.39], [12.19, 20.09]]).all()
        assert (x[4] > [[4.48, 7.38], [12.18, 20.08]]).all()

    field = ti.Matrix.field(2, 2, ti.f32, shape=5)
    ndarray = ti.Matrix.ndarray(2, 2, ti.f32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test()
def test_binary_op_scalarize():
    @ti.func
    def func(a: ti.template()):
        a[0] = [[0.0, 1.0], [2.0, 3.0]]
        a[1] = [[3.0, 4.0], [5.0, 6.0]]
        a[2] = a[0] + a[0]
        a[3] = a[1] * a[1]
        a[4] = ti.max(a[2], a[3])

    def verify(x):
        assert (x[2] == [[0.0, 2.0], [4.0, 6.0]]).all()
        assert (x[3] == [[9.0, 16.0], [25.0, 36.0]]).all()
        assert (x[4] == [[9.0, 16.0], [25.0, 36.0]]).all()

    field = ti.Matrix.field(2, 2, ti.f32, shape=5)
    ndarray = ti.Matrix.ndarray(2, 2, ti.f32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test()
def test_trace_op():
    @ti.kernel
    def test_fun() -> ti.f32:
        x = ti.Matrix([[0.1, 3.0], [5.0, 7.0]])
        return x.trace()

    assert np.abs(test_fun() - 7.1) < 1e-6

    x = ti.Matrix([[0.1, 3.0], [5.0, 7.0]])
    assert np.abs(x.trace() - 7.1) < 1e-6

    with pytest.raises(TaichiCompilationError, match=r"expected a square matrix, got shape \(3, 2\)"):
        x = ti.Matrix([[0.1, 3.0], [5.0, 7.0], [1.0, 2.0]])
        print(x.trace())

    @ti.kernel
    def failed_func():
        x = ti.Matrix([[0.1, 3.0], [5.0, 7.0], [1.0, 2.0]])
        print(x.trace())

    with pytest.raises(TaichiCompilationError, match=r"expected a square matrix, got shape \(3, 2\)"):
        failed_func()


@test_utils.test(debug=True)
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


@test_utils.test(debug=True)
def test_ternary_op_cond_is_scalar():
    @ti.kernel
    def test():
        x = ti.Vector([3, 3, 3])
        y = ti.Vector([5, 5, 5])

        for i in range(10):
            z = ti.select(i % 2, x, y)
            if i % 2 == 1:
                assert z[0] == x[0] and z[1] == x[1] and z[2] == x[2]
            else:
                assert z[0] == y[0] and z[1] == y[1] and z[2] == y[2]

    test()


@test_utils.test(debug=True)
def test_fill_op():
    @ti.kernel
    def test_fun():
        x = ti.Matrix([[0.0 for _ in range(4)] for _ in range(5)])
        x.fill(1.14)
        for i in ti.static(range(5)):
            for j in ti.static(range(4)):
                assert x[i, j] == 1.14

    test_fun()


@test_utils.test(debug=True)
def test_atomic_op_scalarize():
    @ti.func
    def func(x: ti.template()):
        x[0] = [1.0, 2.0, 3.0]
        tmp = ti.Vector([3, 2, 1])
        z = ti.atomic_sub(x[0], tmp)
        assert z[0] == 1.0
        assert z[1] == 2.0
        assert z[2] == 3.0

        # Broadcasting
        x[1] = [1.0, 1.0, 1.0]
        g = ti.atomic_add(x[1], 2)
        assert g[0] == 1.0
        assert g[1] == 1.0
        assert g[2] == 1.0

    def verify(x):
        assert (x[0] == [-2.0, 0.0, 2.0]).all()
        assert (x[1] == [3.0, 3.0, 3.0]).all()

    field = ti.Vector.field(n=3, dtype=ti.f32, shape=10)
    ndarray = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=(10))
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test()
def test_vector_transpose():
    @ti.kernel
    def foo():
        x = ti.Vector([1, 2])
        y = ti.Vector([3, 4])
        z = x @ y.transpose()

    with pytest.raises(
        TaichiCompilationError,
        match=r"`transpose\(\)` cannot apply to a vector. If you want something like `a @ b.transpose\(\)`, write `a.outer_product\(b\)` instead.",
    ):
        foo()


@test_utils.test(debug=True)
def test_cross_scope_matrix_binary_ops():
    n = 128
    x = ti.Vector.field(3, dtype=int, shape=(n, n))
    spring_offsets = [ti.Vector([1, 2]), ti.Vector([2, 3])]

    @ti.kernel
    def test():
        vec = ti.Vector([4, 5])
        ind0 = vec + ti.static(spring_offsets)[0]
        ind1 = ti.lang.ops.add(vec, ti.static(spring_offsets)[1])

        x[ind0] = [100, 10, 1]
        x[ind1] = [1, 10, 100]

    test()

    assert (x[5, 7] == [100, 10, 1]).all()
    assert (x[6, 8] == [1, 10, 100]).all()


@test_utils.test(debug=True)
def test_cross_scope_matrix_ternary_ops():
    n = 128
    x = ti.Vector.field(3, dtype=int, shape=(n, n))
    spring_offsets = [ti.Vector([1, 2]), ti.Vector([2, 3])]

    @ti.kernel
    def test():
        vec = ti.Vector([0, 1])
        ind0 = ti.select(vec, vec, ti.static(spring_offsets)[0])
        x[ind0] = [100, 10, 1]

    test()

    assert (x[1, 1] == [100, 10, 1]).all()


@test_utils.test(debug=True)
def test_cross_scope_matrix_atomic_ops():
    n = 128
    x = ti.Vector.field(3, dtype=int, shape=(n, n))
    spring_offsets = [ti.Vector([1, 2]), ti.Vector([2, 3])]

    @ti.kernel
    def test():
        vec = ti.Vector([0, 1])
        vec += ti.static(spring_offsets)[0]
        x[vec] = [100, 10, 1]

    test()

    assert (x[1, 3] == [100, 10, 1]).all()


@test_utils.test(debug=True)
def test_global_tmp_overwrite():
    # https://github.com/taichi-dev/taichi/issues/6663
    @ti.kernel
    def foo() -> ti.i32:
        p = ti.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        loop = 1
        sig = ti.Vector([0, 0, 0, 0])
        assert p[0, 0] == 1
        while loop == 1:
            assert p[0, 0] == 1
            loop = 0
            p[0, 0] = -1
        for i in range(1):
            sig[i] = 2
        return sig.sum() + p.sum()

    assert foo() == 4


@test_utils.test(debug=True)
def test_matrix_len():
    @ti.kernel
    def test():
        x = ti.Vector([1, 0])
        y = ti.Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        assert len(x) == 2
        assert len(y) == 3

    test()


@test_utils.test()
def test_cross_scope_matrix():
    a = ti.Matrix([[1, 2], [3, 4]])

    @ti.kernel
    def foo() -> ti.types.vector(4, ti.i32):
        return ti.Vector([a[0, 0], a[0, 1], a[1, 0], a[1, 1]])

    assert (foo() == [1, 2, 3, 4]).all()


@test_utils.test(debug=True)
def test_matrix_type_inference():
    @ti.kernel
    def foo():
        a = ti.Vector([1, 2.5])[1]  # should be f32 instead of i32
        assert a == 2.5

    foo()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_matrix_arithmatics():
    f = ti.ndarray(ti.math.vec4, 4)

    @ti.kernel
    def fill(arr: ti.types.ndarray()):
        v0 = ti.math.vec4([0.0, 1.0, 2.0, 3.0])
        v1 = ti.math.vec4([1.0, 2.0, 3.0, 4.0])
        v2 = ti.math.vec4([2.0, 3.0, 4.0, 5.0])
        v3 = ti.math.vec4([4.0, 5.0, 6.0, 7.0])
        arr[0] = v0
        arr[1] = v1
        arr[2] = v2
        arr[3] = v3

    @ti.kernel
    def vec_test(arr: ti.types.ndarray()):
        v0 = arr[0]
        v1 = arr[1]
        v2 = arr[2]
        v3 = arr[3]

        arr[0] = v0 * v1 + v2
        arr[1] = v1 * v2 + v3
        arr[2] = v0 * v2 + v3

    fill(f)
    vec_test(f)

    assert (
        f.to_numpy()
        == np.array(
            [
                [2.0, 5.0, 10.0, 17.0],
                [6.0, 11.0, 18.0, 27.0],
                [4.0, 8.0, 14.0, 22.0],
                [4.0, 5.0, 6.0, 7.0],
            ]
        )
    ).all()


@test_utils.test(
    require=ti.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_matrix_oob():
    @ti.kernel
    def access_vec(i: ti.i32):
        x = ti.Vector([1, 0])
        x[i] = 42

        # To keep x
        assert x[i] == 42

    @ti.kernel
    def access_mat(i: ti.i32, j: ti.i32):
        y = ti.Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        y[i, j] = 42

        # To keep y
        assert y[i, j] == 42

    # works
    access_vec(1)
    access_mat(2, 2)

    # vector overflow
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_vec(2)
    # vector underflow
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_vec(-1)

    # matrix overflow
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_mat(2, 3)
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_mat(3, 0)
    # matrix underflow
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_mat(-1, 0)

    # TODO: As offset information per dimension is lacking, only the accumulated index is checked. These tests will not raise even if the individual indices are incorrect.
    # with pytest.raises(AssertionError, match=r"Out of bound access"):
    #    access_mat(0, 8)
    # with pytest.raises(AssertionError, match=r"Out of bound access"):
    #    access_mat(-1, 10)
    # with pytest.raises(AssertionError, match=r"Out of bound access"):
    #    access_mat(3, -1)


@pytest.mark.parametrize("dtype", [ti.i32, ti.f32, ti.i64, ti.f64])
@pytest.mark.parametrize("shape", [(8,), (6, 12)])
@pytest.mark.parametrize("offset", [None, 0, -4, 4])
@pytest.mark.parametrize("m, n", [(3, 4)])
@test_utils.test(arch=get_host_arch_list())
def test_matrix_from_numpy_with_offset(dtype, shape, offset, m, n):
    import numpy as np

    x = ti.Matrix.field(
        dtype=dtype, m=m, n=n, shape=shape, offset=[offset] * len(shape) if offset is not None else None
    )
    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        ti.i32: np.int32,
        ti.f32: np.float32,
        ti.f64: np.float64,
        ti.i64: np.int64,
    }
    numpy_shape = ((shape,) if isinstance(shape, int) else shape) + (n, m)
    arr = np.ones(numpy_shape, dtype=numpy_dtypes[dtype])
    x.from_numpy(arr)

    @ti.kernel
    def func():
        for I in ti.grouped(x):
            assert all(abs(I - 1.0) < 1e-6)

    func()


@pytest.mark.parametrize("dtype", [ti.i32, ti.f32, ti.i64, ti.f64])
@pytest.mark.parametrize("shape", [(8,), (6, 12)])
@pytest.mark.parametrize("offset", [0, -4, 4])
@pytest.mark.parametrize("m, n", [(3, 4)])
@test_utils.test(arch=get_host_arch_list())
def test_matrix_to_numpy_with_offset(dtype, shape, offset, m, n):
    import numpy as np

    x = ti.Matrix.field(dtype=dtype, m=m, n=n, shape=shape, offset=[offset] * len(shape))
    x.fill(1.0)
    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        ti.i32: np.int32,
        ti.f32: np.float32,
        ti.f64: np.float64,
        ti.i64: np.int64,
    }
    numpy_shape = ((shape,) if isinstance(shape, int) else shape) + (n, m)
    arr = x.to_numpy()

    assert np.allclose(arr, np.ones(numpy_shape, dtype=numpy_dtypes[dtype]))


@test_utils.test()
def test_matrix_dtype():
    a = ti.types.vector(3, dtype=ti.f32)([0, 1, 2])
    assert a.entries.dtype == np.float32

    b = ti.types.matrix(2, 2, dtype=ti.i32)([[0, 1], [2, 3]])
    assert b.entries.dtype == np.int32


@test_utils.test()
def test_matrix_and_func():
    vec4d = ti.types.vector(4, float)
    v = vec4d(1, 2, 3, 4)

    @ti.func
    def length(w: vec4d):
        return w.norm()

    @ti.kernel
    def test() -> ti.f32:
        return length(v)

    approx(test(), 5.477226)


@test_utils.test()
def test_matrix_loop_unique():
    F_x = ti.Vector.field(3, dtype=ti.f32, shape=10)

    @ti.kernel
    def init():
        for u in F_x:
            F_x[u][1] += 1.0

    init()

    for u in range(10):
        assert F_x[u][1] == 1.0
