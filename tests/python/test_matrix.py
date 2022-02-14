import math
import operator

import numpy as np
import pytest
from taichi.lang import impl
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

    c = ops(t1[None].value, t2[None].value)
    assert np.allclose(c.to_numpy(), ops(a, b))


@pytest.mark.parametrize('ops', vector_operation_types)
@test_utils.test(arch=get_host_arch_list())
def test_python_scope_matrix_field(ops):
    t1 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    t2 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    a, b = test_matrix_arrays[:2]
    # ndarray not supported here
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None].value, t2[None].value)
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
        s = w.transpose() @ m
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

    assert np.allclose(r1[None].value.to_numpy(), ops(a, b))
    assert np.allclose(r2[None].value.to_numpy(), ops(a, c))


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

    assert np.allclose(r1[None].value.to_numpy(), ops(a, b))
    assert np.allclose(r2[None].value.to_numpy(), ops(a, c))


@test_utils.test()
def test_matrix_non_constant_index_numpy():
    @ti.kernel
    def func1(a: ti.any_arr(element_dim=2)):
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
    def func2(b: ti.any_arr(element_dim=1, layout=ti.Layout.SOA)):
        for i in range(5):
            for j in range(4):
                b[i][j * j] = j * j

    v = np.empty((10, 5), dtype=np.int32)
    func2(v)
    assert v[0][1] == 0
    assert v[1][1] == 1
    assert v[4][1] == 4
    assert v[9][1] == 9


@test_utils.test(require=ti.extension.dynamic_index,
                 dynamic_index=True,
                 debug=True)
def test_matrix_non_constant_index():
    m = ti.Matrix.field(2, 2, ti.i32, 5)
    v = ti.Vector.field(10, ti.i32, 5)

    @ti.kernel
    def func1():
        for i in range(5):
            for j, k in ti.ndrange(2, 2):
                m[i][j, k] = j * j + k * k
        assert m[1][0, 1] == 1
        assert m[2][1, 0] == 1
        assert m[3][1, 1] == 2

    func1()
    assert m[4][0, 1] == 1

    @ti.kernel
    def func2():
        for i in range(5):
            for j in range(4):
                v[i][j * j] = j * j
        assert v[1][0] == 0
        assert v[1][1] == 1
        assert v[1][4] == 4

    func2()
    assert v[1][9] == 9

    @ti.kernel
    def func3():
        tmp = ti.Vector([1, 2, 3])
        for i in range(3):
            tmp[i] = i * i
            vec = ti.Vector([4, 5, 6])
            for j in range(3):
                vec[tmp[i] % 3] += vec[j % 3]
        assert tmp[0] == 0
        assert tmp[1] == 1
        assert tmp[2] == 4

    func3()

    @ti.kernel
    def func4(k: ti.i32):
        tmp = ti.Vector([k, k * 2, k * 3])
        assert tmp[0] == k
        assert tmp[1] == k * 2
        assert tmp[2] == k * 3

    func4(10)


@test_utils.test(arch=ti.cpu)
def test_matrix_constant_index():
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
    assert all(a[None].value == ti.Vector(data))


@test_utils.test(arch=ti.cpu)
def test_matrix_to_list():
    a = ti.Matrix.field(2, 3, float, ())

    data = [[2, 3, 4], [5, 6, 7]]
    b = ti.Matrix(data)
    assert list(b) == data
    assert len(b) == len(data)

    a[None] = b
    assert all(a[None].value == ti.Matrix(data))


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
            assert ti.get_addr(y, i) - ti.get_addr(x,
                                                   i) == v.dynamic_index_stride
            assert ti.get_addr(z, i) - ti.get_addr(y,
                                                   i) == v.dynamic_index_stride

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
    assert v.dynamic_index_stride is None


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_matrix_field_dynamic_index_not_pure_dense():
    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 2).pointer(ti.i, 4).place(x)
    ti.root.dense(ti.i, 2).dense(ti.i, 4).place(y)

    impl.get_runtime().materialize()
    assert v.dynamic_index_stride is None


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_matrix_field_dynamic_index_different_cell_size_bytes():
    temp = ti.field(ti.f32)

    v = ti.Vector.field(2, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    ti.root.dense(ti.i, 8).place(x, temp)
    ti.root.dense(ti.i, 8).place(y)

    impl.get_runtime().materialize()
    assert v.dynamic_index_stride is None


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
    assert v.dynamic_index_stride is None


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_matrix_field_dynamic_index_different_stride():
    temp = ti.field(ti.f32)

    v = ti.Vector.field(3, ti.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)
    z = v.get_scalar_field(2)

    ti.root.dense(ti.i, 8).place(x, y, temp, z)

    impl.get_runtime().materialize()
    assert v.dynamic_index_stride is None


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
