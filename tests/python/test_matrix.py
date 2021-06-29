import operator

import numpy as np
import pytest

import taichi as ti
from taichi import approx

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


@ti.host_arch_only
def test_python_scope_vector_operations():
    for ops in vector_operation_types:
        a, b = test_vector_arrays[:2]
        m1, m2 = ti.Vector(a), ti.Vector(b)
        c = ops(m1, m2)
        assert np.allclose(c.to_numpy(), ops(a, b))


@ti.host_arch_only
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
@ti.host_arch_only
def test_python_scope_vector_field(ops):
    t1 = ti.Vector.field(2, dtype=ti.i32, shape=())
    t2 = ti.Vector.field(2, dtype=ti.i32, shape=())
    a, b = test_vector_arrays[:2]
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None].value, t2[None].value)
    assert np.allclose(c.to_numpy(), ops(a, b))


@pytest.mark.parametrize('ops', vector_operation_types)
@ti.host_arch_only
def test_python_scope_matrix_field(ops):
    t1 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    t2 = ti.Matrix.field(2, 2, dtype=ti.i32, shape=())
    a, b = test_matrix_arrays[:2]
    # ndarray not supported here
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None].value, t2[None].value)
    print(c)

    assert np.allclose(c.to_numpy(), ops(a, b))


@ti.host_arch_only
def test_constant_matrices():
    assert ti.cos(ti.math.pi / 3) == approx(0.5)
    assert np.allclose((-ti.Vector([2, 3])).to_numpy(), np.array([-2, -3]))
    assert ti.cos(ti.Vector([2,
                             3])).to_numpy() == approx(np.cos(np.array([2,
                                                                        3])))
    assert ti.max(2, 3) == 3
    res = ti.max(4, ti.Vector([3, 4, 5]))
    assert np.allclose(res.to_numpy(), np.array([4, 4, 5]))
    res = ti.Vector([2, 3]) + ti.Vector([3, 4])
    assert np.allclose(res.to_numpy(), np.array([5, 7]))
    res = ti.atan2(ti.Vector([2, 3]), ti.Vector([3, 4]))
    assert res.to_numpy() == approx(
        np.arctan2(np.array([2, 3]), np.array([3, 4])))
    res = ti.Matrix([[2, 3], [4, 5]]) @ ti.Vector([2, 3])
    assert np.allclose(res.to_numpy(), np.array([13, 23]))
    v = ti.Vector([3, 4])
    w = ti.Vector([5, -12])
    r = ti.Vector([1, 2, 3, 4])
    s = ti.Matrix([[1, 2], [3, 4]])
    assert v.normalized().to_numpy() == approx(np.array([0.6, 0.8]))
    assert v.cross(w) == approx(-12 * 3 - 4 * 5)
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
@ti.host_arch_only
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
@ti.host_arch_only
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


@ti.test(arch=ti.cpu)
def test_matrix_non_constant_index():
    m = ti.Matrix.field(2, 2, ti.i32, 5)

    @ti.kernel
    def func():
        for i in range(5):
            for j, k in ti.ndrange(2, 2):
                m[i][j, k] = 12

    with pytest.raises(ti.TaichiSyntaxError):
        func()


@ti.test(arch=ti.cpu)
def test_matrix_constant_index():
    m = ti.Matrix.field(2, 2, ti.i32, 5)

    @ti.kernel
    def func():
        for i in range(5):
            for j, k in ti.static(ti.ndrange(2, 2)):
                m[i][j, k] = 12

    func()

    assert np.allclose(m.to_numpy(), np.ones((5, 2, 2), np.int32) * 12)


@ti.test(arch=ti.cpu)
def test_vector_to_list():
    a = ti.Vector.field(2, float, ())

    data = [2, 3]
    b = ti.Vector(data)
    assert list(b) == data
    assert len(b) == len(data)

    a[None] = b
    assert all(a[None].value == ti.Vector(data))


@ti.test(arch=ti.cpu)
def test_matrix_to_list():
    a = ti.Matrix.field(2, 3, float, ())

    data = [[2, 3, 4], [5, 6, 7]]
    b = ti.Matrix(data)
    assert list(b) == data
    assert len(b) == len(data)

    a[None] = b
    assert all(a[None].value == ti.Matrix(data))


@ti.test()
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
