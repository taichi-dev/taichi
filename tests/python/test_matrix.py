import taichi as ti
import numpy as np
import operator
import pytest

operation_types = [operator.add, operator.sub, operator.matmul]
test_matrix_arrays = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[2, 8], [-1, 3]])]

vector_operation_types = [operator.add, operator.sub]
test_vector_arrays = [
        np.array([42, 42]),
        np.array([24, 24]),
        np.array([83, 12])]


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
# or Python-scope tensor accesses.
# ideally we should use pytest.fixture to parameterize the tests
# over explicit loops
@pytest.mark.parametrize('ops', vector_operation_types)
@ti.host_arch_only
def test_python_scope_vector_tensor(ops):
    t1 = ti.Vector(2, dt=ti.i32, shape=())
    t2 = ti.Vector(2, dt=ti.i32, shape=())
    a, b = test_vector_arrays[:2]
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None].value, t2[None].value)
    assert np.allclose(c.to_numpy(), ops(a, b))


@pytest.mark.parametrize('ops', vector_operation_types)
@ti.host_arch_only
def test_python_scope_matrix_tensor(ops):
    t1 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    t2 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    a, b = test_matrix_arrays[:2]
    # ndarray not supported here
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None].value, t2[None].value)
    print(c)

    assert np.allclose(c.to_numpy(), ops(a, b))


@ti.host_arch_only
def test_constant_matrices():
    print(ti.cos(ti.math.pi / 3))
    print(-ti.Vector([2, 3]))
    print(ti.cos(ti.Vector([2, 3])))
    print(ti.max(2, 3))
    print(ti.max(2, ti.Vector([3, 4, 5])))
    print(ti.Vector([2, 3]) + ti.Vector([3, 4]))
    print(ti.atan2(ti.Vector([2, 3]), ti.Vector([3, 4])))
    print(ti.Matrix([[2, 3], [4, 5]]) @ ti.Vector([2, 3]))
    v = ti.Vector([3, 4])
    w = ti.Vector([5, -12])
    r = ti.Vector([1, 2, 3, 4])
    s = ti.Matrix([[1, 2], [3, 4]])
    print(v.normalized())
    print(v.cross(w))
    w.y = v.x * w[0]
    r.x = r.y
    r.y = r.z
    r.z = r.w
    r.w = r.x
    print(w)
    s[0, 1] = 2
    print(s[0, 1])

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
    r1 = ti.Vector(2, dt=ti.i32, shape=())
    r2 = ti.Vector(2, dt=ti.i32, shape=())
    m3 = ti.Vector(2, dt=ti.i32, shape=())
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
    r1 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    r2 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    m3 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    m3.from_numpy(c)

    @ti.kernel
    def run():
        r1[None] = ops(m1, m2)
        r2[None] = ops(m1, m3[None])

    run()

    assert np.allclose(r1[None].value.to_numpy(), ops(a, b))
    assert np.allclose(r2[None].value.to_numpy(), ops(a, c))
