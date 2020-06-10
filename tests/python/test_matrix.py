import taichi as ti
import numpy as np
import operator
import pytest

operation_types = (operator.add, operator.sub, operator.matmul)
test_matrix_arrays = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))

vector_operation_types = (operator.add, operator.sub)
test_vector_arrays = (np.array([42, 42]), np.array([24, 24]))


@ti.host_arch_only
def test_python_scope_vector_operations():
    for ops in vector_operation_types:
        a, b = test_vector_arrays
        m1, m2 = ti.Vector(a), ti.Vector(b)
        c = ops(m1, m2)
        assert np.allclose(c.to_numpy(), ops(a, b))


@ti.host_arch_only
def test_python_scope_matrix_operations():
    for ops in operation_types:
        a, b = test_matrix_arrays
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
    a, b = test_vector_arrays
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None].value, t2[None].value)
    assert np.allclose(c.to_numpy(), ops(a, b))


@pytest.mark.parametrize('ops', vector_operation_types)
@ti.host_arch_only
def test_python_scope_matrix_tensor(ops):
    t1 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    t2 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    a, b = test_matrix_arrays
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
    print(v.normalized())
    print(v.cross(w))
    w.y = v.x * w[0]
    print(w)

    @ti.kernel
    def func(t: ti.i32):
        m = ti.Matrix([[2, 3], [4, t]])
        print(m @ ti.Vector([2, 3]))
        m += ti.Matrix([[3, 4], [5, t]])
        print(m @ v)
        s = w.transpose() @ m
        print(s)
        print(m)

    func(5)


@pytest.mark.parametrize('ops', vector_operation_types)
@ti.host_arch_only
def test_taichi_scope_vector_operations_with_global_vectors(ops):
    a, b = test_vector_arrays
    m1, m2 = ti.Vector(a), ti.Vector(b)
    c = ti.Vector(2, dt=ti.i32, shape=())

    @ti.kernel
    def run():
        c[None] = ops(m1, m2)

    run()

    assert np.allclose(c[None].value.to_numpy(), ops(a, b))


@pytest.mark.parametrize('ops', vector_operation_types)
@ti.host_arch_only
def test_taichi_scope_matrix_operations_with_global_matrices(ops):
    a, b = test_matrix_arrays
    m1, m2 = ti.Matrix(a), ti.Matrix(b)
    c = ti.Matrix(2, 2, dt=ti.i32, shape=())

    @ti.kernel
    def run():
        c[None] = ops(m1, m2)

    run()

    assert np.allclose(c[None].value.to_numpy(), ops(a, b))
