import taichi as ti
import numpy as np
import operator

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
@ti.host_arch_only
def test_python_scope_vector_tensor_add():
    t1 = ti.Vector(2, dt=ti.i32, shape=())
    t2 = ti.Vector(2, dt=ti.i32, shape=())
    a, b = test_vector_arrays
    t1[None], t2[None] = a, b

    # TODO: hook Matrix.Proxy to redirect to at + Matrix.__add__
    c = t1.at(None) + t2.at(None)
    assert np.allclose(c.to_numpy(), a + b)


@ti.host_arch_only
def test_python_scope_vector_tensor_sub():
    t1 = ti.Vector(2, dt=ti.i32, shape=())
    t2 = ti.Vector(2, dt=ti.i32, shape=())
    a, b = test_vector_arrays
    t1[None], t2[None] = a, b

    # TODO: hook Matrix.Proxy to redirect to at + Matrix.__sub__
    c = t1.at(None) - t2.at(None)
    assert np.allclose(c.to_numpy(), a - b)


@ti.host_arch_only
def test_python_scope_matrix_tensor_add():
    t1 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    t2 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    a, b = test_matrix_arrays
    # ndarray not supported here
    t1[None], t2[None] = a.tolist(), b.tolist()

    # TODO: hook Matrix.Proxy to redirect to at + Matrix.__add__
    c = t1.at(None) + t2.at(None)
    print(c)

    assert np.allclose(c.to_numpy(), a + b)


@ti.host_arch_only
def test_python_scope_matrix_tensor_sub():
    t1 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    t2 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    a, b = test_matrix_arrays
    # ndarray not supported here
    t1[None], t2[None] = a.tolist(), b.tolist()

    # TODO: hook Matrix.Proxy to redirect to at + Matrix.__sub__
    c = t1.at(None) - t2.at(None)
    assert np.allclose(c.to_numpy(), a - b)


@ti.host_arch_only
def test_python_scope_matrix_tensor_matmul():
    t1 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    t2 = ti.Matrix(2, 2, dt=ti.i32, shape=())
    a, b = test_matrix_arrays
    # ndarray not supported here
    t1[None], t2[None] = a.tolist(), b.tolist()

    # TODO: hook Matrix.Proxy to redirect to at + Matrix.__matmul__
    c = t1.at(None) @ t2.at(None)
    assert np.allclose(c.to_numpy(), a @ b)
