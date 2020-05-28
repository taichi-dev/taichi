import taichi as ti
import numpy as np

@ti.host_arch_only
def test_python_scope_matrix_operations():
    a = ti.Vector([2, 3])
    b = ti.Vector([4, 5])

    c = a + b

    assert np.allclose(c.to_numpy(), np.array([6, 8]))

@ti.host_arch_only
def test_python_scope_matrix_tensor_operations():
    a = ti.Vector(2, dt=ti.i32, shape=())
    b = ti.Vector(2, dt=ti.i32, shape=())

    a[None] = [2, 3]
    b[None] = [4, 5]

    # TODO: hook Matrix.Proxy to redirect to at + Matrix.__add__
    c = a.at(None) + b.at(None)

    assert np.allclose(c.to_numpy(), np.array([6, 8]))
