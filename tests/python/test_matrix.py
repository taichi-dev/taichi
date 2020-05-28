import taichi as ti
import numpy as np

@ti.host_arch_only
def test_python_scope_matrix_operations():
    a = ti.Vector([2, 3])
    b = ti.Vector([4, 5])

    c = a + b

    assert np.allclose(c.to_numpy(as_vector=True), np.array([6, 8]))
