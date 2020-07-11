import taichi as ti


@ti.test(ti.cpu)
def test_while():
    assert ti.core.test_threading()
