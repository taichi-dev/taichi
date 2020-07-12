import taichi as ti


@ti.host_arch_only
def test_while():
    assert ti.core.test_threading()
