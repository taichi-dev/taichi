import taichi as ti


@ti.host_arch
def test_while():
  assert ti.core.test_threading()
