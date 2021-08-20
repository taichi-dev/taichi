import taichi as ti


@ti.test(arch=ti.get_host_arch_list())
def test_while():
    assert ti.core.test_threading()
