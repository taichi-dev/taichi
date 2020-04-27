import taichi as ti


def test_cpu_debug_snode_reader():
    ti.init(arch=ti.x64, debug=True)

    x = ti.var(ti.f32, shape=())
    x[None] = 10.0

    assert x[None] == 10.0
