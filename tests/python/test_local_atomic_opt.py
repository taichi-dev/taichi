import taichi as ti


@ti.all_archs
def test_cse():
    A = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 0
        a += 10
        a = a + 123
        A[None] = a

    func()
    assert A[None] == 133


@ti.all_archs
def test_store_forward():
    A = ti.field(ti.f32, shape=())

    @ti.kernel
    def func():
        a = 0
        a = 123
        a += 10
        A[None] = a

    func()
    assert A[None] == 133
