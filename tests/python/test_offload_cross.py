import taichi as ti


@ti.all_archs
def test_offload_with_cross_block_locals():
    ret = ti.var(ti.f32)

    @ti.layout
    def place():
        ti.root.place(ret)

    @ti.kernel
    def ker():
        s = 0
        for i in range(10):
            s += i
        ret[None] = s

    ker()

    assert ret[None] == 45


@ti.all_archs
def test_offload_with_cross_block_locals2():
    ret = ti.var(ti.f32)

    @ti.layout
    def place():
        ti.root.place(ret)

    @ti.kernel
    def ker():
        s = 0
        for i in range(10):
            s += i
        ret[None] = s
        s = ret[None] * 2
        for i in range(10):
            ti.atomic_add(ret[None], s)

    ker()

    assert ret[None] == 45 * 21


@ti.all_archs
def test_offload_with_cross_block_locals2():
    ret = ti.var(ti.f32, shape=())

    @ti.kernel
    def ker():
        s = 1
        t = s
        for i in range(10):
            s += i
        ret[None] = t

    ker()

    assert ret[None] == 1


@ti.archs_excluding(ti.opengl)  # OpenGL doesn't support dynamic range for now
def test_offload_with_flexible_bounds():
    s = ti.var(ti.i32, shape=())
    lower = ti.var(ti.i32, shape=())
    upper = ti.var(ti.i32, shape=())

    @ti.kernel
    def ker():
        for i in range(lower[None], upper[None]):
            s[None] += i

    lower[None] = 10
    upper[None] = 20
    ker()

    assert s[None] == 29 * 10 // 2
