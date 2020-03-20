import taichi as ti


@ti.archs_excluding(ti.metal, ti.opengl)
def test_ad_demote_dense():
    ti.get_runtime().print_preprocessed = True
    a = ti.var(ti.f32, shape=(7, 3, 19))

    @ti.kernel
    def inc():
        for i, j, k in a:
            a[i, j, k] += 1

    inc.grad()
