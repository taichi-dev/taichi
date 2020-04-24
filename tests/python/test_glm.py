import taichi as ti
from taichi import approx


@ti.all_archs
def test_glm_scalar():

    a = ti.var(ti.f32, ())
    b = ti.var(ti.f32, ())
    c = ti.var(ti.f32, ())

    @ti.kernel
    def func():
        a[None] = ti.mix(2.0, 4.0, 0.75)
        b[None] = ti.exp2(2.3)
        c[None] = ti.log2(b[None])

    func()
    assert a[None] == approx(3.5)
    assert b[None] == approx(2**2.3)
    assert c[None] == approx(2.3)


@ti.all_archs
def test_glm_vector():

    u = ti.Vector(3, dt=ti.f32, shape=())
    a = ti.var(ti.f32, ())
    b = ti.var(ti.f32, ())
    v = ti.Vector(3, dt=ti.f32, shape=())

    @ti.kernel
    def func():
        u[None] = ti.vec3(3.0, 4.0, 12.0)
        a[None] = ti.length(u[None])
        b[None] = ti.distance(u[None], ti.vec3(3.0, 1.0, 8.0))
        v[None] = ti.normalize(u[None])

    func()
    assert a[None] == approx(13.0)
    assert b[None] == approx(5.0)
    assert v[None][0] == approx(u[None][0] / 13.0)
