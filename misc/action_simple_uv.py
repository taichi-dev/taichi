import taichi as ti
ti.core.start_recording('record.yml')
ti.init(arch=ti.cc)

n = 512
x = ti.Vector(3, ti.f32, (n, n))


@ti.kernel
def render():
    for i, j in x:
        x[i, j] = [i / x.shape[0], j / x.shape[1], 0]


render()
