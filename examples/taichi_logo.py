import taichi as ti
from renderer_utils import inside_taichi

n = 512
x = ti.var(ti.f32, shape=[n, n])


@ti.kernel
def paint():
    for i, j in ti.ndrange(n * 4, n * 4):
        # 4x4 super sampling:
        ret = 1 - inside_taichi(ti.Vector([i / n / 4, j / n / 4]))
        x[i // 4, j // 4] += ret / 16


paint()

gui = ti.GUI('Logo', (512, 512))
while gui.running:
    gui.set_image(x.to_numpy())
    gui.show()
