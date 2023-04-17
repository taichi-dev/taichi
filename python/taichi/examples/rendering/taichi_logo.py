from taichi.examples.patterns import taichi_logo

import taichi as ti

ti.init()

n = 512
x = ti.field(dtype=ti.f32, shape=(n, n))


@ti.kernel
def paint():
    for i, j in ti.ndrange(n * 4, n * 4):
        # 4x4 super sampling:
        ret = taichi_logo(ti.Vector([i, j]) / (n * 4))
        x[i // 4, j // 4] += ret / 16


def main():
    paint()

    gui = ti.GUI("Logo", (n, n))
    while gui.running:
        gui.set_image(x)
        gui.show()


if __name__ == "__main__":
    main()
