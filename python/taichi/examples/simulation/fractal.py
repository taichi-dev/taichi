import taichi as ti

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


def main():
    gui = ti.GUI("Julia Set", res=(n * 2, n))
    t = 0.0
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        paint(t)
        t += 0.03
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
