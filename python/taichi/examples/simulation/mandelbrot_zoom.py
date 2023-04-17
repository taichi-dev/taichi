import taichi as ti
from taichi.math import cmul, dot, log2, vec2, vec3

ti.init(arch=ti.gpu)

MAXITERS = 100
width, height = 800, 640
pixels = ti.Vector.field(3, ti.f32, shape=(width, height))


@ti.func
def setcolor(z, i):
    v = log2(i + 1 - log2(log2(z.norm()))) / 5
    col = vec3(0.0)
    if v < 1.0:
        col = vec3(v**4, v**2.5, v)
    else:
        v = ti.max(0.0, 2 - v)
        col = vec3(v, v**1.5, v**3)
    return col


@ti.kernel
def render(time: ti.f32):
    zoo = 0.64 + 0.36 * ti.cos(0.02 * time)
    zoo = ti.pow(zoo, 8.0)
    ca = ti.cos(0.15 * (1.0 - zoo) * time)
    sa = ti.sin(0.15 * (1.0 - zoo) * time)
    for i, j in pixels:
        c = 2.0 * vec2(i, j) / height - vec2(1)
        # c *= 1.16
        xy = vec2(c.x * ca - c.y * sa, c.x * sa + c.y * ca)
        c = vec2(-0.745, 0.186) + xy * zoo
        z = vec2(0.0)
        count = 0.0
        while count < MAXITERS and dot(z, z) < 50:
            z = cmul(z, z) + c
            count += 1.0

        if count == MAXITERS:
            pixels[i, j] = [0, 0, 0]
        else:
            pixels[i, j] = setcolor(z, count)


def main():
    gui = ti.GUI("Mandelbrot set zoom", res=(width, height))
    for i in range(100000):
        render(i * 0.03)
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
