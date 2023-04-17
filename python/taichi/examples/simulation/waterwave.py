# Water wave effect partially based on shallow water equations
# https://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form

import taichi as ti

ti.init(arch=ti.gpu)

light_color = 1
gravity = 2.0  # larger gravity makes wave propagates faster
damping = 0.2  # larger damping makes wave vanishes faster when propagating
dx = 0.02
dt = 0.01
shape = 512, 512
pixels = ti.field(dtype=float, shape=shape)
background = ti.field(dtype=float, shape=shape)
height = ti.field(dtype=float, shape=shape)
velocity = ti.field(dtype=float, shape=shape)


@ti.kernel
def reset():
    for i, j in height:
        t = i // 16 + j // 16
        if t % 2 == 0:
            background[i, j] = 0.0
        else:
            background[i, j] = 0.25
        height[i, j] = 0
        velocity[i, j] = 0


@ti.func
def laplacian(i, j):
    return (-4 * height[i, j] + height[i, j - 1] + height[i, j + 1] + height[i + 1, j] + height[i - 1, j]) / (
        4 * dx**2
    )


@ti.func
def gradient(i, j):
    return ti.Vector(
        [
            (height[i + 1, j] if i < shape[0] - 1 else 0) - (height[i - 1, j] if i > 1 else 0),
            (height[i, j + 1] if j < shape[1] - 1 else 0) - (height[i, j - 1] if j > 1 else 0),
        ]
    ) * (0.5 / dx)


@ti.kernel
def create_wave(amplitude: ti.f32, x: ti.f32, y: ti.f32):
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        r2 = (i - x) ** 2 + (j - y) ** 2
        height[i, j] = height[i, j] + amplitude * ti.exp(-0.02 * r2)


@ti.kernel
def update():
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        acceleration = gravity * laplacian(i, j) - damping * velocity[i, j]
        velocity[i, j] = velocity[i, j] + acceleration * dt

    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        height[i, j] = height[i, j] + velocity[i, j] * dt


@ti.kernel
def visualize_wave():
    # visualizes the wave using a fresnel-like shading
    # a brighter color indicates a steeper wave
    # (closer to grazing angle when looked from above)
    for i, j in pixels:
        g = gradient(i, j)
        cos_i = 1 / ti.sqrt(1 + g.norm_sqr())
        brightness = pow(1 - cos_i, 2)
        color = background[i, j]
        pixels[i, j] = (1 - brightness) * color + brightness * light_color


def main():
    print("[Hint] click on the window to create waves")

    reset()
    gui = ti.GUI("Water Wave", shape)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                gui.running = False
            elif e.key == "r":
                reset()
            elif e.key == ti.GUI.LMB:
                x, y = e.pos
                create_wave(3, x * shape[0], y * shape[1])
        update()
        visualize_wave()
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
