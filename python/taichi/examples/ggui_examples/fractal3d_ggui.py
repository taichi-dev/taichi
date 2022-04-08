import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)


@ti.func
def quat_mul(v1, v2):
    return ti.Vector([
        v1.x * v2.x - v1.y * v2.y - v1.z * v2.z - v1.w * v2.w,
        v1.x * v2.y + v1.y * v2.x + v1.z * v2.w - v1.w * v2.z,
        v1.x * v2.z + v1.z * v2.x + v1.w * v2.y - v1.y * v2.w,
        v1.x * v2.w + v1.w * v2.x + v1.y * v2.z - v1.z * v2.y
    ])


@ti.func
def quat_conj(q):
    return ti.Vector([q[0], -q[1], -q[2], -q[3]])


@ti.func
def dot(x, y):
    return x.dot(y)


@ti.func
def xy(v):
    return ti.Vector([v.x, v.y])


@ti.func
def yx(v):
    return ti.Vector([v.y, v.x])


@ti.func
def xz(v):
    return ti.Vector([v.x, v.z])


@ti.func
def zx(v):
    return ti.Vector([v.z, v.x])


@ti.func
def xw(v):
    return ti.Vector([v.x, v.w])


@ti.func
def wx(v):
    return ti.Vector([v.w, v.x])


@ti.func
def xyz(v):
    return ti.Vector([v.x, v.y, v.z])


iters = 10
max_norm = 4


@ti.func
def compute_sdf(z, c):

    md2 = 1.0
    mz2 = dot(z, z)

    for iter in range(iters):
        md2 *= max_norm * mz2
        z = quat_mul(z, z) + c

        mz2 = z.dot(z)
        if (mz2 > max_norm):
            break

    return 0.25 * ti.sqrt(mz2 / md2) * ti.log(mz2)


@ti.func
def compute_normal(z, c):
    J0 = ti.Vector([1.0, 0.0, 0.0, 0.0])
    J1 = ti.Vector([0.0, 1.0, 0.0, 0.0])
    J2 = ti.Vector([0.0, 0.0, 1.0, 0.0])

    z_curr = z

    iterations = 0
    while z_curr.norm() < max_norm and iterations < iters:
        cz = quat_conj(z_curr)

        J0 = ti.Vector([
            dot(J0, cz),
            dot(xy(J0), yx(z_curr)),
            dot(xz(J0), zx(z_curr)),
            dot(xw(J0), wx(z_curr))
        ])
        J1 = ti.Vector([
            dot(J1, cz),
            dot(xy(J1), yx(z_curr)),
            dot(xz(J1), zx(z_curr)),
            dot(xw(J1), wx(z_curr))
        ])
        J2 = ti.Vector([
            dot(J2, cz),
            dot(xy(J2), yx(z_curr)),
            dot(xz(J2), zx(z_curr)),
            dot(xw(J2), wx(z_curr))
        ])

        z_curr = quat_mul(z_curr, z_curr) + c
        iterations += 1

    return ti.Vector([dot(J0, z_curr),
                      dot(J1, z_curr),
                      dot(J2, z_curr)]).normalized()


image_res = (1280, 720)


@ti.data_oriented
class Julia:
    def __init__(self):
        self.image = ti.Vector.field(3, float, image_res)

    @ti.func
    def shade(self, pos, surface_color, normal, light_pos):
        light_color = ti.Vector([1, 1, 1])

        light_dir = (light_pos - pos).normalized()
        return light_color * surface_color * max(0, dot(light_dir, normal))

    @ti.kernel
    def march(self, time_arg: float):
        time = time_arg * 0.15
        c = 0.45 * ti.cos(
            ti.Vector([0.5, 3.9, 1.4, 1.1]) + time *
            ti.Vector([1.2, 1.7, 1.3, 2.5])) - ti.Vector([0.3, 0.0, 0.0, 0.0])

        r = 1.8
        o3 = ti.Vector([
            r * ti.cos(0.3 + 0.37 * time), 0.3 +
            0.8 * r * ti.cos(1.0 + 0.33 * time), r * ti.cos(2.2 + 0.31 * time)
        ]).normalized() * r
        ta = ti.Vector([0.0, 0.0, 0.0])
        cr = 0.1 * ti.cos(0.1 * time)

        for x, y in self.image:
            p = (-ti.Vector([image_res[0], image_res[1]]) +
                 2.0 * ti.Vector([x, y])) / (image_res[1] * 0.75)

            cw = (ta - o3).normalized()
            cp = ti.Vector([ti.sin(cr), ti.cos(cr), 0.0])
            cu = cw.cross(cp).normalized()
            cv = cu.cross(cw).normalized()

            d3 = (p.x * cu + p.y * cv + 2.0 * cw).normalized()

            o = ti.Vector([o3.x, o3.y, o3.z, 0.0])
            d = ti.Vector([d3.x, d3.y, d3.z, 0.0])

            max_t = 10

            t = 0.0
            for step in range(300):
                h = compute_sdf(o + t * d, c)
                t += h
                if h < 0.0001 or t >= max_t:
                    break
            if t < max_t:
                normal = compute_normal(o + t * d, c)
                color = abs(xyz(o + t * d)) / 1.3
                pos = xyz(o + t * d)
                self.image[x, y] = self.shade(pos, color, normal, o3)
            else:
                self.image[x, y] = (0, 0, 0)

    def get_image(self, time):
        self.march(time)
        return self.image


julia = Julia()

window = ti.ui.Window("Fractal 3D", image_res, vsync=True)
canvas = window.get_canvas()

frame_id = 0

while window.running:
    frame_id += 1

    canvas.set_image(julia.get_image(frame_id / 60))

    window.show()
