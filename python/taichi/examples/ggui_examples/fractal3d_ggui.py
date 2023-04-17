import taichi as ti
import taichi.math as tm

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

vec3 = tm.vec3
vec4 = tm.vec4


@ti.func
def quat_mul(v1, v2):
    return vec4(
        v1.x * v2.x - tm.dot(v1.yzw, v2.yzw),
        v1.x * v2.yzw + v2.x * v1.yzw + tm.cross(v1.yzw, v2.yzw),
    )


@ti.func
def quat_conj(q):
    return vec4(q.x, -q.yzw)


iters = 10
max_norm = 4


@ti.func
def compute_sdf(z, c):
    md2 = 1.0
    mz2 = tm.dot(z, z)

    for _ in range(iters):
        md2 *= max_norm * mz2
        z = quat_mul(z, z) + c

        mz2 = z.dot(z)
        if mz2 > max_norm:
            break

    return 0.25 * ti.sqrt(mz2 / md2) * ti.log(mz2)


@ti.func
def compute_normal(z, c):
    J0 = vec4(1, 0, 0, 0)
    J1 = vec4(0, 1, 0, 0)
    J2 = vec4(0, 0, 1, 0)

    z_curr = z

    iterations = 0
    while z_curr.norm() < max_norm and iterations < iters:
        cz = quat_conj(z_curr)

        J0 = vec4(
            tm.dot(J0, cz),
            tm.dot(J0.xy, z_curr.yx),
            tm.dot(J0.xz, z_curr.zx),
            tm.dot(J0.xw, z_curr.wx),
        )
        J1 = vec4(
            tm.dot(J1, cz),
            tm.dot(J1.xy, z_curr.yx),
            tm.dot(J1.xz, z_curr.zx),
            tm.dot(J1.xw, z_curr.wx),
        )
        J2 = vec4(
            tm.dot(J2, cz),
            tm.dot(J2.xy, z_curr.yx),
            tm.dot(J2.xz, z_curr.zx),
            tm.dot(J2.xw, z_curr.wx),
        )

        z_curr = quat_mul(z_curr, z_curr) + c
        iterations += 1

    return tm.normalize(tm.vec3(tm.dot(z_curr, J0), tm.dot(z_curr, J1), tm.dot(z_curr, J2)))


image_res = (1280, 720)


@ti.data_oriented
class Julia:
    def __init__(self):
        self.image = ti.Vector.field(3, float, image_res)

    @ti.func
    def shade(self, pos, surface_color, normal, light_pos):
        _ = self  # make pylint happy
        light_color = vec3(1)

        light_dir = tm.normalize(light_pos - pos)
        return light_color * surface_color * ti.max(0, tm.dot(light_dir, normal))

    @ti.kernel
    def march(self, time_arg: float):
        time = time_arg * 0.15
        c = 0.45 * ti.cos(vec4(0.5, 3.9, 1.4, 1.1) + time * vec4(1.2, 1.7, 1.3, 2.5)) - vec4(0.3, 0, 0, 0)

        r = 1.8
        o3 = (
            tm.normalize(
                vec3(
                    r * ti.cos(0.3 + 0.37 * time),
                    0.3 + 0.8 * r * ti.cos(1.0 + 0.33 * time),
                    r * ti.cos(2.2 + 0.31 * time),
                )
            )
            * r
        )
        ta = vec3(0)
        cr = 0.1 * ti.cos(0.1 * time)

        for x, y in self.image:
            p = (-tm.vec2(image_res) + 2.0 * tm.vec2(x, y)) / (image_res[1] * 0.75)

            cw = tm.normalize(ta - o3)
            cp = vec3(ti.sin(cr), ti.cos(cr), 0)
            cu = tm.normalize(cw.cross(cp))
            cv = tm.normalize(cu.cross(cw))

            d3 = tm.normalize(p.x * cu + p.y * cv + 2.0 * cw)

            o = vec4(o3, 0)
            d = vec4(d3, 0)

            max_t = 10

            t = 0.0
            for step in range(300):
                h = compute_sdf(o + t * d, c)
                t += h
                if h < 0.0001 or t >= max_t:
                    break
            if t < max_t:
                normal = compute_normal(o + t * d, c)
                color = abs((o + t * d).xyz) / 1.3
                pos = (o + t * d).xyz
                self.image[x, y] = self.shade(pos, color, normal, o3)
            else:
                self.image[x, y] = (0, 0, 0)

    def get_image(self, time):
        self.march(time)
        return self.image


def main():
    julia = Julia()

    window = ti.ui.Window("Fractal 3D", image_res, vsync=True)
    canvas = window.get_canvas()

    frame_id = 0

    while window.running:
        frame_id += 1
        canvas.set_image(julia.get_image(frame_id / 60))
        window.show()


if __name__ == "__main__":
    main()
