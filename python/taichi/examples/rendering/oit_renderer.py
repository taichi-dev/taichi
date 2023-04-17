# A demo of order-independent transparency using Taichi
# Reference: https://github.com/nvpro-samples/vk_order_independent_transparency
import taichi as ti
from taichi.math import clamp, mix, normalize, vec3, vec4

ti.init(arch=ti.cuda)
res = (1000, 1000)
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)

eps = 1e-4
inf = 1e10
fov = 0.5
aspect_ratio = res[0] / res[1]

alpha_min = 0.2
alpha_width = 0.3
camera_pos = vec3(0, 0, 5)

Line = ti.types.struct(pos=vec3, dir=vec3)

background_color = vec4(0.2, 0.2, 0.2, 1)

Sphere = ti.types.struct(center=vec3, radius=float, color=vec4)
spheres = Sphere.field()
ti.root.dynamic(ti.j, 1024, chunk_size=64).place(spheres)

ColorWithDepth = ti.types.struct(color=vec4, depth=float)
colors_in_pixel = ColorWithDepth.field()
ti.root.dense(ti.ij, res).dynamic(ti.k, 2048, chunk_size=64).place(colors_in_pixel)


@ti.func
def gooch_lighting(normal: ti.template()):
    light = normalize(vec3(-1, 2, 1))
    warmth = normal * light * 0.5 + 0.5
    return mix(vec3(0, 0.25, 0.75), vec3(1, 1, 1), warmth)


@ti.func
def shading(color: ti.template(), normal: ti.template()):
    colorRGB = color.rgb * gooch_lighting(normal)
    alpha = clamp(alpha_min + color.a * alpha_width, 0, 1)
    return vec4(colorRGB, alpha)


@ti.func
def intersect_sphere(line: ti.template(), sphere: ti.template()):
    color1 = vec4(0)
    color2 = vec4(0)
    dist1 = inf
    dist2 = inf
    l = sphere.center - line.pos
    l2 = l.dot(l)
    r2 = sphere.radius * sphere.radius
    tp = l.dot(line.dir)
    out_of_sphere = l2 > r2
    may_have_intersection = True
    if -eps < l2 - r2 < eps:
        if -eps < tp < eps:
            may_have_intersection = False
        out_of_sphere = tp < 0
    if tp < 0 and out_of_sphere:
        may_have_intersection = False
    if may_have_intersection:
        d2 = l2 - tp * tp
        if d2 <= r2:
            tt = ti.sqrt(r2 - d2)
            t1 = tp - tt
            if t1 > 0:
                hit_pos1 = line.pos + line.dir * t1
                dist1 = t1
                normal1 = normalize(hit_pos1 - sphere.center)
                color1 = shading(sphere.color, normal1)
            t2 = tp + tt
            if t2 > 0:
                hit_pos2 = line.pos + line.dir * t2
                dist2 = t2
                normal2 = normalize(hit_pos2 - sphere.center)
                color2 = shading(sphere.color, normal2)
    return ColorWithDepth(color=color1, depth=dist1), ColorWithDepth(color=color2, depth=dist2)


@ti.func
def get_line_of_vision(u, v):
    ray_dir = vec3(
        (2 * (u + 0.5) / res[0] - 1) * aspect_ratio,
        (2 * (v + 0.5) / res[1] - 1),
        -1.0 / fov,
    )
    ray_dir = normalize(ray_dir)
    return Line(pos=camera_pos, dir=ray_dir)


@ti.func
def get_intersections(u, v):
    line = get_line_of_vision(u, v)
    colors_in_pixel[u, v].deactivate()
    for i in range(spheres.length()):
        hit1, hit2 = intersect_sphere(line, spheres[i])
        if hit1.depth < inf:
            colors_in_pixel[u, v].append(hit1)
        if hit2.depth < inf:
            colors_in_pixel[u, v].append(hit2)


@ti.func
def bubble_sort(u, v):
    l = colors_in_pixel[u, v].length()
    for i in range(l - 1):
        for j in range(l - 1 - i):
            if colors_in_pixel[u, v, j].depth > colors_in_pixel[u, v, j + 1].depth:
                tmp = colors_in_pixel[u, v, j]
                colors_in_pixel[u, v, j] = colors_in_pixel[u, v, j + 1]
                colors_in_pixel[u, v, j + 1] = tmp


@ti.func
def blend(color: ti.template(), base_color: ti.template()):
    color.rgb += (1 - color.a) * base_color.rgb * base_color.a
    color.a += (1 - color.a) * base_color.a


@ti.func
def get_color(u, v):
    color = vec4(0)
    for i in range(colors_in_pixel[u, v].length()):
        blend(color, colors_in_pixel[u, v, i].color)
    blend(color, background_color)
    gamma_corrected = ti.pow(color.rgb * color.a, 1 / 2.2)
    color_buffer[u, v] = gamma_corrected


stratify_res = 5
inv_stratify = 1.0 / stratify_res


@ti.kernel
def generate_sphere(n: ti.i32):
    for i in range(n):
        spheres.append(
            Sphere(
                vec3(ti.random() * 3 - 1.5, ti.random() * 3 - 1.5, ti.random() * 3 - 1.5),
                ti.random() * 0.2 + 0.1,
                (ti.random(), ti.random(), ti.random(), ti.random()),
            )
        )


@ti.kernel
def render():
    for u, v in color_buffer:
        get_intersections(u, v)
        bubble_sort(u, v)
        get_color(u, v)


@ti.kernel
def init():
    spheres.deactivate()
    for u, v in color_buffer:
        colors_in_pixel[u, v].deactivate()


def main():
    gui = ti.GUI("OIT", res, fast_gui=True)
    init()
    generate_sphere(256)
    render()
    gui.set_image(color_buffer)
    while gui.running:
        gui.show()


if __name__ == "__main__":
    main()
