import math
import time

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)
res = 1280, 720
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
max_ray_depth = 6
eps = 1e-4
inf = 1e10

fov = 0.23
dist_limit = 100

camera_pos = ti.Vector([0.0, 0.32, 3.7])
light_pos = [-1.5, 0.6, 0.3]
light_normal = [1.0, 0.0, 0.0]
light_radius = 2.0


@ti.func
def intersect_light(pos, d):
    light_loc = ti.Vector(light_pos)
    dot = -d.dot(ti.Vector(light_normal))
    dist = d.dot(light_loc - pos)
    dist_to_light = inf
    if dot > 0 and dist > 0:
        D = dist / dot
        dist_to_center = (light_loc - (pos + D * d)).norm_sqr()
        if dist_to_center < light_radius**2:
            dist_to_light = D
    return dist_to_light


@ti.func
def out_dir(n):
    u = ti.Vector([1.0, 0.0, 0.0])
    if abs(n[1]) < 1 - eps:
        u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    v = n.cross(u)
    phi = 2 * math.pi * ti.random()
    ay = ti.sqrt(ti.random())
    ax = ti.sqrt(1 - ay**2)
    return ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n


@ti.func
def make_nested(f):
    f = f * 40
    i = int(f)
    if f < 0:
        if i % 2 == 1:
            f -= ti.floor(f)
        else:
            f = ti.floor(f) + 1 - f
    f = (f - 0.2) / 40
    return f


# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
@ti.func
def sdf(o):
    wall = ti.min(o[1] + 0.1, o[2] + 0.4)
    sphere = (o - ti.Vector([0.0, 0.35, 0.0])).norm() - 0.36

    q = ti.abs(o - ti.Vector([0.8, 0.3, 0])) - ti.Vector([0.3, 0.3, 0.3])
    box = ti.Vector([ti.max(0, q[0]), ti.max(0, q[1]), ti.max(0, q[2])]).norm() + ti.min(q.max(), 0)

    O = o - ti.Vector([-0.8, 0.3, 0])
    d = ti.Vector([ti.Vector([O[0], O[2]]).norm() - 0.3, abs(O[1]) - 0.3])
    cylinder = ti.min(d.max(), 0.0) + ti.Vector([ti.max(0, d[0]), ti.max(0, d[1])]).norm()

    geometry = make_nested(ti.min(sphere, box, cylinder))
    geometry = ti.max(geometry, -(0.32 - (o[1] * 0.6 + o[2] * 0.8)))
    return ti.min(wall, geometry)


@ti.func
def ray_march(p, d):
    j = 0
    dist = 0.0
    while j < 100 and sdf(p + dist * d) > 1e-6 and dist < inf:
        dist += sdf(p + dist * d)
        j += 1
    return ti.min(inf, dist)


@ti.func
def sdf_normal(p):
    d = 1e-3
    n = ti.Vector([0.0, 0.0, 0.0])
    sdf_center = sdf(p)
    for i in ti.static(range(3)):
        inc = p
        inc[i] += d
        n[i] = (1 / d) * (sdf(inc) - sdf_center)
    return n.normalized()


@ti.func
def next_hit(pos, d):
    closest, normal, c = inf, ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
    ray_march_dist = ray_march(pos, d)
    if ray_march_dist < dist_limit and ray_march_dist < closest:
        closest = ray_march_dist
        normal = sdf_normal(pos + d * closest)
        hit_pos = pos + d * closest
        t = int((hit_pos[0] + 10) * 1.1 + 0.5) % 3
        c = ti.Vector([0.4 + 0.3 * (t == 0), 0.4 + 0.2 * (t == 1), 0.4 + 0.3 * (t == 2)])
    return closest, normal, c


@ti.kernel
def render():
    for u, v in color_buffer:
        aspect_ratio = res[0] / res[1]
        pos = camera_pos
        d = ti.Vector(
            [
                (2 * fov * (u + ti.random()) / res[1] - fov * aspect_ratio - 1e-5),
                2 * fov * (v + ti.random()) / res[1] - fov - 1e-5,
                -1.0,
            ]
        )
        d = d.normalized()

        throughput = ti.Vector([1.0, 1.0, 1.0])

        depth = 0
        hit_light = 0.00

        while depth < max_ray_depth:
            closest, normal, c = next_hit(pos, d)
            depth += 1
            dist_to_light = intersect_light(pos, d)
            if dist_to_light < closest:
                hit_light = 1
                depth = max_ray_depth
            else:
                hit_pos = pos + closest * d
                if normal.norm_sqr() != 0:
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c
                else:
                    depth = max_ray_depth
        color_buffer[u, v] += throughput * hit_light


def main():
    gui = ti.GUI("SDF Path Tracer", res)
    last_t = 0
    for i in range(50000):
        render()
        interval = 10
        if i % interval == 0 and i > 0:
            print(f"{interval / (time.time() - last_t):.2f} samples/s")
            last_t = time.time()
            img = color_buffer.to_numpy() * (1 / (i + 1))
            img = img / img.mean() * 0.24
            gui.set_image(np.sqrt(img))
            gui.show()


if __name__ == "__main__":
    main()
