import taichi as ti
import time
import math
import numpy as np
from renderer_utils import ray_aabb_intersection, intersect_sphere, ray_plane_intersect, reflect, refract

ti.init(arch=ti.metal)
res = (800, 800)
color_buffer = ti.Vector(3, dt=ti.f32, shape=res)
max_ray_depth = 10
eps = 1e-4
inf = 1e10
fov = 1.0

camera_pos = ti.Vector([0.0, 0.6, 2.0])
lihgt_min_pos = ti.Vector([-0.2, 1.99, 0.3])
light_max_pos = ti.Vector([0.2, 1.99, 0.4])
light_color = ti.Vector([0.9, 0.85, 0.7])
mat_lambertian = 0
mat_metal = 1
mat_glass = 2
refr_idx = 2.4  # diamond!

# right near sphere
sp1_center = ti.Vector([0.35, 0.22, 1.14])
sp1_radius = 0.21
# left far sphere
sp2_center = ti.Vector([-0.28, 0.6, 0.6])
sp2_radius = 0.42


@ti.func
def intersect_light(pos, d):
    intersect, tmin, _ = ray_aabb_intersection(lihgt_min_pos, light_max_pos,
                                               pos, d)
    if tmin < 0 or intersect == 0:
        tmin = inf
    return tmin


@ti.func
def schlick(cos, eta):
    r0 = (1.0 - eta) / (1.0 + eta)
    r0 = r0 * r0
    return r0 + (1 - r0) * ((1.0 - cos)**5)


@ti.func
def out_dir(indir, n, mat):
    u = ti.Vector([1.0, 0.0, 0.0])
    if mat == mat_lambertian:
        if abs(n[1]) < 1 - eps:
            u = ti.normalized(ti.cross(n, ti.Vector([0.0, 1.0, 0.0])))
        v = ti.cross(n, u)
        phi = 2 * math.pi * ti.random()
        ay = ti.sqrt(ti.random())
        ax = ti.sqrt(1 - ay**2)
        u = ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n
    elif mat == mat_metal:
        u = reflect(indir, n)
    else:
        # glass
        cos = ti.dot(indir, n)
        ni_over_nt = refr_idx
        outn = n
        if cos > 0.0:
            outn = -n
            cos = refr_idx * cos
        else:
            ni_over_nt = 1.0 / refr_idx
            cos = -cos
        has_refr, refr_dir = refract(indir, outn, ni_over_nt)
        refl_prob = 1.0
        if has_refr:
            refl_prob = schlick(cos, refr_idx)
        if ti.random() < refl_prob:
            u = reflect(indir, n)
        else:
            u = refr_dir
    return ti.normalized(u)


@ti.func
def next_hit(pos, d):
    closest, normal = inf, ti.Vector.zero(ti.f32, 3)
    c, mat = ti.Vector.zero(ti.f32, 3), mat_lambertian

    # right near sphere
    cur_dist, hit_pos = intersect_sphere(pos, d, sp1_center, sp1_radius)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = ti.normalized(hit_pos - sp1_center)
        c, mat = ti.Vector([1.0, 1.0, 1.0]), mat_glass
    # left far sphere
    cur_dist, hit_pos = intersect_sphere(pos, d, sp2_center, sp2_radius)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = ti.normalized(hit_pos - sp2_center)
        c, mat = ti.Vector([0.8, 0.5, 0.4]), mat_metal
    # left
    pnorm = ti.Vector([1.0, 0.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, d, ti.Vector([-1.0, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([1.0, 0.0, 0.0]), mat_lambertian
    # right
    pnorm = ti.Vector([-1.0, 0.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, d, ti.Vector([1.0, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([0.0, 1.0, 0.0]), mat_lambertian
    # bottom
    pnorm = ti.Vector([0.0, 1.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, d, ti.Vector([0.0, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([1.0, 1.0, 1.0]), mat_lambertian
    # top
    pnorm = ti.Vector([0.0, -1.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, d, ti.Vector([0.0, 2.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([1.0, 1.0, 1.0]), mat_lambertian
    # far
    pnorm = ti.Vector([0.0, 0.0, 1.0])
    cur_dist, _ = ray_plane_intersect(pos, d, ti.Vector([0.0, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([1.0, 1.0, 1.0]), mat_lambertian

    return closest, normal, c, mat


@ti.kernel
def render():
    for u, v in color_buffer:
        aspect_ratio = res[0] / res[1]
        pos = camera_pos
        d = ti.Vector([
            (2 * fov * (u + ti.random()) / res[1] - fov * aspect_ratio - 1e-5),
            (2 * fov * (v + ti.random()) / res[1] - fov - 1e-5),
            -1.0,
        ])
        d = ti.normalized(d)

        throughput = ti.Vector([1.0, 1.0, 1.0])

        depth = 0
        hit_light = 0.0

        while depth < max_ray_depth:
            closest, normal, c, mat = next_hit(pos, d)
            depth += 1
            dist_to_light = intersect_light(pos, d)
            if dist_to_light < closest:
                hit_light = 1.0
                depth = max_ray_depth
                throughput *= light_color
            else:
                if normal.norm_sqr() != 0:
                    hit_pos = pos + closest * d
                    d = out_dir(d, normal, mat)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c
                else:
                    depth = max_ray_depth
        color_buffer[u, v] += throughput * hit_light


gui = ti.GUI('Cornell Box', res)
last_t = 0
for i in range(50000):
    render()
    interval = 10
    if i % interval == 0 and i > 0:
        print("{:.2f} samples/s".format(interval / (time.time() - last_t)))
        last_t = time.time()
        img = color_buffer.to_numpy(as_vector=True) * (1 / (i + 1))
        img = img / img.mean() * 0.24
        gui.set_image(np.sqrt(img))
        gui.show()
