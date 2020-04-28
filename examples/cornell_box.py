import taichi as ti
import time
import math
import numpy as np
from renderer_utils import ray_aabb_intersection, intersect_sphere, ray_plane_intersect, reflect, refract

ti.init(arch=ti.gpu)
res = (800, 800)
color_buffer = ti.Vector(3, dt=ti.f32, shape=res)
count_var = ti.var(ti.i32, shape=(1, ))

max_ray_depth = 10
eps = 1e-4
inf = 1e10
fov = 0.8

camera_pos = ti.Vector([0.0, 0.6, 3.0])

mat_none = 0
mat_lambertian = 1
mat_specular = 2
mat_glass = 3
mat_light = 4

light_y_pos = 2.0 - eps
light_x_min_pos = -0.25
light_x_range = 0.5
light_z_min_pos = 1.0
light_z_range = 0.12
light_area = light_x_range * light_z_range
light_min_pos = ti.Vector([light_x_min_pos, light_y_pos, light_z_min_pos])
light_max_pos = ti.Vector([
    light_x_min_pos + light_x_range, light_y_pos,
    light_z_min_pos + light_z_range
])
light_color = ti.Vector(list(np.array([0.9, 0.85, 0.7])))
light_normal = ti.Vector([0.0, -1.0, 0.0])

# No absorbtion, integrates over a unit hemisphere
lambertian_brdf = 1.0 / math.pi
# diamond!
refr_idx = 2.4

# right near sphere
sp1_center = ti.Vector([0.4, 0.225, 1.75])
sp1_radius = 0.22
# left far sphere
sp2_center = ti.Vector([-0.28, 0.55, 0.8])
sp2_radius = 0.32


@ti.func
def intersect_light(pos, d, tmax):
    hit, t, _ = ray_aabb_intersection(light_min_pos, light_max_pos, pos, d)
    if hit and 0 < t < tmax:
        hit = 1
    else:
        hit = 0
        t = inf
    return hit, t


@ti.func
def intersect_scene(pos, ray_dir):
    closest, normal = inf, ti.Vector.zero(ti.f32, 3)
    c, mat = ti.Vector.zero(ti.f32, 3), mat_none

    # right near sphere
    cur_dist, hit_pos = intersect_sphere(pos, ray_dir, sp1_center, sp1_radius)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = ti.normalized(hit_pos - sp1_center)
        c, mat = ti.Vector([1.0, 1.0, 1.0]), mat_glass
    # left far sphere
    cur_dist, hit_pos = intersect_sphere(pos, ray_dir, sp2_center, sp2_radius)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = ti.normalized(hit_pos - sp2_center)
        c, mat = ti.Vector([0.8, 0.5, 0.4]), mat_specular
    # left
    pnorm = ti.Vector([1.0, 0.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([-1.1, 0.0,
                                                               0.0]), pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([0.65, 0.05, 0.05]), mat_lambertian
    # right
    pnorm = ti.Vector([-1.0, 0.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([1.1, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([0.12, 0.45, 0.15]), mat_lambertian
    # bottom
    gray = ti.Vector([0.93, 0.93, 0.93])
    pnorm = ti.Vector([0.0, 1.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = gray, mat_lambertian
    # top
    pnorm = ti.Vector([0.0, -1.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, 2.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = gray, mat_lambertian
    # far
    pnorm = ti.Vector([0.0, 0.0, 1.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = gray, mat_lambertian
    # light
    hit_l, cur_dist = intersect_light(pos, ray_dir, closest)
    if hit_l and 0 < cur_dist < closest:
        # technically speaking, no need to check the second term
        closest = cur_dist
        normal = light_normal
        c, mat = light_color, mat_light

    return closest, normal, c, mat


@ti.func
def visible_to_light(pos, ray_dir):
    a, b, c, mat = intersect_scene(pos, ray_dir)
    return mat == mat_light


@ti.func
def dot_or_zero(n, l):
    return max(0.0, n.dot(l))


@ti.func
def mis_power_heuristic(pf, pg):
    # Assume 1 sample for each distribution
    f = pf**2
    g = pg**2
    return f / (f + g)


@ti.func
def compute_area_light_pdf(pos, ray_dir):
    hit_l, t = intersect_light(pos, ray_dir, inf)
    pdf = 0.0
    if hit_l:
        l_cos = light_normal.dot(-ray_dir)
        if l_cos > eps:
            tmp = ray_dir * t
            dist_sqr = tmp.dot(tmp)
            pdf = dist_sqr / (light_area * l_cos)
    return pdf


@ti.func
def compute_brdf_pdf(normal, sample_dir):
    return dot_or_zero(normal, sample_dir) / math.pi


@ti.func
def sample_area_light(hit_pos, pos_normal):
    # sampling inside the light area
    x = ti.random() * light_x_range + light_x_min_pos
    z = ti.random() * light_z_range + light_z_min_pos
    on_light_pos = ti.Vector([x, light_y_pos, z])
    return ti.normalized(on_light_pos - hit_pos)


@ti.func
def sample_brdf(normal):
    # cosine hemisphere sampling
    # first, uniformly sample on a disk (r, theta)
    r, theta = 0.0, 0.0
    sx = ti.random() * 2.0 - 1.0
    sy = ti.random() * 2.0 - 1.0
    if sx >= -sy:
        if sx > sy:
            # first region
            r = sx
            div = abs(sy / r)
            if sy > 0.0:
                theta = div
            else:
                theta = 7.0 + div
        else:
            # second region
            r = sy
            div = abs(sx / r)
            if sx > 0.0:
                theta = 1.0 + sx / r
            else:
                theta = 2.0 + sx / r
    else:
        if sx <= sy:
            # third region
            r = -sx
            div = abs(sy / r)
            if sy > 0.0:
                theta = 3.0 + div
            else:
                theta = 4.0 + div
        else:
            # fourth region
            r = -sy
            div = abs(sx / r)
            if sx < 0.0:
                theta = 5.0 + div
            else:
                theta = 6.0 + div
    # Malley's method
    u = ti.Vector([1.0, 0.0, 0.0])
    if abs(normal[1]) < 1 - eps:
        u = ti.cross(normal, ti.Vector([0.0, 1.0, 0.0]))
    v = ti.cross(normal, u)

    theta = theta * math.pi * 0.25
    costt, sintt = ti.cos(theta), ti.sin(theta)
    xy = (u * costt + v * sintt) * r
    zlen = ti.sqrt(max(0.0, 1.0 - xy.dot(xy)))
    return xy + zlen * normal


@ti.func
def sample_direct_light(hit_pos, hit_normal, hit_color):
    direct_li = ti.Vector([0.0, 0.0, 0.0])
    fl = lambertian_brdf * hit_color * light_color
    light_pdf, brdf_pdf = 0.0, 0.0
    # sample area light
    to_light_dir = sample_area_light(hit_pos, hit_normal)
    if to_light_dir.dot(hit_normal) > 0:
        light_pdf = compute_area_light_pdf(hit_pos, to_light_dir)
        brdf_pdf = compute_brdf_pdf(hit_normal, to_light_dir)
        if light_pdf > 0 and brdf_pdf > 0:
            l_visible = visible_to_light(hit_pos, to_light_dir)
            if l_visible:
                w = mis_power_heuristic(light_pdf, brdf_pdf)
                nl = dot_or_zero(to_light_dir, hit_normal)
                direct_li += fl * w * nl / light_pdf
    # sample brdf
    brdf_dir = sample_brdf(hit_normal)
    brdf_pdf = compute_brdf_pdf(hit_normal, brdf_dir)
    if brdf_pdf > 0:
        light_pdf = compute_area_light_pdf(hit_pos, brdf_dir)
        if light_pdf > 0:
            l_visible = visible_to_light(hit_pos, brdf_dir)
            if l_visible:
                w = mis_power_heuristic(brdf_pdf, light_pdf)
                nl = dot_or_zero(brdf_dir, hit_normal)
                direct_li += fl * w * nl / brdf_pdf
    return direct_li


@ti.func
def schlick(cos, eta):
    r0 = (1.0 - eta) / (1.0 + eta)
    r0 = r0 * r0
    return r0 + (1 - r0) * ((1.0 - cos)**5)


@ti.func
def sample_ray_dir(indir, normal, hit_pos, mat):
    u = ti.Vector([0.0, 0.0, 0.0])
    pdf = 1.0
    if mat == mat_lambertian:
        u = sample_brdf(normal)
        pdf = max(eps, compute_brdf_pdf(normal, u))
    elif mat == mat_specular:
        u = reflect(indir, normal)
    elif mat == mat_glass:
        cos = indir.dot(normal)
        ni_over_nt = refr_idx
        outn = normal
        if cos > 0.0:
            outn = -normal
            cos = refr_idx * cos
        else:
            ni_over_nt = 1.0 / refr_idx
            cos = -cos
        has_refr, refr_dir = refract(indir, outn, ni_over_nt)
        refl_prob = 1.0
        if has_refr:
            refl_prob = schlick(cos, refr_idx)
        if ti.random() < refl_prob:
            u = reflect(indir, normal)
        else:
            u = refr_dir
    return ti.normalized(u), pdf


stratify_res = 5
inv_stratify = 1.0 / 5.0


@ti.kernel
def render():
    for u, v in color_buffer:
        aspect_ratio = res[0] / res[1]
        pos = camera_pos
        cur_iter = count_var[0]
        str_x, str_y = (cur_iter / stratify_res), (cur_iter % stratify_res)
        ray_dir = ti.Vector([
            (2 * fov * (u + (str_x + ti.random()) * inv_stratify) / res[1] -
             fov * aspect_ratio - 1e-5),
            (2 * fov * (v + (str_y + ti.random()) * inv_stratify) / res[1] -
             fov - 1e-5),
            -1.0,
        ])
        ray_dir = ti.normalized(ray_dir)

        acc_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        depth = 0
        while depth < max_ray_depth:
            closest, hit_normal, hit_color, mat = intersect_scene(pos, ray_dir)
            if mat == mat_none:
                break

            hit_pos = pos + closest * ray_dir
            hit_light = (mat == mat_light)
            if hit_light:
                acc_color += throughput * light_color
                break
            elif mat == mat_lambertian:
                acc_color += throughput * sample_direct_light(
                    hit_pos, hit_normal, hit_color)

            depth += 1
            ray_dir, pdf = sample_ray_dir(ray_dir, hit_normal, hit_pos, mat)
            pos = hit_pos + 1e-4 * ray_dir
            if mat == mat_lambertian:
                throughput *= lambertian_brdf * hit_color * dot_or_zero(
                    hit_normal, ray_dir) / pdf
            else:
                throughput *= hit_color
        color_buffer[u, v] += acc_color
    count_var[0] = (count_var[0] + 1) % (stratify_res * stratify_res)


gui = ti.GUI('Cornell Box', res)
last_t = time.time()
for i in range(50000):
    render()
    interval = 10
    if i % interval == 0 and i > 0:
        img = color_buffer.to_numpy(as_vector=True) * (1 / (i + 1))
        img = np.sqrt(img / img.mean() * 0.24)
        print("{:.2f} samples/s ({} iters, var={})".format(
            interval / (time.time() - last_t), i, np.var(img)))
        last_t = time.time()
        gui.set_image(img)
        gui.show()

input("Press any key to quit")
