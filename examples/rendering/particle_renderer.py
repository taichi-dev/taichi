import math
import time

import numpy as np
from renderer_utils import (eps, inf, inside_taichi, intersect_sphere, out_dir,
                            ray_aabb_intersection,
                            sphere_aabb_intersect_motion)

import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=4)

res = 1280, 720
num_spheres = 1024
color_buffer = ti.Vector.field(3, dtype=ti.f32)
bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
grid_density = ti.field(dtype=ti.i32)
voxel_has_particle = ti.field(dtype=ti.i32)
max_ray_depth = 4
use_directional_light = True

particle_x = ti.Vector.field(3, dtype=ti.f32)
particle_v = ti.Vector.field(3, dtype=ti.f32)
particle_color = ti.Vector.field(3, dtype=ti.f32)
pid = ti.field(ti.i32)
num_particles = ti.field(ti.i32, shape=())

fov = 0.23
dist_limit = 100

exposure = 1.5
camera_pos = ti.Vector([0.5, 0.32, 2.7])
vignette_strength = 0.9
vignette_radius = 0.0
vignette_center = [0.5, 0.5]
light_direction = [1.2, 0.3, 0.7]
light_direction_noise = 0.03
light_color = [1.0, 1.0, 1.0]

grid_visualization_block_size = 16
grid_resolution = 256 // grid_visualization_block_size

frame_id = 0

render_voxel = False  # see dda()
inv_dx = 256.0
dx = 1.0 / inv_dx

camera_pos = ti.Vector([0.5, 0.27, 2.7])
supporter = 2
shutter_time = 0.5e-3  # half the frame time (1e-3)
sphere_radius = 0.0015
particle_grid_res = 256
max_num_particles_per_cell = 8192 * 1024
max_num_particles = 1024 * 1024 * 4

assert sphere_radius * 2 * particle_grid_res < 1

ti.root.dense(ti.ij, (res[0] // 8, res[1] // 8)).dense(ti.ij,
                                                       8).place(color_buffer)

ti.root.dense(ti.ijk, 2).dense(ti.ijk, particle_grid_res // 8).dense(
    ti.ijk, 8).place(voxel_has_particle)
ti.root.dense(ti.ijk, 4).pointer(ti.ijk, particle_grid_res // 8).dense(
    ti.ijk, 8).dynamic(ti.l, max_num_particles_per_cell, 512).place(pid)

ti.root.dense(ti.l, max_num_particles).place(particle_x, particle_v,
                                             particle_color)
ti.root.dense(ti.ijk, grid_resolution // 8).dense(ti.ijk,
                                                  8).place(grid_density)


@ti.func
def inside_grid(ipos):
    return ipos.min() >= 0 and ipos.max() < grid_resolution


# The dda algorithm requires the voxel grid to have one surrounding layer of void region
# to correctly render the outmost voxel faces
@ti.func
def inside_grid_loose(ipos):
    return ipos.min() >= -1 and ipos.max() <= grid_resolution


@ti.func
def query_density_int(ipos):
    inside = inside_grid(ipos)
    ret = 0
    if inside:
        ret = grid_density[ipos]
    else:
        ret = 0
    return ret


@ti.func
def voxel_color(pos):
    p = pos * grid_resolution

    p -= ti.floor(p)

    boundary = 0.1
    count = 0
    for i in ti.static(range(3)):
        if p[i] < boundary or p[i] > 1 - boundary:
            count += 1
    f = 0.0
    if count >= 2:
        f = 1.0
    return ti.Vector([0.2, 0.3, 0.2]) * (2.3 - 2 * f)


@ti.func
def sdf(o):
    dist = 0.0
    if ti.static(supporter == 0):
        o -= ti.Vector([0.5, 0.002, 0.5])
        p = o
        h = 0.02
        ra = 0.29
        rb = 0.005
        d = (ti.Vector([p[0], p[2]]).norm() - 2.0 * ra + rb, abs(p[1]) - h)
        dist = min(max(d[0], d[1]), 0.0) + ti.Vector(
            [max(d[0], 0.0), max(d[1], 0)]).norm() - rb
    elif ti.static(supporter == 1):
        o -= ti.Vector([0.5, 0.002, 0.5])
        dist = (o.abs() - ti.Vector([0.5, 0.02, 0.5])).max()
    else:
        dist = o[1] - 0.027

    return dist


@ti.func
def ray_march(p, d):
    j = 0
    dist = 0.0
    limit = 200
    while j < limit and sdf(p + dist * d) > 1e-8 and dist < dist_limit:
        dist += sdf(p + dist * d)
        j += 1
    if dist > dist_limit:
        dist = inf
    return dist


@ti.func
def sdf_normal(p):
    d = 1e-3
    n = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        inc = p
        dec = p
        inc[i] += d
        dec[i] -= d
        n[i] = (0.5 / d) * (sdf(inc) - sdf(dec))
    return n.normalized()


@ti.func
def sdf_color(p):
    scale = 0.4
    if inside_taichi(ti.Vector([p[0], p[2]])):
        scale = 1
    return ti.Vector([0.3, 0.5, 0.7]) * scale


# Digital differential analyzer for the grid visualization (render_voxels=True)
@ti.func
def dda(eye_pos, d):
    for i in ti.static(range(3)):
        if abs(d[i]) < 1e-6:
            d[i] = 1e-6
    rinv = 1.0 / d
    rsign = ti.Vector([0, 0, 0])
    for i in ti.static(range(3)):
        if d[i] > 0:
            rsign[i] = 1
        else:
            rsign[i] = -1

    bbox_min = ti.Vector([0.0, 0.0, 0.0]) - 10 * eps
    bbox_max = ti.Vector([1.0, 1.0, 1.0]) + 10 * eps
    inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos, d)
    hit_distance = inf
    normal = ti.Vector([0.0, 0.0, 0.0])
    c = ti.Vector([0.0, 0.0, 0.0])
    if inter:
        near = max(0, near)

        pos = eye_pos + d * (near + 5 * eps)

        o = grid_resolution * pos
        ipos = ti.floor(o).cast(int)
        dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
        running = 1
        i = 0
        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        while running:
            last_sample = query_density_int(ipos)
            if not inside_grid_loose(ipos):
                running = 0
                # normal = [0, 0, 0]

            if last_sample:
                mini = (ipos - o + ti.Vector([0.5, 0.5, 0.5]) -
                        rsign * 0.5) * rinv
                hit_distance = mini.max() * (1 / grid_resolution) + near
                hit_pos = eye_pos + hit_distance * d
                c = voxel_color(hit_pos)
                running = 0
            else:
                mm = ti.Vector([0, 0, 0])
                if dis[0] <= dis[1] and dis[0] < dis[2]:
                    mm[0] = 1
                elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                    mm[1] = 1
                else:
                    mm[2] = 1
                dis += mm * rsign * rinv
                ipos += mm * rsign
                normal = -mm * rsign
            i += 1
    return hit_distance, normal, c


@ti.func
def inside_particle_grid(ipos):
    pos = ipos * dx
    return bbox[0][0] <= pos[0] and pos[0] < bbox[1][0] and bbox[0][1] <= pos[
        1] and pos[1] < bbox[1][1] and bbox[0][2] <= pos[2] and pos[2] < bbox[
            1][2]


# DDA for the particle visualization (render_voxels=False)
@ti.func
def dda_particle(eye_pos, d, t):
    grid_res = particle_grid_res

    # bounding box
    bbox_min = bbox[0]
    bbox_max = bbox[1]

    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])
    c = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if abs(d[i]) < 1e-6:
            d[i] = 1e-6

    inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos, d)
    near = max(0, near)

    closest_intersection = inf

    if inter:
        pos = eye_pos + d * (near + eps)

        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        o = grid_res * pos
        ipos = ti.floor(o).cast(int)
        dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
        running = 1
        # DDA for voxels with at least one particle
        while running:
            inside = inside_particle_grid(ipos)

            if inside:
                # once we actually intersect with a voxel that contains at least one particle, loop over the particle list
                num_particles = voxel_has_particle[ipos]
                if num_particles != 0:
                    num_particles = ti.length(pid.parent(), ipos)
                for k in range(num_particles):
                    p = pid[ipos[0], ipos[1], ipos[2], k]
                    v = particle_v[p]
                    x = particle_x[p] + t * v
                    color = particle_color[p]
                    # ray-sphere intersection
                    dist, poss = intersect_sphere(eye_pos, d, x, sphere_radius)
                    hit_pos = poss
                    if dist < closest_intersection and dist > 0:
                        hit_pos = eye_pos + dist * d
                        closest_intersection = dist
                        normal = (hit_pos - x).normalized()
                        c = color
            else:
                running = 0
                normal = [0, 0, 0]

            if closest_intersection < inf:
                running = 0
            else:
                # hits nothing. Continue ray marching
                mm = ti.Vector([0, 0, 0])
                if dis[0] <= dis[1] and dis[0] <= dis[2]:
                    mm[0] = 1
                elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                    mm[1] = 1
                else:
                    mm[2] = 1
                dis += mm * rsign * rinv
                ipos += mm * rsign

    return closest_intersection, normal, c


@ti.func
def next_hit(pos, d, t):
    closest = inf
    normal = ti.Vector([0.0, 0.0, 0.0])
    c = ti.Vector([0.0, 0.0, 0.0])
    if ti.static(render_voxel):
        closest, normal, c = dda(pos, d)
    else:
        closest, normal, c = dda_particle(pos, d, t)

    if d[2] != 0:
        ray_closest = -(pos[2] + 5.5) / d[2]
        if ray_closest > 0 and ray_closest < closest:
            closest = ray_closest
            normal = ti.Vector([0.0, 0.0, 1.0])
            c = ti.Vector([0.6, 0.7, 0.7])

    ray_march_dist = ray_march(pos, d)
    if ray_march_dist < dist_limit and ray_march_dist < closest:
        closest = ray_march_dist
        normal = sdf_normal(pos + d * closest)
        c = sdf_color(pos + d * closest)

    return closest, normal, c


aspect_ratio = res[0] / res[1]


@ti.kernel
def render():
    for u, v in color_buffer:
        pos = camera_pos
        d = ti.Vector([(2 * fov * (u + ti.random(ti.f32)) / res[1] -
                        fov * aspect_ratio - 1e-5),
                       2 * fov * (v + ti.random(ti.f32)) / res[1] - fov - 1e-5,
                       -1.0])
        d = d.normalized()
        t = (ti.random() - 0.5) * shutter_time

        contrib = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        depth = 0
        hit_sky = 1
        ray_depth = 0

        while depth < max_ray_depth:
            closest, normal, c = next_hit(pos, d, t)
            hit_pos = pos + closest * d
            depth += 1
            ray_depth = depth
            if normal.norm() != 0:
                d = out_dir(normal)
                pos = hit_pos + 1e-4 * d
                throughput *= c

                if ti.static(use_directional_light):
                    dir_noise = ti.Vector([
                        ti.random() - 0.5,
                        ti.random() - 0.5,
                        ti.random() - 0.5
                    ]) * light_direction_noise
                    direct = (ti.Vector(light_direction) +
                              dir_noise).normalized()
                    dot = direct.dot(normal)
                    if dot > 0:
                        dist, _, _ = next_hit(pos, direct, t)
                        if dist > dist_limit:
                            contrib += throughput * ti.Vector(
                                light_color) * dot
            else:  # hit sky
                hit_sky = 1
                depth = max_ray_depth

            max_c = throughput.max()
            if ti.random() > max_c:
                depth = max_ray_depth
                throughput = [0, 0, 0]
            else:
                throughput /= max_c

        if hit_sky:
            if ray_depth != 1:
                # contrib *= max(d[1], 0.05)
                pass
            else:
                # directly hit sky
                pass
        else:
            throughput *= 0

        # contrib += throughput
        color_buffer[u, v] += contrib


support = 2


@ti.kernel
def initialize_particle_grid():
    for p in range(num_particles[None]):
        x = particle_x[p]
        v = particle_v[p]
        ipos = ti.floor(x * particle_grid_res).cast(ti.i32)
        for i in range(-support, support + 1):
            for j in range(-support, support + 1):
                for k in range(-support, support + 1):
                    offset = ti.Vector([i, j, k])
                    box_ipos = ipos + offset
                    if inside_particle_grid(box_ipos):
                        box_min = box_ipos * (1 / particle_grid_res)
                        box_max = (box_ipos + ti.Vector([1, 1, 1])) * (
                            1 / particle_grid_res)
                        if sphere_aabb_intersect_motion(
                                box_min, box_max, x - 0.5 * shutter_time * v,
                                x + 0.5 * shutter_time * v, sphere_radius):
                            ti.append(pid.parent(), box_ipos, p)
                            voxel_has_particle[box_ipos] = 1


@ti.kernel
def copy(img: ti.ext_arr(), samples: ti.i32):
    for i, j in color_buffer:
        u = 1.0 * i / res[0]
        v = 1.0 * j / res[1]

        darken = 1.0 - vignette_strength * max(
            (ti.sqrt((u - vignette_center[0])**2 +
                     (v - vignette_center[1])**2) - vignette_radius), 0)

        for c in ti.static(range(3)):
            img[i, j, c] = ti.sqrt(color_buffer[i, j][c] * darken * exposure /
                                   samples)


def main():
    num_part = 100000
    np_x = np.random.rand(num_part, 3).astype(np.float32) * 0.4 + 0.2
    np_v = np.random.rand(num_part, 3).astype(np.float32) * 0
    np_c = np.zeros((num_part, 3)).astype(np.float32)
    np_c[:, 0] = 0.85
    np_c[:, 1] = 0.9
    np_c[:, 2] = 1

    for i in range(3):
        # bbox values must be multiples of dx
        # bbox values are the min and max particle coordinates, with 3 dx margin
        bbox[0][i] = (math.floor(np_x[:, i].min() * particle_grid_res) -
                      3.0) / particle_grid_res
        bbox[1][i] = (math.floor(np_x[:, i].max() * particle_grid_res) +
                      3.0) / particle_grid_res

    num_particles[None] = num_part
    print('num_input_particles =', num_part)

    @ti.kernel
    def initialize_particle_x(x: ti.ext_arr(), v: ti.ext_arr(),
                              color: ti.ext_arr()):
        for i in range(num_particles[None]):
            for c in ti.static(range(3)):
                particle_x[i][c] = x[i, c]
                particle_v[i][c] = v[i, c]
                particle_color[i][c] = color[i, c]

            for k in ti.static(range(27)):
                base_coord = (inv_dx * particle_x[i] - 0.5).cast(
                    ti.i32) + ti.Vector([k // 9, k // 3 % 3, k % 3])
                grid_density[base_coord // grid_visualization_block_size] = 1

    initialize_particle_x(np_x, np_v, np_c)
    initialize_particle_grid()

    gui = ti.GUI('Particle Renderer', res)

    last_t = 0
    for i in range(500):
        render()

        interval = 10
        if i % interval == 0:
            img = np.zeros((res[0], res[1], 3), dtype=np.float32)
            copy(img, i + 1)
            if last_t != 0:
                print("time per spp = {:.2f} ms".format(
                    (time.time() - last_t) * 1000 / interval))
            last_t = time.time()
            gui.set_image(img)
            gui.show()


if __name__ == '__main__':
    main()
