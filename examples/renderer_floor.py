import taichi as ti
import os
import numpy as np
import math
import time
import random
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion, inside_taichi
import sys

res = 1280, 720
num_spheres = 1024
color_buffer = ti.Vector(3, dt=ti.f32)
bbox = ti.Vector(3, dt=ti.f32)
grid_density = ti.var(dt=ti.i32)
voxel_has_particle = ti.var(dt=ti.i32)
max_ray_depth = 4
use_directional_light = True

fov = 0.23
dist_limit = 100

exposure = 1.5
camera_pos = ti.Vector([0.5, 0.32, 2.7])
light_direction = [1.2, 0.3, 0.7]
light_direction_noise = 0.03
light_color = [1.0, 1.0, 1.0]

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True
ti.cfg.arch = ti.cuda
grid_visualization_block_size = 16
grid_resolution = 256 // grid_visualization_block_size

frame_id = 0

render_voxel = False
inv_dx = 256.0
dx = 1.0 / inv_dx

camera_pos = ti.Vector([0.5, 0.27, 2.7])
supporter = 2
shutter_time = 0.5e-3
sphere_radius = 0.0015
particle_grid_res = 256
max_num_particles_per_cell = 8192
max_num_particles = 1024 * 1024 * 4

assert sphere_radius * 2 * particle_grid_res < 1

@ti.layout
def buffers():
  ti.root.dense(ti.ij, (res[0] // 8, res[1] // 8)).dense(ti.ij,
                                                         8).place(color_buffer)

  ti.root.dense(ti.ijk, 2).dense(ti.ijk, particle_grid_res // 8).dense(
      ti.ijk, 8).place(voxel_has_particle)
  ti.root.dense(ti.ijk, grid_resolution // 8).dense(ti.ijk,
                                                    8).place(grid_density)
  ti.root.dense(ti.i, 2).place(bbox)


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
  return ti.Matrix.normalized(n)


@ti.func
def sdf_color(p):
  scale = 0.4
  if inside_taichi(ti.Vector([p[0], p[2]])):
    scale = 1
  return ti.Vector([0.3, 0.5, 0.7]) * scale



@ti.func
def next_hit(pos, d, t):
  closest = inf
  normal = ti.Vector([0.0, 0.0, 0.0])
  c = ti.Vector([0.0, 0.0, 0.0])

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
  ti.parallelize(6)
  for u, v in color_buffer:
    pos = camera_pos
    d = ti.Vector([(
        2 * fov * (u + ti.random(ti.f32)) / res[1] - fov * aspect_ratio - 1e-5),
                   2 * fov * (v + ti.random(ti.f32)) / res[1] - fov - 1e-5,
                   -1.0])
    d = ti.Matrix.normalized(d)
    t = 0

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
          dir_noise = ti.Vector(
              [ti.random() - 0.5,
               ti.random() - 0.5,
               ti.random() - 0.5]) * light_direction_noise
          direct = ti.Matrix.normalized(
              ti.Vector(light_direction) + dir_noise)
          dot = direct.dot(normal)
          if dot > 0:
            dist, _, _ = next_hit(pos, direct, t)
            if dist > dist_limit:
              contrib += throughput * ti.Vector(light_color) * dot
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

    color_buffer[u, v] += contrib


@ti.kernel
def copy(img: ti.ext_arr()):
  for i, j in color_buffer:
    for c in ti.static(range(3)):
      img[i, j, c] = color_buffer[i, j][c]


def main():
  gui = ti.GUI('Particle Renderer', res)

  last_t = 0
  for i in range(500):
    render()

    interval = 10
    if i % interval == 0:
      img = np.zeros((res[0], res[1], 3), dtype=np.float32)
      copy(img)
      if last_t != 0:
        print("time per spp = {:.2f} ms".format(
            (time.time() - last_t) * 1000 / interval))
      last_t = time.time()
      img = img * (1 / (i + 1)) * exposure
      img = np.sqrt(img)
      gui.set_image(img)
      gui.show()


if __name__ == '__main__':
  main()
