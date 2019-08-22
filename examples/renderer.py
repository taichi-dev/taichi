import taichi_lang as ti
import numpy as np
import cv2
import math
import time
import random
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion
import sys

res = 1280, 720
num_spheres = 1024
color_buffer = ti.Vector(3, dt=ti.f32)
sphere_pos = ti.Vector(3, dt=ti.f32)
render_voxel = False
max_ray_depth = 4
use_directional_light = True

particle_x = ti.Vector(3, dt=ti.f32)
particle_v = ti.Vector(3, dt=ti.f32)
particle_color = ti.var(ti.i32)
pid = ti.var(ti.i32)
num_particles = ti.var(ti.i32)

camera_pos = ti.Vector([0.5, 0.3, 1.5])
vignette_strength = 0.9
vignette_radius = 0.0
vignette_center = [0.5, 0.5]

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True
ti.cfg.arch = ti.cuda
grid_resolution = 16

shutter_time = 3e-4
high_res = True
if high_res:
  sphere_radius = 0.0012
  particle_grid_res = 128
  max_num_particles_per_cell = 256
  max_num_particles = 1024 * 1024 * 8
else:
  sphere_radius = 0.03
  particle_grid_res = 8
  max_num_particles_per_cell = 64
  max_num_particles = 1024

assert sphere_radius * 2 * particle_grid_res < 1


@ti.layout
def buffers():
  ti.root.dense(ti.ij, (res[0] // 8, res[1] // 8)).dense(ti.ij, 8).place(
    color_buffer)
  ti.root.dense(ti.i, num_spheres).place(sphere_pos)
  ti.root.dense(ti.ijk, particle_grid_res // 8).dense(ti.ijk, 8).dynamic(ti.l,
                                                                         max_num_particles_per_cell).place(
    pid)
  ti.root.dense(ti.l, max_num_particles).place(particle_x, particle_v,
                                               particle_color)
  ti.root.place(num_particles)


@ti.func
def query_density_int(ipos):
  return ipos.min() % 3 == 0 and ipos.max() < grid_resolution


@ti.func
def voxel_color(pos):
  p = pos * grid_resolution

  p -= ti.Matrix.floor(p)
  boundary = 0.05
  count = 0
  for i in ti.static(range(3)):
    if p[i] < boundary or p[i] > 1 - boundary:
      count += 1
  f = 0.0
  if count >= 2:
    f = 1.0
  return ti.Vector([0.3, 0.4, 0.3]) * (1 + f)


@ti.func
def dda(pos, d_):
  d = d_
  for i in ti.static(range(3)):
    if ti.abs(d[i]) < 1e-6:
      d[i] = 1e-6
  rinv = 1.0 / d
  rsign = ti.Vector([0, 0, 0])
  for i in ti.static(range(3)):
    if d[i] > 0:
      rsign[i] = 1
    else:
      rsign[i] = -1

  o = grid_resolution * pos
  ipos = ti.Matrix.floor(o).cast(ti.i32)
  dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
  running = 1
  i = 0
  normal = ti.Vector([0.0, 0.0, 0.0])
  hit_pos = ti.Vector([0.0, 0.0, 0.0])
  c = ti.Vector([0.0, 0.0, 0.0])
  while running:
    last_sample = query_density_int(ipos)
    if last_sample:
      mini = (ipos - o + ti.Vector([0.5, 0.5, 0.5]) - rsign * 0.5) * rinv
      hit_distance = mini.max() * (1 / grid_resolution)
      hit_pos = pos + hit_distance * d
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
    if i > grid_resolution * 10:
      running = 0
      normal = [0, 0, 0]
    else:
      c = voxel_color(hit_pos)

  return normal, hit_pos, c


@ti.func
def intersect_spheres(pos, d):
  normal = ti.Vector([0.0, 0.0, 0.0])
  c = ti.Vector([0.0, 0.0, 0.0])
  min_dist = inf
  sid = -1

  for i in range(num_spheres):
    dist = intersect_sphere(pos, d, sphere_pos[i], 0.05)
    if dist < min_dist:
      min_dist = dist
      sid = i

  if min_dist < inf:
    hit_pos = pos + d * min_dist
    normal = ti.Matrix.normalized(hit_pos - sphere_pos[sid])
    c = [0.3, 0.5, 0.2]

  return min_dist, normal, c


@ti.func
def inside_particle_grid(ipos):
  grid_res = particle_grid_res
  return 0 <= ipos[0] and ipos[0] < grid_res and 0 <= ipos[1] and ipos[
    1] < grid_res and 0 <= ipos[2] and ipos[2] < grid_res


@ti.func
def dda_particle(eye_pos, d_, t):
  grid_res = particle_grid_res

  bbox_min = ti.Vector([0.0, 0.0, 0.0])
  bbox_max = ti.Vector([1.0, 1.0, 1.0])

  normal = ti.Vector([0.0, 0.0, 0.0])
  c = ti.Vector([0.0, 0.0, 0.0])
  d = d_
  for i in ti.static(range(3)):
    if ti.abs(d[i]) < 1e-6:
      d[i] = 1e-6

  inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos, d)
  near = ti.max(0, near)

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
    ipos = ti.Matrix.floor(o).cast(ti.i32)
    dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
    running = 1
    while running:
      inside = inside_particle_grid(ipos)

      if inside:
        num_particles = ti.length(pid.parent(), ipos)
        for k in range(num_particles):
          p = pid[ipos[0], ipos[1], ipos[2], k]
          v = particle_v[p]
          x = particle_x[p] + t * v
          color = 2 ** 24 - 1  # particle_color[p]
          dist = intersect_sphere(eye_pos, d, x, sphere_radius)
          if dist < closest_intersection:
            closest_intersection = dist
            hit_pos = eye_pos + d * closest_intersection
            normal = ti.Matrix.normalized(hit_pos - x)
            c = [color // 256 ** 2 / 255.0, color / 256 % 256 / 255.0,
                 color % 256 / 255.0]
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
def next_hit(pos_, d, t):
  pos = pos_
  closest, normal, c = dda_particle(pos_, d, t)
  # closest, normal, c = intersect_spheres(pos, d)
  if d[1] != 0:
    ray_closest = -(pos[1] + 0.2) / d[1]
    if ray_closest > 0 and ray_closest < closest:
      closest = ray_closest
      normal = ti.Vector([0.0, 1.0, 0.0])
      c = ti.Vector([0.3, 0.3, 0.4])
      # c = ti.Vector([1, 1, 1])

  if d[2] != 0:
    ray_closest = -(pos[2] + 0.5) / d[2]
    if ray_closest > 0 and ray_closest < closest:
      closest = ray_closest
      normal = ti.Vector([0.0, 0.0, 1.0])
      c = ti.Vector([0.3, 0.4, 0.4])
      # c = ti.Vector([1, 1, 1])

  return closest, normal, c

  if ti.static(render_voxel):
    return dda(pos, d)
  else:
    return intersect_spheres(pos, d)


aspect_ratio = res[0] / res[1]


@ti.kernel
def render():
  ti.parallelize(6)
  for u, v in color_buffer(0):
    fov = 0.5
    pos = camera_pos
    d = ti.Vector(
      [(2 * fov * (u + ti.random(ti.f32)) / res[1] - fov * aspect_ratio - 1e-3),
       2 * fov * (v + ti.random(ti.f32)) / res[1] - fov - 1e-3,
       -1.0])
    if u < res[0] and v < res[1]:
      # t = ti.min(1, ti.random(ti.f32) * 2) * shutter_time
      t = ti.random(ti.f32) * shutter_time

      contrib = ti.Vector([0.0, 0.0, 0.0])
      throughput = ti.Vector([1.0, 1.0, 1.0])

      d = ti.Matrix.normalized(d)

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
            direct = ti.Matrix.normalized(ti.Vector([1.0, 1.0, 1.0]))
            dot = direct.dot(normal)
            if dot > 0:
              dist, _, _ = next_hit(hit_pos + direct * 1e-4, direct, t)
              if dist > 1000:
                contrib += throughput * ti.Vector([1.0, 0.9, 0.7]) * dot
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
          # contrib *= ti.max(d[1], 0.05)
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
  for p in particle_x(0):
    if p < num_particles:
      x = particle_x[p]
      v = particle_v[p]
      ipos = ti.Matrix.floor(x * particle_grid_res).cast(ti.i32)
      for i in range(-support, support + 1):
        for j in range(-support, support + 1):
          for k in range(-support, support + 1):
            offset = ti.Vector([i, j, k])
            box_ipos = ipos + offset
            if inside_particle_grid(box_ipos):
              box_min = box_ipos * (1 / particle_grid_res)
              box_max = (box_ipos + ti.Vector([1, 1, 1])) * (
                    1 / particle_grid_res)
              if sphere_aabb_intersect_motion(box_min, box_max, x,
                                              x + shutter_time * v,
                                              sphere_radius):
                ti.append(pid.parent(), box_ipos, p)


@ti.func
def color_f32_to_i8(x):
  return ti.cast(ti.min(ti.max(x, 0.0), 1.0) * 255, ti.i32)


@ti.func
def rgb_to_i32(r, g, b):
  return color_f32_to_i8(r) * 65536 + color_f32_to_i8(
    g) * 256 + color_f32_to_i8(b)


@ti.kernel
def copy(img: np.ndarray):
  for i in range(res[0]):
    for j in range(res[1]):
      u = 1.0 * i / res[0]
      v = 1.0 * j / res[1]

      darken = 1.0 - vignette_strength * ti.max((ti.sqrt(
        ti.sqr(u - vignette_center[0]) + ti.sqr(
          v - vignette_center[1])) - vignette_radius), 0)

      coord = ((res[1] - 1 - j) * res[0] + i) * 3
      for c in ti.static(range(3)):
        img[coord + c] = color_buffer[i, j][2 - c] * darken


def main():
  fn = sys.argv[1]
  sand = np.fromfile("../final_particles/sand_new/{:04d}.bin".format(int(fn)),
                     dtype=np.float32)

  for i in range(num_spheres):
    for c in range(3):
      sphere_pos[i][c] = 0.5  # random.random()

  if high_res:
    num_sand_particles = len(sand) // 7
    num_part = num_sand_particles
    sand = sand.reshape((num_sand_particles, 7))
    np_x = sand[:, :3].flatten()
    np_v = sand[:, 3:6].flatten()
    np_c = sand[:, 6].flatten().astype(np.float32)
  else:
    num_part = 128
    np_x = np.random.rand(num_part * 3).astype(np.float32) * 0.8 + 0.1
    np_v = np.random.rand(num_part * 3).astype(np.float32) * 0
    np_c = np.random.randint(0, 256 ** 3, num_part,
                             dtype=np.int32).astype(np.float32)

  num_particles[None] = num_part
  print('num_input_particles =', num_part)

  @ti.kernel
  def initialize_particle_x(x: np.ndarray, v: np.ndarray, color: np.ndarray):
    for i in range(max_num_particles):
      if i < num_particles:
        for c in ti.static(range(3)):
          particle_x[i][c] = x[i * 3 + c]
        for c in ti.static(range(3)):
          particle_v[i][c] = v[i * 3 + c]
        particle_color[i] = ti.cast(color[i], ti.i32)

  initialize_particle_x(np_x, np_v, np_c)
  initialize_particle_grid()

  last_t = 0
  for i in range(1000):
    render()

    interval = 10
    if i % interval == 0:
      img = np.zeros((res[1] * res[0] * 3,), dtype=np.float32)
      copy(img)
      if last_t != 0:
        print(
          "time per spp = {:.2f} ms".format(
            (time.time() - last_t) * 1000 / interval))
      last_t = time.time()
      exposure = 2.5
      img = img.reshape(res[1], res[0], 3) * (1 / (i + 1)) * exposure
      img = np.sqrt(img)
      cv2.imshow('img', img)
      cv2.waitKey(1)
      cv2.imwrite('outputs/{:04d}.png'.format(int(fn)), img * 255)
  cv2.waitKey(0)


if __name__ == '__main__':
  main()
