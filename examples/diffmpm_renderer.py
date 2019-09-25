import taichi_lang as ti
import os
import numpy as np
import cv2
import math
import time
import random
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion, inside_taichi
import sys

res = 1280, 720
num_spheres = 1024
color_buffer = ti.Vector(3, dt=ti.f32)
sphere_pos = ti.Vector(3, dt=ti.f32)
bbox = ti.Vector(3, dt=ti.f32)
grid_density = ti.var(dt=ti.i32)
voxel_has_particle = ti.var(dt=ti.i32)
max_ray_depth = 4
use_directional_light = True

particle_x = ti.Vector(3, dt=ti.f32)
particle_v = ti.Vector(3, dt=ti.f32)
particle_color = ti.var(ti.i32)
pid = ti.var(ti.i32)
num_particles = ti.var(ti.i32)

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

# ti.runtime.print_preprocessed = True
# ti.cfg.print_ir = True
ti.cfg.arch = ti.cuda
grid_visualization_block_size = 16
grid_resolution = 256 // grid_visualization_block_size

scene = sys.argv[1]
folder = sys.argv[2]
frame_id = int(sys.argv[3])
assert scene in ['sand', 'fluid', 'snow']

render_voxel = False
inv_dx = 64.0
dx = 1.0 / inv_dx

if scene == 'fluid':
  supporter = 1
  shutter_time = 1e-3
  sphere_radius = 0.0018
  particle_grid_res = 128
  max_num_particles_per_cell = 256
  max_num_particles = 1024 * 1024 * 4
elif scene == 'snow':
  camera_pos = ti.Vector([0.5, 0.27, 2.7])
  supporter = 2
  shutter_time = 0.5e-3
  sphere_radius = 0.005
  particle_grid_res = 64
  max_num_particles_per_cell = 128
  max_num_particles = 1024 * 1024 * 4

assert sphere_radius * 2 * particle_grid_res < 1


@ti.layout
def buffers():
  ti.root.dense(ti.ij, (res[0] // 8, res[1] // 8)).dense(ti.ij, 8).place(
    color_buffer)
  ti.root.dense(ti.i, num_spheres).place(sphere_pos)
  
  ti.root.dense(ti.ijk, 2).dense(ti.ijk, particle_grid_res // 8).dense(ti.ijk,
                                                                       8).place(
    voxel_has_particle)
  ti.root.dense(ti.ijk, 4).dense(ti.ijk,
                                 particle_grid_res // 8).pointer().dense(ti.ijk,
                                                                         8).dynamic(
    ti.l,
    max_num_particles_per_cell).place(
    pid)
  
  ti.root.dense(ti.l, max_num_particles).place(particle_x, particle_v,
                                               particle_color)
  ti.root.place(num_particles)
  ti.root.dense(ti.ijk, grid_resolution // 8).dense(ti.ijk, 8).place(
    grid_density)
  ti.root.dense(ti.i, 2).place(bbox)


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
  
  p -= ti.Matrix.floor(p)
  boundary = 0.1
  count = 0
  for i in ti.static(range(3)):
    if p[i] < boundary or p[i] > 1 - boundary:
      count += 1
  f = 0.0
  if count >= 2:
    f = 1.0
  return ti.Vector([0.2, 0.3, 0.2]) * (2.3 - 2 * f)


n_pillars = 9


@ti.func
def sdf(o_):
  if ti.static(supporter == 0):
    o = o_ - ti.Vector([0.5, 0.002, 0.5])
    p = o
    h = 0.02
    ra = 0.29
    rb = 0.005
    d = (ti.Vector([p[0], p[2]]).norm() - 2.0 * ra + rb, ti.abs(p[1]) - h)
    dist = ti.min(ti.max(d[0], d[1]), 0.0) + ti.Vector(
      [ti.max(d[0], 0.0), ti.max(d[1], 0)]).norm() - rb
    return dist
  elif ti.static(supporter == 1):
    o = o_ - ti.Vector([0.5, 0.002, 0.5])
    dist = (o.abs() - ti.Vector([0.5, 0.02, 0.5])).max()
  else:
    dist = o_[1] - 0.027
  
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
def sdf_color(p_):
  p = p_
  scale = 0.4
  if inside_taichi(ti.Vector([p[0], p[2]])):
    scale = 1
  return ti.Vector([0.3, 0.5, 0.7]) * scale


@ti.func
def dda(eye_pos, d_):
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
  
  bbox_min = ti.Vector([0.0, 0.0, 0.0]) - 10 * eps
  bbox_max = ti.Vector([1.0, 1.0, 1.0]) + 10 * eps
  inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos, d)
  hit_distance = inf
  normal = ti.Vector([0.0, 0.0, 0.0])
  c = ti.Vector([0.0, 0.0, 0.0])
  if inter:
    near = ti.max(0, near)
    
    pos = eye_pos + d * (near + 5 * eps)
    
    o = grid_resolution * pos
    ipos = ti.Matrix.floor(o).cast(ti.i32)
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
        mini = (ipos - o + ti.Vector([0.5, 0.5, 0.5]) - rsign * 0.5) * rinv
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
  return bbox[0][0] <= ipos[0] * dx and ipos[0] < bbox[1][0] * inv_dx and bbox[0][1] * inv_dx <= ipos[1] and ipos[
    1] < bbox[1][1] * inv_dx and bbox[0][2] * inv_dx <= ipos[2] and ipos[2] < bbox[1][2] * inv_dx

  # pos = ipos * dx
  # return bbox[0][0] - 0.1 < pos[0] and pos[0] < bbox[1][0] + 0.1 and bbox[0][1] - 0.1 < pos[1] and \
  #       pos[1] < bbox[1][1] + 0.1 and bbox[0][2] - 0.1 < pos[2] and pos[2] < bbox[1][2] + 0.1


@ti.func
def dda_particle(eye_pos, d_, t):
  grid_res = particle_grid_res
  
  bbox_min = bbox[0]
  bbox_max = bbox[1]
  
  hit_pos = ti.Vector([0.0, 0.0, 0.0])
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
        num_particles = voxel_has_particle[ipos]
        if num_particles != 0:
          num_particles = ti.length(pid.parent(), ipos)
        for k in range(num_particles):
          p = pid[ipos[0], ipos[1], ipos[2], k]
          v = particle_v[p]
          x = particle_x[p] + t * v
          color = particle_color[p]
          dist, poss = intersect_sphere(eye_pos, d, x, sphere_radius)
          hit_pos = poss
          if dist < closest_intersection and dist > 0:
            hit_pos = eye_pos + dist * d
            closest_intersection = dist
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
  ti.parallelize(6)
  for u, v in color_buffer(0):
    pos = camera_pos
    d = ti.Vector(
      [(2 * fov * (u + ti.random(ti.f32)) / res[1] - fov * aspect_ratio - 1e-5),
       2 * fov * (v + ti.random(ti.f32)) / res[1] - fov - 1e-5,
       -1.0])
    d = ti.Matrix.normalized(d)
    if u < res[0] and v < res[1]:
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
            dir_noise = ti.Vector([ti.random() - 0.5, ti.random() - 0.5,
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
              if sphere_aabb_intersect_motion(box_min, box_max,
                                              x - 0.5 * shutter_time * v,
                                              x + 0.5 * shutter_time * v,
                                              sphere_radius):
                ti.append(pid.parent(), box_ipos, p)
                voxel_has_particle[box_ipos] = 1


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
  sand = np.fromfile(
    "mpm3d/iter{}/{:04d}.bin".format(folder, frame_id),
    dtype=np.float32)
  
  for i in range(num_spheres):
    for c in range(3):
      sphere_pos[i][c] = 0.5  # random.random()
  
  num_sand_particles = len(sand) // 7
  num_part = num_sand_particles
  sand = sand.reshape((7, num_sand_particles))
  sand = np.transpose(sand)
  np_x = sand[:, :3].flatten()
  np_v = sand[:, 3:6].flatten()
  np_c = sand[:, 6].flatten().astype(np.float32) * 0 + 127
  
  for i in range(3):
    print(sand[:, i].min(), sand[:, i].max())
    # bbox values must be multiples of dx
    bbox[0][i] = (math.floor(sand[:, i].min() * particle_grid_res) - 3.0) / particle_grid_res
    bbox[1][i] = (math.floor(sand[:, i].max() * particle_grid_res) + 3.0) / particle_grid_res

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
        # v_c = ti.min(1, particle_v[i].norm()) * 1.5
        # ti.print(v_c)
        if ti.static(scene == 'fluid'):
          v_c = ti.min(particle_v[i].norm() * 0.1, 1)
          particle_color[i] = rgb_to_i32(v_c * 0.3 + 0.3, v_c * 0.4 + 0.4,
                                         0.5 + v_c * 0.5)
        elif ti.static(scene == 'sand'):
          particle_color[i] = ti.cast(color[i], ti.i32)
        else:
          particle_color[i] = rgb_to_i32(0.85, 0.90, 1.0)
        
        # reconstruct grid using particle position and MPM p2g.
        for k in ti.static(range(27)):
          base_coord = (inv_dx * particle_x[i] - 0.5).cast(ti.i32) + ti.Vector(
            [k // 9, k // 3 % 3, k % 3])
          grid_density[base_coord / grid_visualization_block_size] = 1
  
  initialize_particle_x(np_x, np_v, np_c)
  initialize_particle_grid()
  
  output_folder = 'outputs_' + folder
  
  last_t = 0
  for i in range(500):
    render()
    
    interval = 50
    if i % interval == 0:
      img = np.zeros((res[1] * res[0] * 3,), dtype=np.float32)
      copy(img)
      if last_t != 0:
        print(
          "time per spp = {:.2f} ms".format(
            (time.time() - last_t) * 1000 / interval))
      last_t = time.time()
      img = img.reshape(res[1], res[0], 3) * (1 / (i + 1)) * exposure
      img = np.sqrt(img)
      cv2.imshow('img', img)
      cv2.waitKey(1)
  os.makedirs(output_folder, exist_ok=True)
  cv2.imwrite(output_folder + '/{:04d}.png'.format(frame_id), img * 255)


if __name__ == '__main__':
  main()
