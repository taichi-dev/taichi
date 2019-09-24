import taichi_lang as ti
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pdb
from imageio import imread, imwrite

real = ti.f32
ti.set_default_fp(real)

ti.cfg.arch = ti.cuda

num_iterations = 100
res = 512
density_res = 256
inv_density_res = 1.0 / density_res
res_f32 = float(res)
dx = 0.04
n_views = 8
torus_r1 = 0.4
torus_r2 = 0.1
fov = 1
camera_origin_radius = 1
marching_steps = 1000
learning_rate = 0.1

scalar = lambda: ti.var(dt=real)

density = scalar()
target_images = scalar()
images = scalar()
loss = scalar()

@ti.layout
def place():
  # TODO: use sparsity
  ti.root.dense(ti.ijk, res).place(density)
  ti.root.dense(ti.l, n_views).dense(ti.ij, res).place(target_images, images)
  ti.root.place(loss)
  ti.root.lazy_grad()

@ti.func
def in_box(x, y, z):
  # The density grid is contained in a unit box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
  return x >= -0.5 and x < 0.5 and y >= -0.5 and y < 0.5 and z >= -0.5 and z < 0.5

def ray_march(field):
  @ti.kernel
  def kernel(angle: ti.f32, view_id: ti.i32):
    camera_origin = ti.Vector([camera_origin_radius * ti.sin(angle), 0, camera_origin_radius * ti.cos(angle)])

    for y in range(res):
      for x in range(res):
        for k in range(marching_steps):
          dir = ti.Vector([
            fov * (ti.cast(x, ti.f32) / (res_f32 / 2.0) - res_f32 / res_f32),
            fov * (ti.cast(y, ti.f32) / (res_f32 / 2.0) - 1.0),
            -1.0
          ])

          length = ti.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
          dir /= length

          rotated_x = dir[0] * ti.cos(angle) + dir[2] * ti.sin(angle)
          rotated_z = -dir[0] * ti.sin(angle) + dir[2] * ti.cos(angle)
          dir[0] = rotated_x
          dir[2] = rotated_z

          point = camera_origin + (k + 1) * dx * dir

          if in_box(point[0], point[1], point[2]):
            # Convert to coordinates of the density grid box
            box_x = point[0] + 0.5
            box_y = point[1] + 0.5
            box_z = point[2] + 0.5

            # Density grid location
            index_x = ti.cast(ti.floor(box_x * density_res), ti.i32)
            index_y = ti.cast(ti.floor(box_y * density_res), ti.i32)
            index_z = ti.cast(ti.floor(box_z * density_res), ti.i32)

            ti.atomic_add(field[view_id, y, x], density[index_z, index_y, index_x])

  kernel.materialize()
  kernel.grad.materialize()
  return kernel

ray_march_images = ray_march(images)
ray_march_target = ray_march(target_images)

@ti.kernel
def compute_loss(view_id: ti.i32):
  for i in range(res):
    for j in range(res):
      ti.atomic_add(loss, ti.sqr(images[view_id, i, j] - target_images[view_id, i, j]) * (1.0 / (res * res)))

@ti.kernel
def clear_images():
  for v, i, j in images:
    images[v, i, j] = 0

@ti.kernel
def clear_density():
  for i, j, k in density:
    density[i, j, k] = 0
    density.grad[i, j, k] = 0

def create_target_images():
  for view in range(n_views):
    ray_march_target(math.pi * 2 / n_views * view, view)

    img = np.zeros((res, res), dtype=np.float32)
    for i in range(res):
      for j in range(res):
        img[i, j] = target_images[view, i, j]

    imwrite("{}/target_{}.png".format("output_volume_renderer", view), 100 * img)

@ti.func
def in_torus(x, y, z):
  len_xz = ti.sqrt(x*x + z*z)
  qx = len_xz - torus_r1
  len_q = ti.sqrt(qx*qx + y*y)
  dist = len_q - torus_r2
  return dist < 0

@ti.kernel
def create_torus_density():
  for i in range(density_res):
    for j in range(density_res):
      for k in range(density_res):
        # Convert to density coordinates
        x = ti.cast(k, ti.f32) * inv_density_res - 0.5
        y = ti.cast(j, ti.f32) * inv_density_res - 0.5
        z = ti.cast(i, ti.f32) * inv_density_res - 0.5

        l = ti.sqrt(x*x + y*y + z*z)

        # Swap x, y to rotate the torus
        if in_torus(y, x, z):
          density[i, j, k] = inv_density_res
        else:
          density[i, j, k] = 0.0

@ti.kernel
def apply_grad():
  # gradient descent
  for i in range(density_res):
    for j in range(density_res):
      for k in range(density_res):
        density[i, j, k] -= learning_rate * density.grad[i, j, k]

def main():
  create_torus_density()
  create_target_images()
  clear_density()

  for iter in range(num_iterations):
    clear_images()
    with ti.Tape(loss):
      for view in range(n_views):
        ray_march_images(math.pi * 2 / n_views * view, view)
        compute_loss(view)

        img = np.zeros((res, res), dtype=np.float32)
        for i in range(res):
          for j in range(res):
            img[i, j] = images[view, i, j]

        imwrite("{}/image_{}.png".format("output_volume_renderer", view), 100 * img)

    print('Iter', iter, ' Loss =', loss[None])
    apply_grad()


if __name__ == '__main__':
  main()
