import taichi_lang as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.set_default_fp(real)

max_steps = 4096
vis_interval = 128
output_vis_interval = 2
steps = 512
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

loss = scalar()

# On CPU this may not work since during AD, min's adjoint uses cmp_le, which might return -1 instead of 1
ti.cfg.arch = ti.cuda

x = vec()
v = vec()
rotation = scalar()
# angular velocity
omega = scalar()

halfsize = vec()

inverse_mass = scalar()
inverse_inertia = scalar()

v_inc = vec()
omega_inc = scalar()

head_id = 0
goal = [0.3, 0.2]

n_objects = 3
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -9.8
friction = 0.0
penalty = 1e4
damping = 0

n_springs = 1
spring_anchor_a = ti.global_var(ti.i32)
spring_anchor_b = ti.global_var(ti.i32)
spring_length = scalar()
spring_offset_a = vec()
spring_offset_b = vec()
spring_stiffness = scalar()


@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, rotation,
                                                              omega, v_inc,
                                                              omega_inc)
  ti.root.dense(ti.i, n_objects).place(halfsize, inverse_mass, inverse_inertia)
  ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                       spring_length, spring_offset_a,
                                       spring_offset_b, spring_stiffness)
  ti.root.place(loss)
  ti.root.lazy_grad()


dt = 0.001
learning_rate = 0.001


@ti.func
def rotation_matrix(r):
  return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])


@ti.kernel
def initialize_properties():
  for i in range(n_objects):
    inverse_mass[i] = 1.0 / (4 * halfsize[i][0] * halfsize[i][1])
    inverse_inertia[i] = 1.0 / (4 / 3 * halfsize[i][0] * halfsize[i][1] * (
        halfsize[i][0] * halfsize[i][0] + halfsize[i][1] * halfsize[i][1]))
    # ti.print(inverse_mass[i])
    # ti.print(inverse_inertia[i])


@ti.func
def cross(a, b):
  return a[0] * b[1] - a[1] * b[0]

@ti.func
def to_world(t, i, rela_x):
  rot = rotation[t, i]
  rot_matrix = rotation_matrix(rot)
  
  rela_pos = rot_matrix @ rela_x
  rela_v = omega[t, i] * ti.Vector([-rela_pos[1], rela_pos[0]])
  
  world_x = x[t, i] + rela_pos
  world_v = v[t, i] + rela_v
  
  return world_x, world_v, rela_pos


@ti.func
def apply_impulse(t, i, impulse, location):
  ti.atomic_add(v_inc[t + 1, i], impulse * inverse_mass[i])
  ti.atomic_add(omega_inc[t + 1, i],
                cross(location - x[t, i], impulse) * inverse_inertia[i])


@ti.kernel
def collide(t: ti.i32):
  for i in range(n_objects):
    hs = halfsize[i]
    for k in ti.static(range(4)):
      # the corner for collision detection
      offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1])
      
      corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)
      corner_v = corner_v + dt * gravity * ti.Vector([0.0, 1.0])
      
      # Apply impulse so that there's no sinking
      normal = ti.Vector([0.0, 1.0])
      tao = ti.Vector([1.0, 0.0])
      
      rn = cross(rela_pos, normal)
      rt = cross(rela_pos, tao)
      impulse_contribution = inverse_mass[i] + ti.sqr(rn) * \
                             inverse_inertia[i]
      timpulse_contribution = inverse_mass[i] + ti.sqr(rt) * \
                              inverse_inertia[i]
      
      # ti.print(impulse_contribution)
      
      rela_v_ground = normal.dot(corner_v)
      
      impulse = 0.0
      timpulse = 0.0
      if rela_v_ground < 0 and corner_x[1] < ground_height:
        impulse = -(1 + elasticity) * rela_v_ground / impulse_contribution
        if impulse > 0:
          # friction
          timpulse = -corner_v.dot(tao) / timpulse_contribution
          timpulse = ti.min(friction * impulse,
                            ti.max(-friction * impulse, timpulse))
      
      if corner_x[1] < ground_height:
        # apply penalty
        impulse = impulse - dt * penalty * (
            corner_x[1] - ground_height) / impulse_contribution

      apply_impulse(t, i, impulse * normal + timpulse * tao, corner_x)

@ti.kernel
def apply_spring_force(t: ti.i32):
  for i in range(n_springs):
    a = spring_anchor_a[i]
    b = spring_anchor_b[i]
    pos_a, _, rela_a = to_world(t, a, spring_offset_a[i])
    pos_b, _, rela_b = to_world(t, b, spring_offset_b[i])
    dist = pos_a - pos_b
    length = dist.norm()
    impulse = dt * (length - spring_length[i]) * spring_stiffness[i] * ti.Vector.normalized(dist)
    apply_impulse(t, a, -impulse, pos_a)
    apply_impulse(t, b, impulse, pos_b)



@ti.kernel
def advance(t: ti.i32):
  for i in range(n_objects):
    s = math.exp(-dt * damping)
    v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
      [0.0, 1.0])
    x[t, i] = x[t - 1, i] + dt * v[t, i]
    omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
    rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
  loss[None] = (x[t, head_id] - ti.Vector(goal)).norm_sqr()


def forward(output=None):
  initialize_properties()
  
  interval = vis_interval
  if output:
    interval = output_vis_interval
    os.makedirs('rigid_body/{}/'.format(output), exist_ok=True)
  
  for t in range(1, steps):
    collide(t - 1)
    apply_spring_force(t - 1)
    advance(t)
    
    if (t + 1) % interval == 0:
      img = np.ones(shape=(vis_resolution, vis_resolution, 3),
                    dtype=np.float32) * 0.8
      
      color = (0.3, 0.5, 0.8)
      for i in range(n_objects):
        points = []
        for k in range(4):
          offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
          rot = rotation[t, i]
          rot_matrix = np.array(
            [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
          
          pos = np.array(
            [x[t, i][0], x[t, i][1]]) + offset_scale * rot_matrix @ np.array(
            [halfsize[i][0], halfsize[i][1]])
          points.append(
            (int(pos[0] * vis_resolution),
             vis_resolution - int(pos[1] * vis_resolution)))
        
        cv2.fillConvexPoly(img, points=np.array(points), color=color)
      
      y = int((1 - ground_height) * vis_resolution)
      cv2.line(img, (0, y), (vis_resolution - 2, y), color=(0.1, 0.1, 0.1),
               thickness=4)
      
      def circle(x, y, color):
        radius = 0.02
        cv2.circle(img, center=(
          int(vis_resolution * x), int(vis_resolution * (1 - y))),
                   radius=int(radius * vis_resolution), color=color,
                   thickness=-1)

      circle(x[t, head_id][0], x[t, head_id][1], (0.4, 0.6, 0.6))
      circle(goal[0], goal[1], (0.6, 0.2, 0.2))

      for i in range(n_springs):
        def get_world_loc(i, offset):
          rot = rotation[t, i]
          rot_matrix = np.array(
            [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
          pos = np.array(
            [[x[t, i][0]], [x[t, i][1]]]) + rot_matrix @ np.array([[offset[0]], [offset[1]]])
          pos = pos * vis_resolution
          return (int(pos[0, 0]), vis_resolution - int(pos[1, 0]))
        pt1 = get_world_loc(spring_anchor_a[i], spring_offset_a[i])
        pt2 = get_world_loc(spring_anchor_b[i], spring_offset_b[i])
        cv2.line(img, pt1, pt2, (0.2, 0.2, 0.4), thickness=6)
    
      
      cv2.imshow('img', img)
      cv2.waitKey(1)
      if output:
        cv2.imwrite('rigid_body/{}/{:04d}.png'.format(output, t), img * 255)
  
  loss[None] = 0
  compute_loss(steps - 1)


@ti.kernel
def clear():
  for t in range(0, max_steps):
    for i in range(0, n_objects):
      x.grad[t, i] = ti.Vector([0.0, 0.0])
      v.grad[t, i] = ti.Vector([0.0, 0.0])
      rotation.grad[t, i] = 0.0
      omega.grad[t, i] = 0.0
      
      v_inc[t, i] = ti.Vector([0.0, 0.0])
      omega_inc[t, i] = 0.0
      
      v_inc.grad[t, i] = ti.Vector([0.0, 0.0])
      omega_inc.grad[t, i] = 0.0


@ti.kernel
def clear2():
  for i in range(0, n_objects):
    halfsize.grad[i] = [0.0, 0.0]
    inverse_inertia.grad[i] = 0
    inverse_mass.grad[i] = 0

def add_spring():
  i = 0
  spring_anchor_a[i] = 0
  spring_anchor_b[i] = 1
  spring_length[i] = 0.2
  spring_offset_a[i] = [0.05, 0.01]
  spring_offset_b[i] = [0.0, 0.0]
  spring_stiffness[i] = 10
def main():
  for i in range(n_objects):
    x[0, i] = [0.5 + 0.2 * i, 0.5 + 0.3 * i]
    halfsize[i] = [0.08, 0.03]
    rotation[0, i] = math.pi / 4 + 0.01
    omega[0, i] = 5

  add_spring()


  forward('initial')
  for iter in range(300):
    clear()
    clear2()
    loss.grad[None] = -1

    tape = ti.tape()

    with tape:
      forward()

    tape.grad()
    
    print('Iter=', iter, 'Loss=', loss[None])
    
    for i in range(n_objects):
      for d in range(2):
        print(halfsize.grad[i][d])
        halfsize[i][d] += learning_rate * halfsize.grad[i][d]

  
  clear()
  clear2()
  forward('final')


if __name__ == '__main__':
  main()
