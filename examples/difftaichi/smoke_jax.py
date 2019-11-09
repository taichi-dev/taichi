# Fluid simulation from autograd:
# https://github.com/HIPS/autograd/blob/master/examples/fluidsim/fluidsim.py

from __future__ import absolute_import
from __future__ import print_function
from builtins import range
import jax.numpy as np
from jax import value_and_grad
from jax import jit
from jax import device_put
from jax import vjp

from scipy.optimize import minimize
from imageio import imread

import cv2

import matplotlib
import matplotlib.pyplot as plt
import os

n_grid = 110

dtype = np.float64

# Fluid simulation code based on
# "Real-Time Fluid Dynamics for Games" by Jos Stam
# http://www.intpowertechcorp.com/GDC03.pdf

@jit
def project(vx, vy):
    """Project the velocity field to be approximately mass-conserving,
       using a few iterations of Gauss-Seidel."""
    p = np.zeros(vx.shape, dtype=dtype)
    h = 1.0/vx.shape[0]
    div = -0.5 * h * (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)
                    + np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1))

    # Note: for some reason JAX crashes when num_iteration != 5/6.
    for k in range(6):
        p = (div + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)
                 + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1))/4.0

    vx -= 0.5*(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/h
    vy -= 0.5*(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))/h
    return vx, vy

@jit
def d_project(vx, vy, d_vx, d_vy):
    _, fun = vjp(project, vx, vy)
    return fun((d_vx, d_vy))

@jit
def advect(f, vx, vy):
    """Move field f according to x and y velocities (u and v)
       using an implicit Euler integrator."""
    rows, cols = f.shape
    cell_ys, cell_xs = np.meshgrid(np.arange(rows), np.arange(cols))
    cell_xs = cell_xs.astype(dtype)
    cell_ys = cell_ys.astype(dtype)
    center_xs = (cell_xs - vx).ravel()
    center_ys = (cell_ys - vy).ravel()

    # Compute indices of source cells.
    left_ix = np.floor(center_xs).astype(int)
    top_ix  = np.floor(center_ys).astype(int)
    rw = center_xs - left_ix              # Relative weight of right-hand cells.
    bw = center_ys - top_ix               # Relative weight of bottom cells.
    left_ix  = np.mod(left_ix,     rows)  # Wrap around edges of simulation.
    right_ix = np.mod(left_ix + 1, rows)
    top_ix   = np.mod(top_ix,      cols)
    bot_ix   = np.mod(top_ix  + 1, cols)

    # A linearly-weighted sum of the 4 surrounding cells.
    flat_f = (1 - rw) * ((1 - bw)*f[left_ix,  top_ix] + bw*f[left_ix,  bot_ix]) \
                 + rw * ((1 - bw)*f[right_ix, top_ix] + bw*f[right_ix, bot_ix])
    return np.reshape(flat_f, (rows, cols))

@jit
def d_advect(f, vx, vy, d_f):
    _, fun = vjp(advect, f, vx, vy)
    return fun(d_f)

def simulate(vx, vy, smoke, num_time_steps, ax=None, render=False):
    print("Running simulation...")
    for t in range(num_time_steps):
        if ax: plot_matrix(ax, smoke, t, render)
        vx_updated = advect(vx, vx, vy)
        vy_updated = advect(vy, vx, vy)
        vx, vy = project(vx_updated, vy_updated)
        smoke = advect(smoke, vx, vy)
    if ax: plot_matrix(ax, smoke, num_time_steps, render)
    return smoke

def d_simulate(vx, vy, smoke, target, num_time_steps, ax=None, render=False):
    print("Running differentiated simulation...")
    states = []
    for t in range(num_time_steps):
        states.append((vx, vy, smoke))
        vx_updated = advect(vx, vx, vy)
        vy_updated = advect(vy, vx, vy)
        vx, vy = project(vx_updated, vy_updated)
        smoke = advect(smoke, vx, vy)

    d_smoke = 2 * (smoke - target)

    d_vx = np.zeros_like(vx)
    d_vy = np.zeros_like(vy)
    for t in range(num_time_steps-1, -1, -1):
        vx, vy, smoke = states[t]
        vx_updated = advect(vx, vx, vy)
        vy_updated = advect(vy, vx, vy)
        new_vx, new_vy = project(vx_updated, vy_updated)
        # new_smoke = advect(smoke, new_vx, new_vy)

        d_smoke, d_vx, d_vy = d_advect(smoke, vx, vy, d_smoke)
        d_vx_updated, d_vy_updated = d_project(vx_updated, vy_updated, d_vx, d_vy)
        d_vy0, d_vx, d_vy1 = d_advect(vy, vx, vy, d_vy_updated)
        d_vy += d_vy0 + d_vy1
        d_vx0, d_vx1, d_vy = d_advect(vx, vx, vy, d_vx_updated)
        d_vx += d_vx0 + d_vx1

    return d_vx, d_vy

def plot_matrix(ax, mat, t, render=False):
    plt.cla()
    ax.matshow(mat)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    if render:
        matplotlib.image.imsave('output_autograd/step{0:03d}.png'.format(t), mat)
    plt.pause(0.001)


if __name__ == '__main__':
    simulation_timesteps = 100
    basepath = os.path.dirname(__file__)
    os.system("mkdir -p output_autograd")

    print("Loading initial and target states...")
    init_smoke = cv2.resize(imread(os.path.join(basepath, 'init_smoke.png')), (n_grid, n_grid))[:, :, 0]
    target = cv2.resize(imread('taichi.png'), (n_grid, n_grid))[:, :, 0]
    init_smoke = device_put(init_smoke.astype(np.float32))
    target = device_put(target.astype(np.float32))
    rows, cols = target.shape

    init_dx_and_dy = np.zeros((2, rows, cols), dtype=dtype).ravel()

    def distance_from_target_image(smoke):
        print(smoke.dtype)
        return np.mean((target - smoke)**2)

    def convert_param_vector_to_matrices(params):
        vx = np.reshape(params[:(rows*cols)], (rows, cols))
        vy = np.reshape(params[(rows*cols):], (rows, cols))
        return vx, vy

    def objective(params):
        init_vx, init_vy = convert_param_vector_to_matrices(params)
        final_smoke = simulate(init_vx, init_vy, init_smoke, simulation_timesteps)
        return distance_from_target_image(final_smoke)

    # Specify gradient of objective function using JAX.
    #objective_with_grad = value_and_grad(objective)

    import time
    for i in range(10):
      t = time.time()
      r = objective(init_dx_and_dy)
      print('loss dtype', r.dtype)
      print('forward time', (time.time() - t) * 1000, 'ms/iter')
      t = time.time()
      #r = objective_with_grad(init_dx_and_dy)
      init_vx, init_vy = convert_param_vector_to_matrices(init_dx_and_dy)
      r = d_simulate(init_vx, init_vy, init_smoke, target, simulation_timesteps)
      print('total time', (time.time() - t) * 1000, 'ms/iter')

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, frameon=False)

    def callback(params):
        init_vx, init_vy = convert_param_vector_to_matrices(params)
        simulate(init_vx, init_vy, init_smoke, simulation_timesteps, ax)

    print("Optimizing initial conditions...")
    result = minimize(objective_with_grad, init_dx_and_dy, jac=True, method='CG',
                      options={'maxiter':10, 'disp':False}, callback=callback)

    print("Rendering optimized flow...")
    init_vx, init_vy = convert_param_vector_to_matrices(result.x)
    simulate(init_vx, init_vy, init_smoke, simulation_timesteps, ax, render=True)

    print("Converting frames to an animated GIF...")
    os.system("convert -delay 5 -loop 0 output_autograd/step*.png"
              " -delay 250 output_autograd/step100.png output_autograd/surprise.gif")  # Using imagemagick.
