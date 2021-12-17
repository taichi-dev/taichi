import math
import os
import sys

from utils import get_benchmark_dir

import taichi as ti

sys.path.append(os.path.join(get_benchmark_dir(), '../', 'tests', 'python'))

from fuse_test_template import (template_fuse_dense_x2y2z,
                                template_fuse_reduction)
from utils import *


@benchmark_async
def chain_copy(scale):
    template_fuse_dense_x2y2z(size=scale * 1024**2,
                              repeat=1,
                              benchmark_repeat=100,
                              benchmark=True)


@benchmark_async
def increments(scale):
    template_fuse_reduction(size=scale * 1024**2,
                            repeat=10,
                            benchmark_repeat=10,
                            benchmark=True)


@benchmark_async
def fill_array(scale):
    a = ti.field(dtype=ti.f32, shape=scale * 1024**2)

    @ti.kernel
    def fill():
        for i in a:
            a[i] = 1.0

    def repeated_fill():
        for _ in range(10):
            fill()

    ti.benchmark(repeated_fill, repeat=10)


@benchmark_async
def fill_scalar(scale):
    a = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def fill():
        a[None] = 1.0

    def repeated_fill():
        for _ in range(1000):
            fill()

    ti.benchmark(repeated_fill, repeat=5)


@benchmark_async
def sparse_saxpy(scale):
    a = ti.field(dtype=ti.f32)
    b = ti.field(dtype=ti.f32)

    block_count = 2**int((math.log(scale, 2)) // 2) * 4
    block_size = 32
    # a, b always share the same sparsity
    ti.root.pointer(ti.ij, block_count).dense(ti.ij, block_size).place(a, b)

    @ti.kernel
    def initialize():
        for i, j in ti.ndrange(block_count * block_size,
                               block_count * block_size):
            if (i // block_size + j // block_size) % 4 == 0:
                a[i, j] = i + j

    @ti.kernel
    def saxpy(x: ti.template(), y: ti.template(), alpha: ti.f32):
        for i, j in x:
            y[i, j] = alpha * x[i, j] + y[i, j]

    def task():
        initialize()
        saxpy(a, b, 2)
        saxpy(b, a, 1.1)
        saxpy(b, a, 1.1)
        saxpy(a, b, 1.1)
        saxpy(a, b, 1.1)
        saxpy(a, b, 1.1)

    ti.benchmark(task, repeat=10)


@benchmark_async
def autodiff(scale):

    n = 1024**2 * scale

    a = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
    b = ti.field(dtype=ti.f32, shape=n)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def compute_loss():
        for i in a:
            loss[None] += a[i]

    @ti.kernel
    def accumulate_grad():
        for i in a:
            b[i] += a.grad[i]

    def task():
        for i in range(10):
            with ti.Tape(loss=loss):
                # The forward kernel of compute_loss should be completely eliminated (except for the last one)
                compute_loss()

            accumulate_grad()

    ti.benchmark(task, repeat=10)


@benchmark_async
def stencil_reduction(scale):
    a = ti.field(dtype=ti.f32)
    b = ti.field(dtype=ti.f32)
    total = ti.field(dtype=ti.f32, shape=())

    block_count = scale * 64
    block_size = 1024
    # a, b always share the same sparsity
    ti.root.pointer(ti.i, block_count).dense(ti.i, block_size).place(a, b)

    @ti.kernel
    def initialize():
        for i in range(block_size, block_size * (block_count - 1)):
            a[i] = i

    @ti.kernel
    def stencil():
        for i in a:
            b[i] = a[i - 1] + a[i] + a[i + 1]

    @ti.kernel
    def reduce():
        for i in a:
            total[None] += b[i]

    @ti.kernel
    def clear_b():
        for i in a:
            b[i] = 0

    def task():
        initialize()
        for i in range(3):
            stencil()
            reduce()
            clear_b()

    ti.benchmark(task, repeat=5)


@benchmark_async
def mpm_splitted(scale):
    quality = int(3 * scale**(1 / 3))
    # Use a larger value for higher-res simulations

    n_particles, n_grid = 9000 * quality**2, 128 * quality
    dx, inv_dx = 1 / n_grid, float(n_grid)
    dt = 1e-4 / quality
    p_vol, p_rho = (dx * 0.5)**2, 1
    p_mass = p_vol * p_rho
    E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
        (1 + nu) * (1 - 2 * nu))  # Lame parameters
    x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
    v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
    C = ti.Matrix.field(2, 2, dtype=float,
                        shape=n_particles)  # affine velocity field
    F = ti.Matrix.field(2, 2, dtype=float,
                        shape=n_particles)  # deformation gradient
    material = ti.field(dtype=int, shape=n_particles)  # material id
    Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
    grid_v = ti.Vector.field(2, dtype=float,
                             shape=(n_grid,
                                    n_grid))  # grid node momentum/velocity
    grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass

    @ti.kernel
    def substep():
        for i, j in grid_m:
            grid_v[i, j] = [0, 0]
            grid_m[i, j] = 0
        for p in x:
            F[p] = (ti.Matrix.identity(float, 2) +
                    dt * C[p]) @ F[p]  # deformation gradient update
        for p in x:  # Particle state update and scatter to grid (P2G)
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            h = ti.exp(
                10 * (1.0 - Jp[p])
            )  # Hardening coefficient: snow gets harder when compressed
            if material[p] == 1:  # jelly, make it softer
                h = 0.3
            mu, la = mu_0 * h, lambda_0 * h
            if material[p] == 0:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if material[p] == 2:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                  1 + 4.5e-3)  # Plasticity
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if material[
                    p] == 0:  # Reset deformation gradient to avoid numerical instability
                F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif material[p] == 2:
                F[p] = U @ sig @ V.transpose(
                )  # Reconstruct elastic deformation gradient after plasticity
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
            ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
            affine = stress + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(
                    3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base +
                       offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
        for i, j in grid_m:
            if grid_m[i, j] > 0:  # No need for epsilon here
                grid_v[i, j] = (
                    1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
                grid_v[i, j][1] -= dt * 50  # gravity
        for i, j in grid_m:
            if grid_m[i, j] > 0:  # No need for epsilon here
                if i < 3 and grid_v[i, j][0] < 0:
                    grid_v[i, j][0] = 0  # Boundary conditions
        for i, j in grid_m:
            if grid_m[i, j] > 0:  # No need for epsilon here
                if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
        for i, j in grid_m:
            if grid_m[i, j] > 0:  # No need for epsilon here
                if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
        for i, j in grid_m:
            if grid_m[i, j] > 0:  # No need for epsilon here
                if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
        for p in x:  # grid to particle (G2P)
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(
                    3, 3)):  # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            v[p], C[p] = new_v, new_C
        for p in x:
            x[p] += dt * v[p]  # advection

    group_size = n_particles // 3

    @ti.kernel
    def initialize():
        for i in range(n_particles):
            x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
            ]
            material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
            v[i] = ti.Matrix([0, 0])
            F[i] = ti.Matrix([[1, 0], [0, 1]])
            Jp[i] = 1

    initialize()

    def task():
        for s in range(int(2e-3 // dt)):
            substep()

    ti.benchmark(task, repeat=5)


@benchmark_async
def multires(scale):
    num_levels = 4

    x = []
    for i in range(num_levels):
        x.append(ti.field(dtype=ti.f32))

    # TODO: Using 1024 instead of 512 hangs the CUDA case. Need to figure out why.
    n = 512 * 1024 * scale

    block_size = 16
    assert n % block_size**2 == 0

    for i in range(num_levels):
        ti.root.pointer(ti.i, n // 2**i // block_size**2).pointer(
            ti.i, block_size).dense(ti.i, block_size).place(x[i])

    @ti.kernel
    def initialize():
        for i in range(n):
            x[0][i] = i

    @ti.kernel
    def downsample(l: ti.template()):
        for i in x[l]:
            if i % 2 == 0:
                x[l + 1][i // 2] = x[l][i]

    initialize()

    def task():
        for l in range(num_levels - 1):
            downsample(l)

    ti.benchmark(task, repeat=5)


@benchmark_async
def deep_hierarchy(scale):
    branching = 4
    num_levels = 8 + int(math.log(scale, branching))

    x = ti.field(dtype=ti.f32)

    n = 256 * 1024 * scale

    assert n % (branching**num_levels) == 0

    snode = ti.root
    for i in range(num_levels):
        snode = snode.pointer(ti.i, branching)

    snode.dense(ti.i, n // (branching**num_levels)).place(x)

    @ti.kernel
    def initialize():
        for i in range(n):
            x[i] = 0

    initialize()

    # Not fusible, but no modification to the mask/list of x either
    @ti.kernel
    def jitter():
        for i in x:
            if i % 2 == 0:
                x[i] += x[i + 1]

    def task():
        for i in range(5):
            jitter()

    ti.benchmark(task, repeat=5)
