import os

import pytest

import taichi as ti
from taichi import approx


def run_mpm88_test():
    dim = 2
    N = 64
    n_particles = N * N
    n_grid = 128
    dx = 1 / n_grid
    inv_dx = 1 / dx
    dt = 2.0e-4
    p_vol = (dx * 0.5)**2
    p_rho = 1
    p_mass = p_vol * p_rho
    E = 400

    x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    J = ti.field(dtype=ti.f32, shape=n_particles)
    grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

    @ti.kernel
    def substep():
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
            affine = ti.Matrix([[stress, 0], [0, stress]],
                               dt=ti.f32) + p_mass * C[p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (offset.cast(float) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    grid_v[base + offset].atomic_add(
                        weight * (p_mass * v[p] + affine @ dpos))
                    grid_m[base + offset].atomic_add(weight * p_mass)

        for i, j in grid_m:
            if grid_m[i, j] > 0:
                bound = 3
                inv_m = 1 / grid_m[i, j]
                grid_v[i, j] = inv_m * grid_v[i, j]
                grid_v[i, j][1] -= dt * 9.8
                if i < bound and grid_v[i, j][0] < 0:
                    grid_v[i, j][0] = 0
                if i > n_grid - bound and grid_v[i, j][0] > 0:
                    grid_v[i, j][0] = 0
                if j < bound and grid_v[i, j][1] < 0:
                    grid_v[i, j][1] = 0
                if j > n_grid - bound and grid_v[i, j][1] > 0:
                    grid_v[i, j][1] = 0

        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]
            new_v = ti.Vector.zero(ti.f32, 2)
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.Vector([i, j]).cast(float) - fx
                    g_v = grid_v[base + ti.Vector([i, j])]
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
            v[p] = new_v
            x[p] += dt * v[p]
            J[p] *= 1 + dt * new_C.trace()
            C[p] = new_C

    # gui = ti.core.GUI("MPM88", ti.veci(512, 512))
    # canvas = gui.get_canvas()

    for i in range(n_particles):
        x[i] = [i % N / N * 0.4 + 0.2, i / N / N * 0.4 + 0.05]
        v[i] = [0, -3]
        J[i] = 1

    for frame in range(10):
        for s in range(50):
            grid_v.fill([0, 0])
            grid_m.fill(0)
            substep()

    pos = x.to_numpy()
    pos[:, 1] *= 2
    regression = [
        0.31722742,
        0.15826741,
        0.10224003,
        0.07810827,
    ]
    for i in range(4):
        assert (pos**(i + 1)).mean() == approx(regression[i], rel=1e-2)


@ti.test()
def test_mpm88():
    run_mpm88_test()


def _is_appveyor():
    # AppVeyor adds `APPVEYOR=True` ('true' on Ubuntu)
    # https://www.appveyor.com/docs/environment-variables/
    return os.getenv('APPVEYOR', '').lower() == 'true'


#TODO: Remove exclude of ti.metal
@pytest.mark.skipif(_is_appveyor(), reason='Stuck on Appveyor.')
@ti.test(require=ti.extension.async_mode, exclude=[ti.metal], async_mode=True)
def test_mpm88_async():
    # It seems that all async tests on Appveyor run super slow. For example,
    # on Appveyor, 10+ tests have passed during the execution of
    # test_fuse_dense_x2y2z. Maybe thread synchronizations are expensive?
    run_mpm88_test()


@ti.test(exclude=[ti.vulkan])
def test_mpm88_numpy_and_ndarray():
    import numpy as np

    dim = 2
    N = 64
    n_particles = N * N
    n_grid = 128
    dx = 1 / n_grid
    inv_dx = 1 / dx
    dt = 2.0e-4
    p_vol = (dx * 0.5)**2
    p_rho = 1
    p_mass = p_vol * p_rho
    E = 400

    @ti.kernel
    def substep(x: ti.any_arr(element_dim=1), v: ti.any_arr(element_dim=1),
                C: ti.any_arr(element_dim=2), J: ti.any_arr(),
                grid_v: ti.any_arr(element_dim=1), grid_m: ti.any_arr()):
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
            affine = ti.Matrix([[stress, 0], [0, stress]],
                               dt=ti.f32) + p_mass * C[p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (offset.cast(float) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    grid_v[base + offset].atomic_add(
                        weight * (p_mass * v[p] + affine @ dpos))
                    grid_m[base + offset].atomic_add(weight * p_mass)

        for i, j in grid_m:
            if grid_m[i, j] > 0:
                bound = 3
                inv_m = 1 / grid_m[i, j]
                grid_v[i, j] = inv_m * grid_v[i, j]
                grid_v[i, j][1] -= dt * 9.8
                if i < bound and grid_v[i, j][0] < 0:
                    grid_v[i, j][0] = 0
                if i > n_grid - bound and grid_v[i, j][0] > 0:
                    grid_v[i, j][0] = 0
                if j < bound and grid_v[i, j][1] < 0:
                    grid_v[i, j][1] = 0
                if j > n_grid - bound and grid_v[i, j][1] > 0:
                    grid_v[i, j][1] = 0

        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]
            new_v = ti.Vector.zero(ti.f32, 2)
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.Vector([i, j]).cast(float) - fx
                    g_v = grid_v[base + ti.Vector([i, j])]
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
            v[p] = new_v
            x[p] += dt * v[p]
            J[p] *= 1 + dt * new_C.trace()
            C[p] = new_C

    def run_test(x, v, C, J, grid_v, grid_m):
        for i in range(n_particles):
            x[i] = [i % N / N * 0.4 + 0.2, i / N / N * 0.4 + 0.05]
            v[i] = [0, -3]
            J[i] = 1

        for frame in range(10):
            for s in range(50):
                grid_v.fill(0)
                grid_m.fill(0)
                substep(x, v, C, J, grid_v, grid_m)

        pos = x if isinstance(x, np.ndarray) else x.to_numpy()
        pos[:, 1] *= 2
        regression = [
            0.31722742,
            0.15826741,
            0.10224003,
            0.07810827,
        ]
        for i in range(4):
            assert (pos**(i + 1)).mean() == approx(regression[i], rel=1e-2)

    def test_numpy():
        x = np.zeros((n_particles, dim), dtype=np.float32)
        v = np.zeros((n_particles, dim), dtype=np.float32)
        C = np.zeros((n_particles, dim, dim), dtype=np.float32)
        J = np.zeros(n_particles, dtype=np.float32)
        grid_v = np.zeros((n_grid, n_grid, dim), dtype=np.float32)
        grid_m = np.zeros((n_grid, n_grid), dtype=np.float32)
        run_test(x, v, C, J, grid_v, grid_m)

    def test_ndarray():
        x = ti.Vector.ndarray(dim, ti.f32, n_particles)
        v = ti.Vector.ndarray(dim, ti.f32, n_particles)
        C = ti.Matrix.ndarray(dim, dim, ti.f32, n_particles)
        J = ti.ndarray(ti.f32, n_particles)
        grid_v = ti.Vector.ndarray(dim, ti.f32, (n_grid, n_grid))
        grid_m = ti.ndarray(ti.f32, (n_grid, n_grid))
        run_test(x, v, C, J, grid_v, grid_m)

    test_numpy()
    test_ndarray()
