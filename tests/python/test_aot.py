import json
import os
import sys
import tempfile

import numpy as np
import pytest

import taichi as ti


@ti.test(arch=ti.cc)
def test_record():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorded_file = os.path.join(tmpdir, 'record.yml')
        ti.aot.start_recording(recorded_file)

        loss = ti.field(float, (), needs_grad=True)
        x = ti.field(float, 233, needs_grad=True)

        @ti.kernel
        def compute_loss():
            for i in x:
                loss[None] += x[i]**2

        compute_loss()
        ti.aot.stop_recording()

        assert os.path.exists(recorded_file)

        # Make sure kernel info is in the file
        with open(recorded_file, 'r') as f:
            assert 'compute_loss' in ''.join(f.readlines())


@ti.test(arch=ti.opengl)
def test_save():
    density = ti.field(float, shape=(4, 4))

    @ti.kernel
    def init():
        for i, j in density:
            density[i, j] = 1

    @ti.kernel
    def foo(n: ti.template()):
        for i in range(n):
            density[0, 0] += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module(ti.opengl)
        m.add_field('density', density)
        m.add_kernel(init)
        with m.add_kernel_template(foo) as kt:
            kt.instantiate(n=6)
            kt.instantiate(n=8)
        m.save(tmpdir, '')
        with open(os.path.join(tmpdir, 'metadata.json')) as json_file:
            json.load(json_file)


@ti.test(arch=ti.opengl)
def test_non_dense_snode():
    n = 8
    x = ti.field(dtype=ti.f32)
    y = ti.field(dtype=ti.f32)
    blk = ti.root.dense(ti.i, n)
    blk.place(x)
    blk.dense(ti.i, n).place(y)

    with pytest.raises(RuntimeError, match='AOT: only supports dense field'):
        m = ti.aot.Module(ti.opengl)
        m.add_field('x', x)
        m.add_field('y', y)


@ti.test(arch=ti.opengl)
def test_mpm88_aot():
    n_particles = 8192
    n_grid = 128
    dx = 1 / n_grid
    dt = 2e-4

    p_rho = 1
    p_vol = (dx * 0.5)**2
    p_mass = p_vol * p_rho
    gravity = 9.8
    bound = 3
    E = 400

    x = ti.Vector.field(2, float, n_particles)
    v = ti.Vector.field(2, float, n_particles)
    C = ti.Matrix.field(2, 2, float, n_particles)
    J = ti.field(float, n_particles)

    grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
    grid_m = ti.field(float, (n_grid, n_grid))

    @ti.kernel
    def substep():
        for i, j in grid_m:
            grid_v[i, j] = [0, 0]
            grid_m[i, j] = 0
        for p in x:
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
            affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base +
                       offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
        for i, j in grid_m:
            if grid_m[i, j] > 0:
                grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * gravity
            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0
            if i > n_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0
            if j > n_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0
        for p in x:
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset - fx) * dx
                weight = w[i].x * w[j].y
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
            v[p] = new_v
            x[p] += dt * v[p]
            J[p] *= 1 + dt * new_C.trace()
            C[p] = new_C

    @ti.kernel
    def init():
        for i in range(n_particles):
            x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
            v[i] = [0, -1]
            J[i] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module(ti.opengl)
        m.add_field("x", x)
        m.add_field("v", v)
        m.add_field("C", C)
        m.add_field("J", J)
        m.add_field("grid_v", grid_v)
        m.add_field("grid_m", grid_m)
        m.add_kernel(substep)
        m.add_kernel(init)
        m.save(tmpdir, '')
        with open(os.path.join(tmpdir, 'metadata.json')) as json_file:
            json.load(json_file)


@ti.test(arch=ti.opengl)
def test_opengl_8_ssbo():
    # 6 ndarrays + gtmp + args
    n = 4
    density1 = ti.ndarray(dtype=ti.f32, shape=(4, 4))
    density2 = ti.ndarray(dtype=ti.f32, shape=(4, 4))
    density3 = ti.ndarray(dtype=ti.f32, shape=(4, 4))
    density4 = ti.ndarray(dtype=ti.f32, shape=(4, 4))
    density5 = ti.ndarray(dtype=ti.f32, shape=(4, 4))
    density6 = ti.ndarray(dtype=ti.f32, shape=(4, 4))

    @ti.kernel
    def init(d: ti.i32, density1: ti.any_arr(), density2: ti.any_arr(),
             density3: ti.any_arr(), density4: ti.any_arr(),
             density5: ti.any_arr(), density6: ti.any_arr()):
        for i, j in density1:
            density1[i, j] = d + 1
            density2[i, j] = d + 2
            density3[i, j] = d + 3
            density4[i, j] = d + 4
            density5[i, j] = d + 5
            density6[i, j] = d + 6

    init(0, density1, density2, density3, density4, density5, density6)
    assert (density1.to_numpy() == (np.zeros(shape=(n, n)) + 1)).all()
    assert (density2.to_numpy() == (np.zeros(shape=(n, n)) + 2)).all()
    assert (density3.to_numpy() == (np.zeros(shape=(n, n)) + 3)).all()
    assert (density4.to_numpy() == (np.zeros(shape=(n, n)) + 4)).all()
    assert (density5.to_numpy() == (np.zeros(shape=(n, n)) + 5)).all()
    assert (density6.to_numpy() == (np.zeros(shape=(n, n)) + 6)).all()


@ti.test(arch=ti.opengl)
def test_opengl_exceed_max_ssbo():
    # 7 ndarrays + gtmp + args > 8 (maximum allowed)
    n = 4
    density1 = ti.ndarray(dtype=ti.f32, shape=(n, n))
    density2 = ti.ndarray(dtype=ti.f32, shape=(n, n))
    density3 = ti.ndarray(dtype=ti.f32, shape=(n, n))
    density4 = ti.ndarray(dtype=ti.f32, shape=(n, n))
    density5 = ti.ndarray(dtype=ti.f32, shape=(n, n))
    density6 = ti.ndarray(dtype=ti.f32, shape=(n, n))
    density7 = ti.ndarray(dtype=ti.f32, shape=(n, n))

    @ti.kernel
    def init(d: ti.i32, density1: ti.any_arr(), density2: ti.any_arr(),
             density3: ti.any_arr(), density4: ti.any_arr(),
             density5: ti.any_arr(), density6: ti.any_arr(),
             density7: ti.any_arr()):
        for i, j in density1:
            density1[i, j] = d + 1
            density2[i, j] = d + 2
            density3[i, j] = d + 3
            density4[i, j] = d + 4
            density5[i, j] = d + 5
            density6[i, j] = d + 6
            density7[i, j] = d + 7

    with pytest.raises(RuntimeError):
        init(0, density1, density2, density3, density4, density5, density6,
             density7)


@ti.test(arch=ti.opengl)
def test_mpm88_ndarray():
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
            affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
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

    x = ti.Vector.ndarray(dim, ti.f32, n_particles)
    v = ti.Vector.ndarray(dim, ti.f32, n_particles)
    C = ti.Matrix.ndarray(dim, dim, ti.f32, n_particles)
    J = ti.ndarray(ti.f32, n_particles)
    grid_v = ti.Vector.ndarray(dim, ti.f32, (n_grid, n_grid))
    grid_m = ti.ndarray(ti.f32, (n_grid, n_grid))

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module(ti.opengl)
        m.add_kernel(substep, (x, v, C, J, grid_v, grid_m))

        m.save(tmpdir, '')
        with open(os.path.join(tmpdir, 'metadata.json')) as json_file:
            json.load(json_file)
