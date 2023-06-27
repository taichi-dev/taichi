import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_aot_field_range_hint():
    density = ti.field(float, shape=(8, 8))

    @ti.kernel
    def init():
        for i, j in density:
            density[i, j] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module()
        m.add_field("density", density)
        m.add_kernel(init)
        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as json_file:
            res = json.load(json_file)
            for kernel in res["kernels"]:
                if kernel["name"] == "init":
                    range_hint2 = kernel["tasks_attribs"][0]["range_for_attribs"]
                    assert range_hint2["end"] - range_hint2["begin"] == 64


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_aot_bind_id():
    density = ti.field(dtype=ti.f32, shape=(8, 8))
    density1 = ti.ndarray(dtype=ti.math.ivec2, shape=(8, 8))

    @ti.kernel
    def init(x: ti.f32, density1: ti.types.ndarray(ndim=2)):
        for i, j in density1:
            density[i, j] = x
            density1[i, j] = x + 1

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module()
        m.add_kernel(init, {"density1": density1})
        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as json_file:
            res = json.load(json_file)
            for kernel in res["kernels"]:
                if kernel["name"] == "init":
                    buffer_binds = kernel["tasks_attribs"][0]["buffer_binds"]
                    for buffer_bind in buffer_binds:
                        if buffer_bind["buffer"]["type"] == 0:  # Root
                            assert buffer_bind["binding"] != -1
                        elif buffer_bind["buffer"]["type"] == 2:  # Rets
                            assert buffer_bind["binding"] != -1
                    args = kernel["ctx_attribs"]["arg_attribs_vec_"]
                    assert len(args) == 2
                    arg_0, arg_1 = args[0]["value"], args[1]["value"]
                    if args[0]["key"] == [1]:
                        arg_0, arg_1 = arg_1, arg_0

                    assert arg_0["is_array"] == False
                    assert arg_0["indices"] == [0]
                    assert arg_0["dtype"] == 1
                    assert arg_0["ptype"] == 0

                    assert arg_1["is_array"] == True
                    assert arg_1["field_dim"] == 2
                    assert arg_1["indices"] == [1]
                    assert arg_1["dtype"] == 5
                    assert arg_1["element_shape"] == [2]
                    assert arg_1["ptype"] == 1


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_save():
    density = ti.field(float, shape=(4, 4))

    @ti.kernel
    def init():
        for i, j in density:
            density[i, j] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module()
        m.add_field("density", density)
        m.add_kernel(init)
        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as json_file:
            json.load(json_file)


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_save_template_kernel():
    density = ti.field(float, shape=(4, 4))

    @ti.kernel
    def foo(n: ti.template()):
        for i in range(n):
            density[0, 0] += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module()
        m.add_field("density", density)
        with m.add_kernel_template(foo) as kt:
            kt.instantiate(n=6)
            kt.instantiate(n=8)
        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as json_file:
            json.load(json_file)


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_non_dense_snode():
    n = 8
    x = ti.field(dtype=ti.f32)
    y = ti.field(dtype=ti.f32)
    blk = ti.root.dense(ti.i, n)
    blk.place(x)
    blk.dense(ti.i, n).place(y)

    with pytest.raises(RuntimeError, match="AOT: only supports dense field"):
        m = ti.aot.Module()
        m.add_field("x", x)
        m.add_field("y", y)


@test_utils.test(arch=[ti.opengl, ti.gles, ti.vulkan])
def test_mpm88_aot():
    n_particles = 8192
    n_grid = 128
    dx = 1 / n_grid
    dt = 2e-4

    p_rho = 1
    p_vol = (dx * 0.5) ** 2
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
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
            affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
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
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
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
        m = ti.aot.Module()
        m.add_field("x", x)
        m.add_field("v", v)
        m.add_field("C", C)
        m.add_field("J", J)
        m.add_field("grid_v", grid_v)
        m.add_field("grid_m", grid_m)
        m.add_kernel(substep)
        m.add_kernel(init)
        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as json_file:
            json.load(json_file)


@test_utils.test(arch=[ti.opengl, ti.vulkan])
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
    def init(
        d: ti.i32,
        density1: ti.types.ndarray(),
        density2: ti.types.ndarray(),
        density3: ti.types.ndarray(),
        density4: ti.types.ndarray(),
        density5: ti.types.ndarray(),
        density6: ti.types.ndarray(),
    ):
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


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_mpm99_aot():
    quality = 1  # Use a larger value for higher-res simulations
    n_particles, n_grid = 9000 * quality**2, 128 * quality
    dx, inv_dx = 1 / n_grid, float(n_grid)
    dt = 1e-4 / quality
    p_vol, p_rho = (dx * 0.5) ** 2, 1
    p_mass = p_vol * p_rho
    E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
    v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
    C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
    F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
    material = ti.field(dtype=int, shape=n_particles)  # material id
    Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
    grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
    grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
    grid_v_int = ti.Vector.field(2, dtype=int, shape=(n_grid, n_grid))  # grid node momentum/velocity
    grid_m_int = ti.field(dtype=int, shape=(n_grid, n_grid))  # grid node mass

    v_exp = 24
    m_exp = 40

    @ti.kernel
    def substep():
        for i, j in grid_m:
            grid_v[i, j] = [0, 0]
            grid_m[i, j] = 0
            grid_v_int[i, j] = [0, 0]
            grid_m_int[i, j] = 0
        for p in x:  # Particle state update and scatter to grid (P2G)
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]  # deformation gradient update
            h = ti.exp(10 * (1.0 - Jp[p]))  # Hardening coefficient: snow gets harder when compressed
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
                    new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if material[p] == 0:  # Reset deformation gradient to avoid numerical instability
                F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif material[p] == 2:
                F[p] = U @ sig @ V.transpose()  # Reconstruct elastic deformation gradient after plasticity
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (
                J - 1
            )
            stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
            affine = stress + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_int[base + offset] += int(
                    ti.floor(0.5 + weight * (p_mass * v[p] + affine @ dpos) * (2.0**v_exp))
                )
                grid_m_int[base + offset] += int(ti.floor(0.5 + weight * p_mass * (2.0**m_exp)))
        for i, j in grid_m:
            if grid_m_int[i, j] > 0:  # No need for epsilon here
                # grid_v[i, j] = (1.0 / grid_m[i, j]) * grid_v[i, j] # Momentum to velocity
                grid_v[i, j] = (2 ** (m_exp - v_exp) / grid_m_int[i, j]) * grid_v_int[i, j]  # Momentum to velocity
                grid_v[i, j][1] -= dt * 50  # gravity
                if i < 3 and grid_v[i, j][0] < 0:
                    grid_v[i, j][0] = 0  # Boundary conditions
                if i > n_grid - 3 and grid_v[i, j][0] > 0:
                    grid_v[i, j][0] = 0
                if j < 3 and grid_v[i, j][1] < 0:
                    grid_v[i, j][1] = 0
                if j > n_grid - 3 and grid_v[i, j][1] > 0:
                    grid_v[i, j][1] = 0
        for p in x:  # grid to particle (G2P)
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            v[p], C[p] = new_v, new_C
            x[p] += dt * v[p]  # advection

    group_size = n_particles // 3

    @ti.kernel
    def initialize():
        for i in range(n_particles):
            x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size),
            ]
            material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
            v[i] = ti.Matrix([0, 0])
            F[i] = ti.Matrix([[1, 0], [0, 1]])
            Jp[i] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module()
        m.add_field("x", x)
        m.add_field("v", v)
        m.add_field("C", C)
        m.add_field("J", Jp)
        m.add_field("grid_v", grid_v)
        m.add_field("grid_m", grid_m)
        m.add_field("grid_v_int", grid_v_int)
        m.add_field("grid_m_int", grid_m_int)
        m.add_field("material", material)
        m.add_kernel(initialize)
        m.add_kernel(substep)

        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as json_file:
            json.load(json_file)


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_mpm88_ndarray():
    dim = 2
    N = 64
    n_particles = N * N
    n_grid = 128
    dx = 1 / n_grid
    inv_dx = 1 / dx
    dt = 2.0e-4
    p_vol = (dx * 0.5) ** 2
    p_rho = 1
    p_mass = p_vol * p_rho
    E = 400

    @ti.kernel
    def substep(
        x: ti.types.ndarray(dtype=ti.math.vec2),
        v: ti.types.ndarray(dtype=ti.math.vec2),
        C: ti.types.ndarray(dtype=ti.math.mat2),
        J: ti.types.ndarray(),
        grid_v: ti.types.ndarray(dtype=ti.math.vec2),
        grid_m: ti.types.ndarray(),
    ):
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
            affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (offset.cast(float) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    ti.atomic_add(grid_v[base + offset], weight * (p_mass * v[p] + affine @ dpos))
                    ti.atomic_add(grid_m[base + offset], weight * p_mass)

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
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
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
        m = ti.aot.Module()
        template_args = {
            "x": x,
            "v": v,
            "C": C,
            "J": J,
            "grid_m": grid_m,
            "grid_v": grid_v,
        }
        m.add_kernel(substep, template_args=template_args)

        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as json_file:
            json.load(json_file)


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_aot_ndarray_template_mixed():
    @ti.kernel
    def run(arr: ti.types.ndarray(), val1: ti.f32, val2: ti.template()):
        for i in arr:
            arr[i] = val1 + val2

    with tempfile.TemporaryDirectory() as tmpdir:
        x = ti.ndarray(dtype=ti.f32, shape=16)
        m = ti.aot.Module()
        m.add_kernel(run, template_args={"arr": x, "val2": 42})
        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as json_file:
            res = json.load(json_file)
            for kernel in res["kernels"]:
                if kernel["name"] == "run":
                    args_count = len(kernel["ctx_attribs"]["arg_attribs_vec_"])
                    assert args_count == 2, res  # `arr` and `val1`


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_aot_ndarray_without_template_args():
    @ti.kernel
    def kernel1(arr: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for I in ti.grouped(arr):
            arr[I] = 0.0

    @ti.kernel
    def kernel2(arr: ti.types.ndarray(dtype=ti.math.vec2, ndim=2)):
        for I in ti.grouped(arr):
            arr[I] = 0.0

    @ti.kernel
    def kernel3(arr: ti.types.ndarray(dtype=ti.math.mat2, ndim=2)):
        for I in ti.grouped(arr):
            arr[I] = 0.0

    m = ti.aot.Module()
    m.add_kernel(kernel1)
    m.add_kernel(kernel2)
    m.add_kernel(kernel3)


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_archive():
    density = ti.field(float, shape=(4, 4))

    @ti.kernel
    def init():
        for i, j in density:
            density[i, j] = 1

    def sha1sum(filename):
        with open(filename, "rb") as f:
            data = f.read()
            return hashlib.sha1(data).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module()
        m.add_field("density", density)
        m.add_kernel(init)
        tcm_path = f"{tmpdir}/x.tcm"
        m.archive(tcm_path)
        first_sha = sha1sum(tcm_path)
        with zipfile.ZipFile(tcm_path, "r") as z:
            assert z.read("__version__") == bytes(".".join(str(x) for x in ti.__version__), "utf-8")
        os.remove(tcm_path)
        m.archive(tcm_path)
        second_sha = sha1sum(tcm_path)
        assert first_sha == second_sha


@test_utils.test(arch=[ti.opengl, ti.vulkan])
def test_sequential_dispatch():
    g_init_builder = ti.graph.GraphBuilder()
    g_init_substep = g_init_builder.create_sequential()

    ivec3 = ti.types.vector(3, ti.i32)

    @ti.kernel
    def init_data(test_vec: ivec3):
        pass

    sym_args = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "test_arg", ti.types.vector(3, ti.i32))

    g_init_substep.dispatch(init_data, sym_args)
    g_init_builder.append(g_init_substep)

    g_init = g_init_builder.compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module()
        m.add_graph("g_init", g_init)
        m.save(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json"), "r") as json_file:
            json.load(json_file)


@test_utils.test(arch=[ti.vulkan])
def test_vulkan_cgraph_short():
    a = ti.ndarray(ti.u8, shape=(16))
    c = 2

    @ti.kernel
    def test(a: ti.types.ndarray(), c: ti.u8):
        for i in a:
            a[i] = i + c

    sym_a = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "a", ti.u8, ndim=1)
    sym_c = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "c", ti.u8)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_a, sym_c)
    g = g_init.compile()

    g.run({"a": a, "c": c})

    m = ti.aot.Module(caps=[ti.DeviceCapability.spirv_has_int8])
    m.add_graph("g_init", g)
    with tempfile.TemporaryDirectory() as tmpdir:
        m.save(tmpdir)


@test_utils.test(arch=[ti.vulkan])
def test_devcap():
    module = ti.aot.Module(
        ti.vulkan,
        caps=[
            ti.DeviceCapability.spirv_has_float16,
            ti.DeviceCapability.spirv_has_atomic_float16_minmax,
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        module.save(tmpdir)

        with open(tmpdir + "/metadata.json") as f:
            j = json.load(f)
            caps = j["required_caps"]
            for cap in caps:
                key = cap["key"]
                if key == "spirv_version":
                    assert cap["value"] == 0x10300
                elif key == "spirv_has_float16":
                    assert cap["value"] == 1
                elif key == "spirv_has_atomic_float16_minmax":
                    assert cap["value"] == 1
                else:
                    assert False


@test_utils.test(arch=[ti.vulkan])
def test_devcap_weird_user_input():
    with pytest.raises(RuntimeError, match="unexpected device capability name"):
        ti.aot.Module(
            ti.vulkan,
            caps=[
                "Never gonna give you up"
                "Never gonna let you down"
                "Never gonna run around and desert you"
                "Never gonna make you cry"
                "Never gonna say goodbye"
                "Never gonna tell a lie and hurt you"
            ],
        )


@test_utils.test(arch=[ti.vulkan])
def test_module_arch_fallback():
    with pytest.warns(
        Warning,
        match=r"AOT compilation to a different arch than the current one is not yet supported, switching",
    ):
        m = ti.aot.Module(ti.cpu)


@test_utils.test(arch=[ti.vulkan])
def test_save_kernel_with_rwtexture():
    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    m = ti.aot.Module()
    m.add_kernel(write)
    with tempfile.TemporaryDirectory() as tmpdir:
        m.save(tmpdir)


@test_utils.test(arch=[ti.vulkan])
def test_read_kernel_with_texture():
    @ti.kernel
    def read(tex: ti.types.texture(num_dimensions=2), arr: ti.types.ndarray()):
        for i, j in arr:
            arr[i, j] = tex.fetch(ti.Vector([i, j]), 0).x

    res = (128, 128)
    tex = ti.Texture(ti.Format.r32f, res)
    arr = ti.ndarray(ti.f32, res)

    m = ti.aot.Module()
    m.add_kernel(read, template_args={"tex": tex, "arr": arr})
    with tempfile.TemporaryDirectory() as tmpdir:
        m.save(tmpdir)


@test_utils.test(arch=[ti.vulkan])
def test_rwtexture_with_ndarray():
    @ti.kernel
    def init_texture_from_ndarray(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
        img: ti.types.ndarray(ndim=2),
    ):
        for i, j in img:
            tex.store(ti.Vector([i, j]), ti.Vector([img[i, j], 0.0, 0.0, 0.0]) / 255.0)

    m = ti.aot.Module()
    tex = ti.Texture(ti.Format.r32f, (128, 128))
    img = ti.ndarray(ti.f32, (128, 128))
    m.add_kernel(
        init_texture_from_ndarray,
        template_args={
            "img": img,
        },
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        m.save(tmpdir)
