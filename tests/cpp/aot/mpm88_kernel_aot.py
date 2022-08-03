import argparse
import os

import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str)
args = parser.parse_args()


def compile_mpm88_graph(arch):
    n_particles = 8192 * 5
    n_grid = 128
    dt = 2e-4

    p_rho = 1
    gravity = 9.8
    bound = 3
    E = 400

    ti.init(arch=arch)

    @ti.kernel
    def substep_reset_grid(grid_v: ti.any_arr(field_dim=2),
                           grid_m: ti.any_arr(field_dim=2)):
        for i, j in grid_m:
            grid_v[i, j] = [0, 0]
            grid_m[i, j] = 0

    @ti.kernel
    def substep_p2g(x: ti.any_arr(field_dim=1), v: ti.any_arr(field_dim=1),
                    C: ti.any_arr(field_dim=1), J: ti.any_arr(field_dim=1),
                    grid_v: ti.any_arr(field_dim=2),
                    grid_m: ti.any_arr(field_dim=2)):
        for p in x:
            dx = 1 / grid_v.shape[0]
            p_vol = (dx * 0.5)**2
            p_mass = p_vol * p_rho
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

    @ti.kernel
    def substep_update_grid_v(grid_v: ti.any_arr(field_dim=2),
                              grid_m: ti.any_arr(field_dim=2)):
        for i, j in grid_m:
            num_grid = grid_v.shape[0]
            if grid_m[i, j] > 0:
                grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * gravity
            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0
            if i > num_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0
            if j > num_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0

    @ti.kernel
    def substep_g2p(x: ti.any_arr(field_dim=1), v: ti.any_arr(field_dim=1),
                    C: ti.any_arr(field_dim=1), J: ti.any_arr(field_dim=1),
                    grid_v: ti.any_arr(field_dim=2),
                    pos: ti.any_arr(field_dim=1)):
        for p in x:
            dx = 1 / grid_v.shape[0]
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
            pos[p] = [x[p][0], x[p][1], 0]
            J[p] *= 1 + dt * new_C.trace()
            C[p] = new_C

    @ti.kernel
    def init_particles(x: ti.any_arr(field_dim=1), v: ti.any_arr(field_dim=1),
                       J: ti.any_arr(field_dim=1)):
        for i in range(x.shape[0]):
            x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
            v[i] = [0, -1]
            J[i] = 1

    pos = ti.Vector.ndarray(3, ti.f32, n_particles)
    x = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))
    v = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))

    C = ti.Matrix.ndarray(2, 2, ti.f32, shape=(n_particles))
    J = ti.ndarray(ti.f32, shape=(n_particles))
    grid_v = ti.Vector.ndarray(2, ti.f32, shape=(n_grid, n_grid))
    grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid))

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    tmpdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    mod = ti.aot.Module(arch)

    mod.add_kernel(init_particles, template_args={'x': x, 'v': v, 'J': J})
    mod.add_kernel(substep_reset_grid,
                   template_args={
                       'grid_v': grid_v,
                       'grid_m': grid_m
                   })
    mod.add_kernel(substep_p2g,
                   template_args={
                       'x': x,
                       'v': v,
                       'C': C,
                       'J': J,
                       'grid_v': grid_v,
                       'grid_m': grid_m
                   })
    mod.add_kernel(substep_update_grid_v,
                   template_args={
                       'grid_v': grid_v,
                       'grid_m': grid_m
                   })
    mod.add_kernel(substep_g2p,
                   template_args={
                       'x': x,
                       'v': v,
                       'C': C,
                       'J': J,
                       'grid_v': grid_v,
                       'pos': pos
                   })

    mod.save(tmpdir, '')


if __name__ == "__main__":
    if args.arch == "cpu":
        compile_mpm88_graph(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_mpm88_graph(arch=ti.cuda)
    elif args.arch == "vulkan":
        compile_mpm88_graph(arch=ti.vulkan)
    else:
        assert False
