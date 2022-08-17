import argparse
import os

import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str)
parser.add_argument("--cgraph", action='store_true', default=False)
args = parser.parse_args()


def compile_mpm88(arch, save_compute_graph):
    ti.init(arch, vk_api_version="1.0")

    if ti.lang.impl.current_cfg().arch != arch:
        return

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

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    tmpdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    if save_compute_graph:
        N_ITER = 50

        sym_x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                             'x',
                             ti.f32,
                             field_dim=1,
                             element_shape=(2, ))
        sym_v = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                             'v',
                             ti.f32,
                             field_dim=1,
                             element_shape=(2, ))
        sym_C = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                             'C',
                             ti.f32,
                             field_dim=1,
                             element_shape=(2, 2))
        sym_J = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                             'J',
                             ti.f32,
                             field_dim=1,
                             element_shape=())
        sym_grid_v = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                  'grid_v',
                                  ti.f32,
                                  field_dim=2,
                                  element_shape=(2, ))
        sym_grid_m = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                  'grid_m',
                                  ti.f32,
                                  field_dim=2,
                                  element_shape=())
        sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                               'pos',
                               ti.f32,
                               field_dim=1,
                               element_shape=(3, ))

        g_init_builder = ti.graph.GraphBuilder()
        g_init_builder.dispatch(init_particles, sym_x, sym_v, sym_J)

        g_update_builder = ti.graph.GraphBuilder()
        substep = g_update_builder.create_sequential()

        substep.dispatch(substep_reset_grid, sym_grid_v, sym_grid_m)
        substep.dispatch(substep_p2g, sym_x, sym_v, sym_C, sym_J, sym_grid_v,
                         sym_grid_m)
        substep.dispatch(substep_update_grid_v, sym_grid_v, sym_grid_m)
        substep.dispatch(substep_g2p, sym_x, sym_v, sym_C, sym_J, sym_grid_v,
                         sym_pos)

        for i in range(N_ITER):
            g_update_builder.append(substep)

        g_init = g_init_builder.compile()
        g_update = g_update_builder.compile()

        # GGUI only supports vec3 vertex so we need an extra `pos` here
        # This is not necessary if you're not going to render it using GGUI.
        # Let's keep this hack here so that the shaders serialized by this
        # script can be loaded and rendered in the provided script in taichi-aot-demo.
        pos = ti.Vector.ndarray(3, ti.f32, n_particles)
        x = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))
        v = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))

        C = ti.Matrix.ndarray(2, 2, ti.f32, shape=(n_particles))
        J = ti.ndarray(ti.f32, shape=(n_particles))
        grid_v = ti.Vector.ndarray(2, ti.f32, shape=(n_grid, n_grid))
        grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid))

        mod = ti.aot.Module(arch)
        mod.add_graph('init', g_init)
        mod.add_graph('update', g_update)
        mod.save(tmpdir, '')
    else:
        pos = ti.Vector.ndarray(3, ti.f32, n_particles)
        x = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))
        v = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))

        C = ti.Matrix.ndarray(2, 2, ti.f32, shape=(n_particles))
        J = ti.ndarray(ti.f32, shape=(n_particles))
        grid_v = ti.Vector.ndarray(2, ti.f32, shape=(n_grid, n_grid))
        grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid))

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
    compile_for_cgraph = args.cgraph
    if args.arch == "cpu":
        compile_mpm88(arch=ti.cpu, save_compute_graph=compile_for_cgraph)
    elif args.arch == "cuda":
        compile_mpm88(arch=ti.cuda, save_compute_graph=compile_for_cgraph)
    elif args.arch == "vulkan":
        compile_mpm88(arch=ti.vulkan, save_compute_graph=compile_for_cgraph)
    elif args.arch == "opengl":
        compile_mpm88(arch=ti.opengl, save_compute_graph=compile_for_cgraph)
    else:
        assert False
