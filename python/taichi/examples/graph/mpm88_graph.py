import argparse

import taichi as ti

ti.init(arch=ti.vulkan)
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
N_ITER = 500  # Use 500 to make speed diff more obvious


@ti.kernel
def substep_reset_grid(grid_v: ti.types.ndarray(ndim=2), grid_m: ti.types.ndarray(ndim=2)):
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0


@ti.kernel
def substep_p2g(
    x: ti.types.ndarray(ndim=1),
    v: ti.types.ndarray(ndim=1),
    C: ti.types.ndarray(ndim=1),
    J: ti.types.ndarray(ndim=1),
    grid_v: ti.types.ndarray(ndim=2),
    grid_m: ti.types.ndarray(ndim=2),
):
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


@ti.kernel
def substep_update_grid_v(grid_v: ti.types.ndarray(ndim=2), grid_m: ti.types.ndarray(ndim=2)):
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


@ti.kernel
def substep_g2p(
    x: ti.types.ndarray(ndim=1),
    v: ti.types.ndarray(ndim=1),
    C: ti.types.ndarray(ndim=1),
    J: ti.types.ndarray(ndim=1),
    grid_v: ti.types.ndarray(ndim=2),
):
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
def init_particles(
    x: ti.types.ndarray(ndim=1),
    v: ti.types.ndarray(ndim=1),
    J: ti.types.ndarray(ndim=1),
):
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1


F_x = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))
F_v = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))

F_C = ti.Matrix.ndarray(2, 2, ti.f32, shape=(n_particles))
F_J = ti.ndarray(ti.f32, shape=(n_particles))
F_grid_v = ti.Vector.ndarray(2, ti.f32, shape=(n_grid, n_grid))
F_grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args, unknown = parser.parse_known_args()

    if not args.baseline:
        print("running in graph mode")
        # Build graph
        sym_x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", dtype=ti.math.vec2, ndim=1)
        sym_v = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "v", dtype=ti.math.vec2, ndim=1)
        sym_C = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "C", dtype=ti.math.mat2, ndim=1)
        sym_J = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "J", ti.f32, ndim=1)
        sym_grid_v = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "grid_v", dtype=ti.math.vec2, ndim=2)
        sym_grid_m = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "grid_m", dtype=ti.f32, ndim=2)
        g_init_builder = ti.graph.GraphBuilder()
        g_init_builder.dispatch(init_particles, sym_x, sym_v, sym_J)

        g_update_builder = ti.graph.GraphBuilder()
        substep = g_update_builder.create_sequential()

        substep.dispatch(substep_reset_grid, sym_grid_v, sym_grid_m)
        substep.dispatch(substep_p2g, sym_x, sym_v, sym_C, sym_J, sym_grid_v, sym_grid_m)
        substep.dispatch(substep_update_grid_v, sym_grid_v, sym_grid_m)
        substep.dispatch(substep_g2p, sym_x, sym_v, sym_C, sym_J, sym_grid_v)

        for i in range(N_ITER):
            g_update_builder.append(substep)

        # Compile
        g_init = g_init_builder.compile()
        g_update = g_update_builder.compile()

        # Run
        g_init.run({"x": F_x, "v": F_v, "J": F_J})

        gui = ti.GUI("MPM88")
        while gui.running:
            g_update.run(
                {
                    "x": F_x,
                    "v": F_v,
                    "C": F_C,
                    "J": F_J,
                    "grid_v": F_grid_v,
                    "grid_m": F_grid_m,
                }
            )
            gui.clear(0x112F41)
            gui.circles(
                F_x.to_numpy(),  # false+, pylint: disable=no-member
                radius=1.5,
                color=0x068587,
            )
            gui.show()
    else:
        init_particles(F_x, F_v, F_J)
        gui = ti.GUI("MPM88")
        while gui.running and not gui.get_event(gui.ESCAPE):
            for s in range(N_ITER):
                substep_reset_grid(F_grid_v, F_grid_m)
                substep_p2g(F_x, F_v, F_C, F_J, F_grid_v, F_grid_m)
                substep_update_grid_v(F_grid_v, F_grid_m)
                substep_g2p(F_x, F_v, F_C, F_J, F_grid_v)
            gui.clear(0x112F41)
            gui.circles(
                F_x.to_numpy(),  # false+, pylint: disable=no-member
                radius=1.5,
                color=0x068587,
            )
            gui.show()


if __name__ == "__main__":
    main()
